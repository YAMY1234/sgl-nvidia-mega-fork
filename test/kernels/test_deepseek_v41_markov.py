# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The sharded DSpark Markov chain (sm103/markov.py) draws SGLang's MarkovSampleStep
tokens and keeps its step logits bit for bit when the bias products sum exactly: one
rank and four ranks (spawned, symmetric memory), 1 to 64 rows (direct exchange,
rank-local fold and the wide kernel), greedy, sampled and mixed temperatures, BF16 and
FP32 base logits, local-shard and full-vocab base logits, three launches (epoch tags and
both staging parities) and CUDA graph replay with the noise forked onto its side stream.
Rows at temperature 0 leave their step logits untouched. Long graph replays of mixed row
counts and launch counters crossing 2**32 exercise the epoch-tag reuse, and when one rank
never joins, its peers fault within the deadline (torchrun per case, since a faulted
rank leaves its CUDA context unusable).

Run with ``python test/kernels/test_deepseek_v41_markov.py``.
"""

import os
import socket
import subprocess
import sys
import textwrap
import time

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.deepseek_v41.sm103 import markov

VOCAB = 129_280  # DeepSeek-V4.1
RANK = 256
STEPS = 5
MAX_BS = 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = ((0.0,) * 8, (1.0,) * 8, (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1))
# Far above any launch, so only a missing peer reaches it.
DEADLINE_S = 20.0
# The fault-injection deadline.
FAULT_DEADLINE_S = 2.0
# Graph replays of the long test: LONG_LAUNCHES launches per replay.
LONG_LAUNCHES = 64
LONG_REPLAYS = 1100
LONG_ROWS = (8, 1, 3, 8, 2)
# The wide kernel (9-64 rows at TP4).
WIDE_BS = 64
WIDE_ROWS = (9, 16, 32, 64)
WIDE_GRAPH_ROWS = (64, 9, 1, 8, 33)
WIDE_REPLAYS = 14_000
# Per-CTA launch counters (markov.cu).
NARROW_COUNT_WORDS = slice(256, None)
WIDE_COUNT_WORDS = slice(1280, None)
ARRIVAL_WORDS = slice(128, 136)


def _gpu_ready() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in markov.SUPPORTED_CAPABILITIES
    )


needs_gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100/SM103 GPU")
needs_4_gpus = pytest.mark.skipif(
    not _gpu_ready() or torch.cuda.device_count() < 4, reason="needs 4 SM100/SM103 GPUs"
)


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """SGLang's MarkovSampleStep per step over the full vocab, as the unsharded draft runs it."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.float32,
            logits_out=None if cache is None else cache[: base.shape[0], step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _assert_cache(cache, want_cache, temperatures, rows: int, what) -> None:
    """Rows that keep logits match; temperature-0 rows stay zero."""
    keeps = temperatures[:rows] != 0
    assert torch.equal(cache[:rows][keeps], want_cache[:rows][keeps]), what
    assert not cache[:rows][~keeps].any(), what


def _make(group, rank, world, w1, w2, max_bs=MAX_BS, keeps_logits=True):
    return markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        deadline_s=DEADLINE_S,
        keeps_logits=keeps_logits,
    )


def _check_equal(device, group, rank: int, world: int, base_dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(group, rank, world, w1, w2)
    # A greedy draft's chain has no logit staging; it serves greedy blocks only.
    greedy_chain = _make(group, rank, world, w1, w2, keeps_logits=False)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    for rows in (1, 2, 4, 8):
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temps in TEMPERATURE_SETS:
            temperatures = torch.tensor(temps, device=device)
            for keep in (True, False):  # sampled draft vs greedy draft (no logits)
                want_cache = (
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for sampler in (chain,) if keep else (chain, greedy_chain):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                            sampler.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = (rows, temps, keep, local, sampler is chain)
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g)
    assert int(chain.control[ARRIVAL_WORDS].abs().sum()) == 0  # arrivals reset


def _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g):
    """CUDA graph: noise forked onto the side stream, replayed with new offsets."""
    rows = 2
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
    local = base[:, :, shard].contiguous()

    def forked_sample():
        chain.fork_noise(rows=rows, key=KEY, offset=offset)
        chain.join_noise()
        chain.sample(
            base_logits=local,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        forked_sample()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forked_sample()
    for step_offset in (5, 6, 1 << 33):
        offset.fill_(step_offset)
        graph.replay()
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        torch.cuda.synchronize()
        assert torch.equal(tokens[:rows], want), step_offset
        _assert_cache(cache, want_cache, temperatures, rows, step_offset)


def _check_long_replay(device, group, rank: int, world: int, seed: int):
    """Graphs of mixed row counts share the launch counters; > 2 tag periods of
    launches, then a fresh chain whose launch counters cross 2**32."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    base = (torch.randn(MAX_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (MAX_BS,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([11], dtype=torch.int64, device=device)
    want_cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    want_sampled = _reference(w1, w2, base, anchor, temperatures, offset, want_cache)
    want_greedy = _reference(w1, w2, base, anchor, temperatures, offset, None)

    chain = _make(group, rank, world, w1, w2)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full(
        (LONG_LAUNCHES, MAX_BS, STEPS), -7, dtype=torch.int64, device=device
    )

    def launches():
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            if keep:
                chain.launch_noise(rows=rows, key=KEY, offset=offset)
            chain.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache if keep else None,
                draft_tokens=tokens[i],
            )

    def check(what):
        torch.cuda.synchronize()
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            want = want_sampled if keep else want_greedy
            assert torch.equal(tokens[i, :rows], want[:rows]), (what, i)
        tokens.fill_(-7)

    launches()
    check("eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launches()
    for replay in range(LONG_REPLAYS):
        graph.replay()
        if replay % 100 == 99:
            check(replay)
    _assert_cache(cache, want_cache, temperatures, MAX_BS, "long")
    del graph

    # A fresh mailbox and counters three launches short of 2**32.
    chain = _make(group, rank, world, w1, w2)
    chain.control[NARROW_COUNT_WORDS] = -3
    for launch in range(6):
        for rows in (1, MAX_BS):  # the direct exchange and the fold
            out = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
            cache.zero_()
            chain.launch_noise(rows=rows, key=KEY, offset=offset)
            chain.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache,
                draft_tokens=out,
            )
            torch.cuda.synchronize()
            assert torch.equal(out[:rows], want_sampled[:rows]), ("wrap", launch, rows)
            _assert_cache(cache, want_cache, temperatures, rows, ("wrap", launch))


def _check_wide(device, group, rank: int, world: int, seed: int):
    """9-64 rows on the wide kernel: tokens and kept logits, a graph replay of mixed
    row counts across all three kernels, and a wide counter wrap."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS)
    assert chain.wide_rows == WIDE_BS and chain.max_rows == WIDE_BS
    greedy_chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS, keeps_logits=False)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([5], dtype=torch.int64, device=device)
    mixed = torch.tensor(TEMPERATURE_SETS[2] * (WIDE_BS // 8), device=device)
    for rows in WIDE_ROWS:
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
            device, torch.bfloat16
        )
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temperatures in (torch.zeros(WIDE_BS, device=device), mixed):
            for keep in (True, False):
                want_cache = (
                    torch.zeros(WIDE_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for sampler in (chain,) if keep else (chain, greedy_chain):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (WIDE_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                            sampler.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = ("wide", rows, keep, local, bool(temperatures.any()))
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    # One graph per row count; replays interleave the direct, narrow and wide kernels.
    base = (torch.randn(WIDE_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (WIDE_BS,), generator=g).to(device)
    want_cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    want = _reference(w1, w2, base, anchor, mixed, offset, want_cache)
    cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    outs = [
        torch.full((rows, STEPS), -7, dtype=torch.int64, device=device)
        for rows in WIDE_GRAPH_ROWS
    ]

    def launch(rows, out):
        chain.launch_noise(rows=rows, key=KEY, offset=offset)
        chain.sample(
            base_logits=local[:rows],
            anchor_tokens=anchor[:rows],
            temperatures=mixed,
            logits_cache=cache,
            draft_tokens=out,
        )

    graphs = []
    for rows, out in zip(WIDE_GRAPH_ROWS, outs):
        launch(rows, out)  # warm-up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(rows, out)
        graphs.append(graph)
    for replay in range(WIDE_REPLAYS):
        for graph in graphs:
            graph.replay()
        if replay % 2000 == 1999:
            torch.cuda.synchronize()
            for rows, out in zip(WIDE_GRAPH_ROWS, outs):
                assert torch.equal(out, want[:rows]), (replay, rows)
                out.fill_(-7)
    _assert_cache(cache, want_cache, mixed, WIDE_BS, "wide graph")
    del graphs

    # A fresh chain whose wide counters are three launches short of 2**32.
    chain = _make(group, rank, world, w1, w2, max_bs=WIDE_BS)
    chain.wide_control[WIDE_COUNT_WORDS] = -3
    for launch_index in range(6):
        out = torch.full((WIDE_BS, STEPS), -7, dtype=torch.int64, device=device)
        cache.zero_()
        launch(WIDE_BS, out)
        torch.cuda.synchronize()
        assert torch.equal(out, want), ("wide wrap", launch_index)
        _assert_cache(cache, want_cache, mixed, WIDE_BS, ("wide wrap", launch_index))


def test_unsupported_reason():
    def reason(**kw):
        args = dict(
            world=4,
            vocab_size=VOCAB,
            num_steps=STEPS,
            embed_weight=torch.empty(VOCAB, RANK, dtype=torch.bfloat16),
            head_weight=torch.empty(VOCAB, RANK, dtype=torch.bfloat16),
            cache_dtype=torch.float32,
            capability=(10, 3),
        )
        args.update(kw)
        return markov.unsupported_reason(**args)

    assert reason() is None
    assert reason(capability=(10, 0)) is None
    assert "tensor-parallel" in reason(world=3)
    assert "capability" in reason(capability=(9, 0))
    assert "draft positions" in reason(num_steps=9)
    assert "vocabulary" in reason(vocab_size=VOCAB + 16)
    assert "step-logits" in reason(cache_dtype=torch.float16)
    assert "embedding" in reason(embed_weight=torch.empty(VOCAB, 128))
    assert markov.narrow_row_limit(world=4, max_bs=64) == 8
    assert markov.narrow_row_limit(world=8, max_bs=64) == 4
    assert markov.narrow_row_limit(world=1, max_bs=2) == 2


def _nvrtc_available() -> bool:
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _nvrtc_available(), reason="needs NVRTC (cuda.bindings)")
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_every_variant_compiles(arch, monkeypatch):
    if not torch.cuda.is_available():  # a CPU node: no stream can be capturing
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for steps in (1, STEPS, markov.MAX_STEPS):
        for base_dtype in markov.LOGIT_DTYPES:
            for cache_dtype in markov.LOGIT_DTYPES:
                for direct in (True, False):
                    cubin = markov.compile_cubin(
                        steps=steps,
                        base_dtype=base_dtype,
                        cache_dtype=cache_dtype,
                        direct=direct,
                        arch=arch,
                    )
                    assert markov.NARROW_SYMBOL.encode() in cubin
                    assert (markov.WIDE_SYMBOL.encode() in cubin) == (not direct)


@needs_gpu
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_one_rank_matches_markov_sample_step(base_dtype):
    _check_equal(torch.device("cuda"), None, 0, 1, base_dtype, seed=1)


@needs_gpu
def test_one_rank_long_replay_and_counter_wrap():
    _check_long_replay(torch.device("cuda"), None, 0, 1, seed=3)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _four_rank_worker(rank: int, world: int, port: int, mode: str) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    try:
        if mode == "long":
            _check_long_replay(device, group, rank, world, seed=9)
        elif mode == "wide":
            _check_wide(device, group, rank, world, seed=11)
        else:
            for base_dtype in (torch.bfloat16, torch.float32):
                _check_equal(device, group, rank, world, base_dtype, seed=7)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@needs_4_gpus
@pytest.mark.parametrize("mode", ["equal", "long", "wide"])
def test_four_ranks_match_markov_sample_step(mode):
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port(), mode), nprocs=4, join=True
    )


FAULT_WORKER = textwrap.dedent(
    """
    import os, sys, time
    import torch
    import torch.distributed as dist
    from sgl_nvidia_mega.deepseek_v41.sm103 import markov

    rows, deadline = int(sys.argv[1]), float(sys.argv[2])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("cpu:gloo,cuda:nccl")
    group = dist.new_group(list(range(world)), backend="gloo")
    vocab, steps = 129_280, 5
    w = torch.zeros(vocab, 256, dtype=torch.bfloat16, device=dev)
    chain = markov.ShardedMarkov(embed_weight=w, head_weight=w, vocab_size=vocab,
        num_steps=steps, cache_dtype=torch.float32, group=group, rank=rank, world=world,
        max_bs=64, deadline_s=deadline, keeps_logits=False)
    base = torch.randn(rows, steps, vocab // world, device=dev)
    tokens = torch.zeros(rows, steps, dtype=torch.int64, device=dev)
    def run():
        chain.sample(base_logits=base, anchor_tokens=torch.zeros(rows, dtype=torch.int32, device=dev),
                     temperatures=torch.zeros(64, device=dev), logits_cache=None,
                     draft_tokens=tokens)
    run(); torch.cuda.synchronize()  # every rank joins once: compiled and healthy
    dist.barrier(group=group)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * deadline + 20)  # never joins the second launch
        os._exit(0)
    t0 = time.time()
    try:
        run(); torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}", flush=True)
    os._exit(0)
    """
)


@needs_4_gpus
@pytest.mark.parametrize("rows", [1, 4, 16])  # direct exchange, fold, wide kernel
def test_a_missing_rank_faults_its_peers_in_time(rows, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(FAULT_WORKER)
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
         f"--master_port={_free_port()}", str(script), str(rows), str(FAULT_DEADLINE_S)],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    log = out.stdout + out.stderr
    assert log.count("HEALTHY") == 4, log[-4000:]
    # the waiting ranks fault (the kernel's store to 0x0 surfaces as a CUDA error);
    # none returns as if the launch had completed
    assert "NO-FAULT" not in log, log[-4000:]
    faulted = [line for line in log.splitlines() if "FAULTED after" in line]
    assert len(faulted) == 3, log[-4000:]
    print("\n".join(faulted))
    for line in faulted:
        assert float(line.split("after ")[1].split(" s")[0]) < 4 * FAULT_DEADLINE_S + 10
    assert time.time() - t0 < 300


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
