# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

H, EPS, WORLD, M_CAP, STAGES = 4096, 1e-6, 8, 28, 3
K = 2 * H
N = H // WORLD
ROWS = (1, 2, 3, 8, 9, 13, 14, 16)
SENTINEL = -32768

gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


def _tactics(rows):
    from sgl_nvidia_mega.qwen3_5.draft.fc_tp import LOWM_CANDIDATES
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import default_tactic

    tacs = [t for t in LOWM_CANDIDATES if t[3] == 0 or rows <= 8]
    d = tuple(default_tactic(rows, N, K, fp8=False))
    return tacs + ([d] if d not in tacs else [])


def _spec(rows, tac, push=0, release="wait"):
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import PUSH_RELEASES
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    # the stored launch (released after its wait); a push form in either raced release
    pdl = dict(PUSH_RELEASES[release])
    s = trt_lowm.spec(
        m=rows,
        k=K,
        n_bf16=N,
        fp8_act="bf16",
        tactic_bf16=tac,
        out_stride=H,
        push=push,
        **pdl,
    )
    try:
        trt_lowm.prepare(s)
    except Exception as exc:
        return None, str(exc)
    return s, ""


def _weights(device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    fc = (K**-0.5 * torch.randn(H, K, device=device, generator=g)).bfloat16()
    inn = (0.1 * torch.randn(H, device=device, generator=g)).bfloat16()
    return fc, inn


def _input(rows, device, seed):
    g = torch.Generator(device=device).manual_seed(100 + seed)
    x = torch.randn(rows, K, device=device, generator=g).bfloat16()
    x[rows - 1] = 0
    x[rows - 1, 0] = (
        1e-39  # a BF16 subnormal: products below the smallest BF16 subnormal round to +-0
    )
    return x


def _sanitized(t):
    v = t.view(torch.int16)
    return torch.where(v == SENTINEL, torch.zeros_like(v), v)


class _Box:
    """A decode-hop mailbox [stages][world][m_cap][H] (int16 view) and control block on this GPU."""

    def __init__(self, device, uc=None, mc=None):
        self.uc_t = None
        if uc is None:
            self.uc_t = torch.full(
                (STAGES, WORLD, M_CAP, H), SENTINEL, dtype=torch.int16, device=device
            )
            uc = self.uc_t.data_ptr()
        self.uc, self.mc = uc, mc if mc is not None else uc
        self.ctrl = torch.zeros(4, dtype=torch.int32, device=device)

    def view(self):
        return self.uc_t

    def desc(self, rank, base=None):
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        return trt_lowm.push_descriptor(
            mailbox=self.mc if base is None else base,
            ctrl=self.ctrl.data_ptr(),
            rank=rank,
            world=WORLD,
            m_cap=M_CAP,
            row_elems=H,
            col0=rank * N,
            device=self.ctrl.device,
        )

    def site(self, rank=0):
        from sgl_nvidia_mega.qwen3_5.sm103.hop import Site

        return Site(self.mc, self.uc, self.ctrl.data_ptr(), rank, WORLD, M_CAP)


def _hop(box, device):
    import types

    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop

    return types.SimpleNamespace(
        site=box.site(0),
        world=WORLD,
        device=device,
        max_m=M_CAP,
        supports=lambda m: 1 <= m <= M_CAP,
        kernels=decode_hop._kernels(device.index, int(30e9)),
    )


@gpu
@pytest.mark.parametrize("rows", ROWS)
def test_the_push_writes_the_stored_words_into_its_slot_only(rows):
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    device = torch.device("cuda", torch.cuda.current_device())
    fc, _ = _weights(device)
    x = _input(rows, device, rows)
    box = _Box(device)
    checked = 0
    for tac in _tactics(rows):
        s, why = _spec(rows, tac)
        p, why_p = _spec(rows, tac, push=2)
        if s is None or p is None:
            print(f"rows {rows} tactic {tac}: refused ({why or why_p})")
            continue
        for rank in (0, 3, 7):
            w = fc[rank * N : (rank + 1) * N]
            block = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
            trt_lowm.run(s, out=block[:, rank * N : (rank + 1) * N], x16=x, w_bf16=w)
            for stage in range(STAGES):
                box.view().fill_(SENTINEL)
                box.ctrl.zero_()
                box.ctrl[0] = stage
                untouched = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
                trt_lowm.run(
                    p,
                    out=untouched[:, rank * N : (rank + 1) * N],
                    x16=x,
                    w_bf16=w,
                    push=box.desc(rank),
                )
                torch.cuda.synchronize()
                got = box.view()[stage, rank, :rows, rank * N : (rank + 1) * N]
                want = _sanitized(block[:, rank * N : (rank + 1) * N])
                assert torch.equal(got, want), (rows, tac, rank, stage)
                rest = box.view().clone()
                rest[stage, rank, :rows, rank * N : (rank + 1) * N] = SENTINEL
                assert bool((rest == SENTINEL).all()), (
                    rows,
                    tac,
                    rank,
                    stage,
                    "wrote outside its slot",
                )
                assert not untouched.any(), "the push form stored its window"
                checked += 1
        negz = int((block.view(torch.int16) == SENTINEL).sum())
        print(
            f"PUSH-WORDS rows {rows} tactic {tac}: ranks 0/3/7 x stages 0-2 equal; -0.0 words mapped {negz}"
        )
    assert checked


def _consume(box, device, rows, inn, cols):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    zero = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
    partial = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
    return kernels.draft_all_reduce_rms_norm(
        _hop(box, device),
        local_contribution=partial,
        residual=zero,
        gamma=inn,
        eps=EPS,
        weight_bias=1.0,
        pushed=True,
        cols=cols,
    )


@gpu
@pytest.mark.parametrize("rows", ROWS)
def test_the_owner_slot_consume_is_the_full_consume_bit_for_bit(rows):
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    device = torch.device("cuda", torch.cuda.current_device())
    torch.zeros(1, device=device)
    kernels.build(device.index, int(30e9))
    fc, inn = _weights(device, seed=1)
    x = _input(rows, device, 7 * rows)
    tac = _tactics(rows)[0]
    s, _ = _spec(rows, tac)
    p, _ = _spec(rows, tac, push=2)
    assert s is not None and p is not None, tac
    box = _Box(device)
    blocks = []
    for r in range(WORLD):
        b = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
        trt_lowm.run(
            s, out=b[:, r * N : (r + 1) * N], x16=x, w_bf16=fc[r * N : (r + 1) * N]
        )
        blocks.append(b)
    want_res = torch.cat(
        [blocks[r][:, r * N : (r + 1) * N] for r in range(WORLD)], dim=1
    )
    for stage in range(STAGES):
        box.view().fill_(SENTINEL)
        box.ctrl.zero_()
        box.ctrl[0] = stage
        for r in range(WORLD):
            box.view()[stage, r, :rows] = _sanitized(blocks[r])
        norm_a, res_a = _consume(box, device, rows, inn, cols=False)
        torch.cuda.synchronize()
        assert bool((box.view() == SENTINEL).all()), "full consume left a slot armed"
        assert int(box.ctrl[0]) == (stage + 1) % STAGES and int(box.ctrl[2]) == 1
        # push arm: eight push GEMMs (each its columns), then the owner-slot consume
        box.ctrl.zero_()
        box.ctrl[0] = stage
        for r in range(WORLD):
            dummy = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
            trt_lowm.run(
                p,
                out=dummy[:, r * N : (r + 1) * N],
                x16=x,
                w_bf16=fc[r * N : (r + 1) * N],
                push=box.desc(r),
            )
        norm_b, res_b = _consume(box, device, rows, inn, cols=True)
        torch.cuda.synchronize()
        assert bool((box.view() == SENTINEL).all()), (
            "owner-slot consume left a slot written"
        )
        assert int(box.ctrl[0]) == (stage + 1) % STAGES and int(box.ctrl[2]) == 1
        assert torch.equal(res_a.view(torch.int16), res_b.view(torch.int16)), (
            rows,
            stage,
            "residual",
        )
        assert torch.equal(norm_a.view(torch.int16), norm_b.view(torch.int16)), (
            rows,
            stage,
            "norm",
        )
        assert torch.equal(_sanitized(res_b), _sanitized(want_res)), (
            rows,
            stage,
            "not the concatenation",
        )
    print(
        f"OWNER-CONSUME rows {rows} tactic {tac}: residual + norm bitwise = full consume, stages 0-2, re-armed"
    )


def _one_device_multicast(nbytes, device_index):
    from sgl_nvidia_mega.common import jit

    cu = jit.driver()

    def ok(r, what):
        return jit.check(r, what)

    (dev,) = ok(cu.cuDeviceGet(device_index), "cuDeviceGet")
    (sup,) = ok(
        cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev
        ),
        "cuDeviceGetAttribute",
    )
    if not sup:
        raise RuntimeError("no multicast support")
    prop = cu.CUmulticastObjectProp()
    prop.numDevices = 1
    prop.handleTypes = 0
    prop.flags = 0
    prop.size = nbytes
    (gran,) = ok(
        cu.cuMulticastGetGranularity(
            prop, cu.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED
        ),
        "cuMulticastGetGranularity",
    )
    size = -(-nbytes // gran) * gran
    prop.size = size
    (mc,) = ok(cu.cuMulticastCreate(prop), "cuMulticastCreate")
    ok(cu.cuMulticastAddDevice(mc, dev), "cuMulticastAddDevice")
    aprop = cu.CUmemAllocationProp()
    aprop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    aprop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    aprop.location.id = device_index
    aprop.requestedHandleTypes = 0
    (agran,) = ok(
        cu.cuMemGetAllocationGranularity(
            aprop,
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    size = -(-size // max(gran, agran)) * max(gran, agran)
    (mem,) = ok(cu.cuMemCreate(size, aprop, 0), "cuMemCreate")
    ok(cu.cuMulticastBindMem(mc, 0, mem, 0, size, 0), "cuMulticastBindMem")
    acc = cu.CUmemAccessDesc()
    acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = device_index
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    (uc_va,) = ok(
        cu.cuMemAddressReserve(size, max(gran, agran), 0, 0), "cuMemAddressReserve"
    )
    ok(cu.cuMemMap(uc_va, size, 0, mem, 0), "cuMemMap uc")
    ok(cu.cuMemSetAccess(uc_va, size, [acc], 1), "cuMemSetAccess uc")
    (mc_va,) = ok(
        cu.cuMemAddressReserve(size, max(gran, agran), 0, 0), "cuMemAddressReserve"
    )
    ok(cu.cuMemMap(mc_va, size, 0, mc, 0), "cuMemMap mc")
    ok(cu.cuMemSetAccess(mc_va, size, [acc], 1), "cuMemSetAccess mc")
    return int(uc_va), int(mc_va), (mc, mem, size)


class _McBox(_Box):
    def __init__(self, device):
        nbytes = STAGES * WORLD * M_CAP * H * 2
        uc, mc, self._keep = _one_device_multicast(nbytes, device.index)
        super().__init__(device, uc=uc, mc=mc)
        self.n = STAGES * WORLD * M_CAP * H
        self.fill()

    def fill(self):
        from sgl_nvidia_mega.common import jit

        cu = jit.driver()
        torch.cuda.synchronize()
        jit.check(cu.cuMemsetD16(self.uc, 0x8000, self.n), "cuMemsetD16")
        torch.cuda.synchronize()

    def read(self):
        from sgl_nvidia_mega.common import jit

        cu = jit.driver()
        out = torch.empty((STAGES, WORLD, M_CAP, H), dtype=torch.int16, device="cuda")
        torch.cuda.synchronize()
        jit.check(cu.cuMemcpyDtoD(out.data_ptr(), self.uc, self.n * 2), "cuMemcpyDtoD")
        torch.cuda.synchronize()
        return out


def _mc_box(device):
    try:
        return _McBox(device)
    except Exception as exc:
        pytest.skip(f"no one-device multicast mapping: {exc}")


@gpu
@pytest.mark.parametrize("rows", (2, 13, 16))
def test_the_multicast_push_lands_the_same_words(rows):
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    device = torch.device("cuda", torch.cuda.current_device())
    box = _mc_box(device)
    fc, _ = _weights(device)
    x = _input(rows, device, rows)
    for tac in _tactics(rows):
        s, _ = _spec(rows, tac)
        p, _ = _spec(rows, tac, push=1)
        if s is None or p is None:
            continue
        rank, stage = 5, 2
        w = fc[rank * N : (rank + 1) * N]
        block = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
        trt_lowm.run(s, out=block[:, rank * N : (rank + 1) * N], x16=x, w_bf16=w)
        box.fill()
        box.ctrl.zero_()
        box.ctrl[0] = stage
        trt_lowm.run(p, out=block[:, :N].clone(), x16=x, w_bf16=w, push=box.desc(rank))
        got = box.read()
        assert torch.equal(
            got[stage, rank, :rows, rank * N : (rank + 1) * N],
            _sanitized(block[:, rank * N : (rank + 1) * N]),
        ), (rows, tac)
        got[stage, rank, :rows, rank * N : (rank + 1) * N] = SENTINEL
        assert bool((got == SENTINEL).all()), (rows, tac)
    print(
        f"MULTICAST-PUSH rows {rows}: multimem.st words equal (rank 5, stage 2), nothing else written"
    )


class _HopOnBox:
    def __init__(self, box, device, rank):
        from sgl_nvidia_mega.qwen3_5.draft.hop import DraftHop

        self.box, self.device, self.rank = box, device, rank
        self.hop = _hop(box, device)
        self.hop.site = box.site(rank)
        self.draft = DraftHop.__new__(DraftHop)
        self.draft._pushed = None

    def mark_pushed(self, partial):
        self.draft.mark_pushed(partial)

    def _publish_if_needed(self, partial):
        if self.draft._take_pushed(partial):
            return True
        stage = int(self.box.ctrl[0])
        rows = int(partial.shape[0])
        self.box.view()[stage, self.rank, :rows] = _sanitized(partial)
        return False

    def layer_input(self, partial, residual, norm, cols=False):
        from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

        self._publish_if_needed(partial)
        return kernels.draft_all_reduce_rms_norm(
            self.hop,
            local_contribution=partial,
            residual=residual,
            gamma=norm.weight,
            eps=norm.variance_epsilon,
            weight_bias=1.0,
            pushed=True,
            cols=cols,
        )

    def ffn_input(self, partial, residual, norm):
        from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

        self._publish_if_needed(partial)
        # DraftHop.ffn_input's consume: add-norm, multiplier 1 + the raw weight in fp32
        return kernels.draft_all_reduce_rms_norm(
            self.hop,
            local_contribution=partial,
            residual=residual,
            gamma=norm.weight,
            eps=norm.variance_epsilon,
            weight_bias=1.0,
            pushed=True,
        )


class _N:
    def __init__(self, weight):
        self.weight = weight
        self.gemma_weight = torch.add(weight.float(), 1.0).bfloat16()
        self.variance_epsilon = EPS


@gpu
def test_fc_tp_builds_its_push_and_runs_it_bit_for_bit(monkeypatch):
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.draft.fc_tp import LOWM_CANDIDATES, FcTp
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import HeadLowM
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda obj, src=0, group=None: None
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.zeros(1, device=device)
    kernels.build(device.index, int(30e9))
    fc, inn = _weights(device, seed=12)
    rank = 6
    box = _Box(device)
    hop = _HopOnBox(box, device, rank)
    w_slice = fc[rank * N : (rank + 1) * N]
    rows_all = (2, 8, 13, 16)
    lowm = HeadLowM(
        w_slice,
        rows_all,
        cpu_group=None,
        rank=rank,
        max_rows=16,
        candidates=LOWM_CANDIDATES,
        label="test fc",
        cold_copies=8,
        out_stride=H,
    )
    g = torch.Generator(device=device).manual_seed(5)
    table = (0.02 * torch.randn(4096, H, device=device, generator=g)).bfloat16()
    emb, hid = (
        _N((0.1 * torch.randn(H, device=device, generator=g)).bfloat16()),
        _N((0.1 * torch.randn(H, device=device, generator=g)).bfloat16()),
    )
    front = FcTp(
        weight_slice=w_slice,
        table=table,
        norm_embedding=emb,
        norm_hidden=hid,
        input_norm=_N(inn),
        rank=rank,
        world=WORLD,
        hop=hop,
        mnnvl=None,
        lowm=lowm,
        push=2,
        cpu_group=None,
    )
    print(
        f"FCTP-PUSH-BUILT store tactics { {m: (s.k_warps, s.ksplit) for m, s in lowm.specs.items()} } push "
        f"{ {m: (s.k_warps, s.n_tiles_bf16, s.ksplit, s.swap_ab, s.release_after_wait) for m, s in front.push_specs.items()} }"
    )
    assert sorted(front.push_specs) == sorted(lowm.specs)
    from sgl_nvidia_mega.qwen3_5.draft.fc_input import mtp_fused_input

    for i, rows in enumerate(sorted(front.push_specs)):
        ids = torch.randint(0, 4096, (rows,), device=device, generator=g)
        hidden = torch.randn(rows, H, device=device, generator=g).bfloat16()
        full = mtp_fused_input(
            ids, table, hidden, emb.weight, hid.weight, EPS, EPS, 1, 0
        )
        stored = front.push_specs[rows]._replace(push=0)
        trt_lowm.prepare(stored)
        wins = []
        for r in range(WORLD):
            b = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
            trt_lowm.run(
                stored,
                out=b[:, r * N : (r + 1) * N],
                x16=full,
                w_bf16=fc[r * N : (r + 1) * N],
            )
            wins.append(b)
        stage = i % STAGES
        # the reference: eight full blocks published, the full add-norm consume
        box.view().fill_(SENTINEL)
        box.ctrl.zero_()
        box.ctrl[0] = stage
        for r in range(WORLD):
            box.view()[stage, r, :rows] = _sanitized(wins[r])
        zero = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
        norm_ref, res_ref = kernels.draft_all_reduce_rms_norm(
            hop.hop,
            local_contribution=zero,
            residual=zero,
            gamma=inn,
            eps=EPS,
            weight_bias=1.0,
            pushed=True,
        )
        box.view().fill_(SENTINEL)
        box.ctrl.zero_()
        box.ctrl[0] = stage
        for r in range(WORLD):
            if r != rank:
                box.view()[stage, r, :rows, r * N : (r + 1) * N] = _sanitized(
                    wins[r][:, r * N : (r + 1) * N]
                )
        front.block.zero_()
        residual, normed = front(ids, hidden)
        torch.cuda.synchronize()
        assert not front.block.any(), "the push front wrote its block"
        assert torch.equal(residual.view(torch.int16), res_ref.view(torch.int16)), rows
        assert torch.equal(normed.view(torch.int16), norm_ref.view(torch.int16)), rows
        assert (
            bool((box.view() == SENTINEL).all())
            and int(box.ctrl[0]) == (stage + 1) % STAGES
        )
    print(
        f"FCTP-PUSH-RUN rows {sorted(front.push_specs)}: residual + norm bitwise = the full consume of the windows"
    )


KO = 1024


def _oproj_tactics(rows):
    from sgl_nvidia_mega.qwen3_5.draft.oproj_push import CANDIDATES
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import default_tactic

    tacs = [t for t in CANDIDATES if t[3] == 0 or rows <= 8]
    d = tuple(default_tactic(rows, H, KO, fp8=False))
    return tacs + ([d] if d not in tacs else [])


def _oproj_spec(rows, tac, push=0):
    from sgl_nvidia_mega.qwen3_5.draft.head_lowm import PDL
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    s = trt_lowm.spec(
        m=rows, k=KO, n_bf16=H, fp8_act="bf16", tactic_bf16=tac, push=push, **PDL
    )
    try:
        trt_lowm.prepare(s)
    except Exception as exc:
        return None, str(exc)
    return s, ""


def _oproj_weights(device, seed):
    path = os.environ.get("OPROJ_CKPT") or os.environ.get("FC_TP_CKPT")
    if path:
        import json

        from safetensors import safe_open

        names = json.load(open(os.path.join(path, "model.safetensors.index.json")))[
            "weight_map"
        ]
        hits = sorted(
            n
            for n in names
            if n.startswith("mtp.") and n.endswith("self_attn.o_proj.weight")
        )
        if hits:
            with safe_open(os.path.join(path, names[hits[0]]), "pt", device="cpu") as f:
                w = f.get_tensor(hits[0]).to(device=device, dtype=torch.bfloat16)
            print(f"o_proj weights from {path}: {hits[0]} {tuple(w.shape)}")
            return [w[:, r * KO : (r + 1) * KO].contiguous() for r in range(WORLD)]
    g = torch.Generator(device=device).manual_seed(seed)
    return [
        (KO**-0.5 * torch.randn(H, KO, device=device, generator=g)).bfloat16()
        for _ in range(WORLD)
    ]


@gpu
@pytest.mark.parametrize("rows", (2, 8, 13, 16))
def test_the_oproj_push_writes_the_stored_words(rows):
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    device = torch.device("cuda", torch.cuda.current_device())
    w = _oproj_weights(device, seed=5)[2]
    g = torch.Generator(device=device).manual_seed(rows)
    o = torch.randn(rows, KO, device=device, generator=g).bfloat16()
    box = _Box(device)
    for tac in _oproj_tactics(rows):
        s, why = _oproj_spec(rows, tac)
        p, _ = _oproj_spec(rows, tac, push=2)
        if s is None or p is None:
            print(f"o_proj rows {rows} tactic {tac}: refused ({why})")
            continue
        out = torch.empty(rows, H, dtype=torch.bfloat16, device=device)
        trt_lowm.run(s, out=out, x16=o, w_bf16=w)
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm as T

        desc = T.push_descriptor(
            mailbox=box.mc,
            ctrl=box.ctrl.data_ptr(),
            rank=2,
            world=WORLD,
            m_cap=M_CAP,
            row_elems=H,
            col0=0,
            device=device,
        )
        for stage in range(STAGES):
            box.view().fill_(SENTINEL)
            box.ctrl.zero_()
            box.ctrl[0] = stage
            trt_lowm.run(
                p,
                out=torch.empty(rows, H, dtype=torch.bfloat16, device=device),
                x16=o,
                w_bf16=w,
                push=desc,
            )
            torch.cuda.synchronize()
            assert torch.equal(box.view()[stage, 2, :rows], _sanitized(out)), (
                rows,
                tac,
                stage,
            )
            rest = box.view().clone()
            rest[stage, 2, :rows] = SENTINEL
            assert bool((rest == SENTINEL).all()), (rows, tac, stage)
        print(
            f"OPROJ-PUSH-WORDS rows {rows} tactic {tac}: the stored partial's words, stages 0-2"
        )


@gpu
def test_race_push_runs_its_chain_and_picks_every_width(monkeypatch):
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.draft import head_lowm
    from sgl_nvidia_mega.qwen3_5.draft.fc_tp import LOWM_CANDIDATES
    from sgl_nvidia_mega.qwen3_5.draft.oproj_push import CANDIDATES
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda obj, src=0, group=None: None
    )
    device = torch.device("cuda", torch.cuda.current_device())
    torch.zeros(1, device=device)
    kernels.build(device.index, int(30e9))
    fc, inn = _weights(device, seed=4)
    picks = head_lowm.race_push(
        fc[3 * N : 4 * N],
        (2, 8, 13, 16),
        candidates=LOWM_CANDIDATES,
        cpu_group=None,
        rank=3,
        world=WORLD,
        label="test fc",
        consume=dict(gamma=inn, eps=EPS, weight_bias=1.0, cols=True),
        out_stride=H,
        col0=3 * N,
        cold_copies=8,
    )
    print(f"RACE-PUSH fc picks {picks}")
    assert sorted(picks) == [2, 8, 13, 16] and all(
        r in head_lowm.PUSH_RELEASES for _, r in picks.values()
    )
    wo = _oproj_weights(device, seed=6)[0]
    picks = head_lowm.race_push(
        wo,
        (2, 13),
        candidates=CANDIDATES,
        cpu_group=None,
        rank=0,
        world=WORLD,
        label="test o_proj",
        consume=dict(gamma=inn, eps=EPS, weight_bias=1.0),
        cold_copies=8,
    )
    print(f"RACE-PUSH o_proj picks {picks}")
    assert sorted(picks) == [2, 13]


@gpu
def test_oproj_push_builds_and_its_consume_is_the_stored_partials(monkeypatch):
    import types

    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5 import runtime
    from sgl_nvidia_mega.qwen3_5.draft import oproj_push
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda obj, src=0, group=None: None
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "measured_option", lambda name: True)
    monkeypatch.setattr(runtime, "slot_supported", lambda name, why, **_: not why)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.zeros(1, device=device)
    kernels.build(device.index, int(30e9))
    rank = 1
    ws = _oproj_weights(device, seed=13)
    box = _Box(device)
    hop = _HopOnBox(box, device, rank)
    g = torch.Generator(device=device).manual_seed(6)
    post = _N((0.1 * torch.randn(H, device=device, generator=g)).bfloat16())
    parts = types.SimpleNamespace(
        hop=hop,
        layer=types.SimpleNamespace(
            o_proj=types.SimpleNamespace(weight=ws[rank], bias=None),
            post_attention_layernorm=post,
        ),
        binding=types.SimpleNamespace(tp_size=WORLD, tp_rank=rank, tp_cpu_group=None),
    )
    monkeypatch.setattr(oproj_push, "PUSH_MODE", 2)
    built = oproj_push.maybe_build(parts)
    assert built is not None and sorted(built.specs) == list(range(2, 17))
    for rows in (2, 13, 16):
        o = torch.randn(rows, KO, device=device, generator=g).bfloat16()
        residual = torch.randn(rows, H, device=device, generator=g).bfloat16()
        stored = built.specs[rows]._replace(push=0)
        trt_lowm.prepare(stored)
        parts_out = []
        for r in range(WORLD):
            out = torch.empty(rows, H, dtype=torch.bfloat16, device=device)
            trt_lowm.run(stored, out=out, x16=o, w_bf16=ws[r])
            parts_out.append(out)
        box.view().fill_(SENTINEL)
        box.ctrl.zero_()
        for r in range(WORLD):
            if r != rank:
                box.view()[0, r, :rows] = _sanitized(parts_out[r])
        ref = hop.ffn_input(parts_out[rank], residual, post)
        box.view().fill_(SENTINEL)
        box.ctrl.zero_()
        for r in range(WORLD):
            if r != rank:
                box.view()[0, r, :rows] = _sanitized(parts_out[r])
        pushed = built(o)
        got = hop.ffn_input(pushed, residual, post)
        torch.cuda.synchronize()
        for a, b in zip(got, ref):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), rows
        assert bool((box.view() == SENTINEL).all())
    print(
        f"OPROJ-PUSH-RUN tactics { {m: (s.k_warps, s.n_tiles_bf16, s.ksplit, s.release_after_wait) for m, s in built.specs.items()} }"
    )
