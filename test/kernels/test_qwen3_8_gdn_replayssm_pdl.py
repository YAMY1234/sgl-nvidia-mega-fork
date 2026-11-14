# SPDX-License-Identifier: Apache-2.0
"""Slot gdn_native_pdl's ReplaySSM verify (``gdn_front/replayssm_pdl.py``) on one
SM100-family GPU, at Qwen3.8's TP1 GatedDeltaNet shape, the output and this step's ring
entries bit for bit:

- without the dependent launch, against SGLang's ``gdn_replayssm_spec_decode`` verify
  launch (sgl-project/sglang#46185's arithmetic), with the ring cursors as their own
  arrays and as columns of one entry per replay row (#49719's request sub-pool); with
  it, waiting at the q / k / v reads or before the gates, the bits without it;
- with the cursors as columns of one entry, the bits with them as their own arrays.
"""

import inspect

import pytest
import torch

H, K, HV, V = 16, 128, 48, 128  # Qwen3.8 at TP1
D = 2 * H * K + HV * V
DRAFT, SLOTS = 4, 96


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _columns(write_pos, cache_base, is_flush):
    """The cursors as columns of one entry per replay row (int32 write_pos / cache_base,
    int8 flush flags); the other columns hold values a unit-stride read would trip on."""
    entry = torch.full((len(write_pos), 3), 7, dtype=torch.int32, device="cuda")
    entry[:, 0], entry[:, 2] = write_pos, cache_base
    flags = torch.ones(len(is_flush), 4, dtype=torch.int8, device="cuda")
    flags[:, 1] = is_flush
    return entry[:, 0], entry[:, 2], flags[:, 1]


def _operands(seed, bs, ring_len):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = bs * DRAFT

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=g).to(dtype)

    qkv = rand(tokens, D)  # strided q / k / v, as the conv's split leaves them
    ba = rand(tokens, 2 * HV)
    gates = dict(
        q=qkv[:, : H * K].view(tokens, H, K),
        k=qkv[:, H * K : 2 * H * K].view(tokens, H, K),
        v=qkv[:, 2 * H * K :].view(tokens, HV, V),
        a=ba[:, HV:],
        b=ba[:, :HV],
        A_log=rand(HV, dtype=torch.float32),
        dt_bias=rand(HV, dtype=torch.float32),
    )
    state = rand(SLOTS, HV, V, K)
    ring = [
        rand(SLOTS, HV, ring_len, V),
        rand(SLOTS, H, ring_len, K),
        -torch.rand(SLOTS, HV, ring_len, device="cuda", generator=g),
        rand(SLOTS, HV, ring_len, V),
        rand(SLOTS, H, ring_len, K),
    ]
    write_pos = torch.randint(0, ring_len - DRAFT, (SLOTS,), device="cuda", generator=g)
    write_pos[:8] = 0  # rows with no history
    cache_base = torch.randint(0, ring_len, (SLOTS,), device="cuda", generator=g)
    # flush rows return early in the verify launch; a misread flag would skip a row
    is_flush = (torch.rand(SLOTS, device="cuda", generator=g) < 0.1).to(torch.int8)
    states = (torch.randperm(SLOTS - 1, device="cuda", generator=g)[:bs] + 1).int()
    if bs > 1:
        states[0] = -1  # a padded request: zero output, ring untouched
    rows = dict(
        query_start_loc=torch.arange(
            0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda"
        ),
        ssm_state_indices=states,
        replay_indices=torch.randperm(SLOTS, device="cuda", generator=g)[:bs].int(),
    )
    return gates, state, ring, rows, (write_pos.int(), cache_base.int(), is_flush)


def _plugin(gates, state, ring, rows, cursors, ring_len, **pdl):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import replayssm_pdl

    got_ring = [t.clone() for t in ring]
    got = gates["q"].new_zeros(len(gates["q"]), HV, V)  # flush rows keep these
    replayssm_pdl.verify(
        *gates.values(), got, state, *got_ring, None, *rows.values(), *cursors,
        max_cache_len=ring_len, max_spec_len=DRAFT, scale=K**-0.5, null_block_id=-1,
        block_v=32, num_warps=2, **pdl,
    )  # fmt: skip
    return got, got_ring


def _equal(x, y):
    return torch.equal(x[0], y[0]) and all(map(torch.equal, x[1], y[1]))


@pytest.mark.parametrize("bs", [1, 4, 8, 16, 33, 64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("ring_len", [16, 32])
def test_verify_is_sglangs(ring_len, strided, bs):
    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as sg

    kernel = sg.gdn_replayssm_spec_circular_kernel.fn
    if "STRIDE_WP" not in inspect.signature(kernel).parameters:
        pytest.skip("this SGLang's verify predates sgl-project/sglang#49719")
    gates, state, ring, rows, cursors = _operands(7 + 10 * bs + ring_len, bs, ring_len)
    if strided:
        cursors = _columns(*cursors)
    want_ring = [t.clone() for t in ring]
    want = gates["q"].new_zeros(len(gates["q"]), HV, V)
    sg.gdn_replayssm_spec_decode(
        **gates, checkpoint_state=state, d_cache=want_ring[0], k_cache=want_ring[1],
        g_cache=want_ring[2], rawv_cache=want_ring[3], rawk_cache=want_ring[4],
        beta_cache=None, out=want, **rows, write_pos=cursors[0], cache_base=cursors[1],
        is_flush=cursors[2], max_cache_len=ring_len, max_spec_len=DRAFT, scale=K**-0.5,
        use_qk_l2norm_in_kernel=True, null_block_id=-1, launch_mode="verify",
        block_v=32, num_warps=2,
    )  # fmt: skip
    eager = _plugin(gates, state, ring, rows, cursors, ring_len)
    assert _equal(eager, (want, want_ring))
    for gate in (False, True):
        pdl = _plugin(
            gates, state, ring, rows, cursors, ring_len, pdl=True,
            wait_for_gate_producer=gate,
        )  # fmt: skip
        assert _equal(pdl, eager), gate


@pytest.mark.parametrize("bs", [4, 8, 16, 33, 64])
@pytest.mark.parametrize("pdl, gate", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("ring_len", [16, 32])
def test_strided_cursors_are_the_unit_stride_bits(pdl, gate, ring_len, bs):
    gates, state, ring, rows, cursors = _operands(
        5 + int(pdl) + 2 * int(gate) + 10 * bs + ring_len, bs, ring_len
    )
    flags = dict(pdl=pdl, wait_for_gate_producer=gate)
    want = _plugin(gates, state, ring, rows, cursors, ring_len, **flags)
    got = _plugin(gates, state, ring, rows, _columns(*cursors), ring_len, **flags)
    assert want[0][DRAFT:].abs().sum() > 0  # the rows past the padded one ran
    assert _equal(got, want)
