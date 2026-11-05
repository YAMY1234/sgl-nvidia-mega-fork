# SPDX-License-Identifier: Apache-2.0
"""Raw router ABI and smoke inputs, using CPU tensors and CUDA launch spies."""

import importlib.util
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_qwen3_8_experts import gpu_doubles, signature, view

from sgl_nvidia_mega.qwen3_8._providers import experts

CASES = ((1, 0), (4, 0), (1, 1), (4, 1), (12, 2), (16, 2))


@pytest.mark.parametrize("rows,variant", CASES)
@pytest.mark.parametrize("capturing", [False, True])
def test_fp32_rejected_before_resources_or_raw_write(
    monkeypatch, rows, variant, capturing
):
    calls, state, _, _ = gpu_doubles(monkeypatch)
    state.capturing = capturing
    run = experts.build(signature(rows, variant))
    io = replace(
        view(rows),
        routing_logits=torch.empty(rows, 512, dtype=torch.float32, device="meta"),
    )
    with pytest.raises(ValueError, match="BF16"):
        run(io)
    assert calls == [] and run._resources == {}


@pytest.mark.parametrize("shape,transposed", [((4, 256), False), ((512, 4), True)])
def test_wrong_shape_or_strides_rejected_without_preparation(
    monkeypatch, shape, transposed
):
    calls, _, _, _ = gpu_doubles(monkeypatch)
    logits = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    if transposed:
        logits = logits.T
        assert logits.shape == (4, 512) and not logits.is_contiguous()
    with pytest.raises(ValueError, match="raw ABI"):
        experts.build(signature())(replace(view(), routing_logits=logits))
    assert calls == []


@pytest.mark.parametrize("rows,variant", CASES)
def test_bf16_tensor_identity_and_pointer_reach_exact_raw_family(
    monkeypatch, rows, variant
):
    calls, _, _, _ = gpu_doubles(monkeypatch)
    logits = torch.empty(rows, 512, dtype=torch.bfloat16)
    io = replace(view(rows), routing_logits=logits)
    pointer = logits.data_ptr()
    assert experts.build(signature(rows, variant))(io) is None
    name, args = next((n, a) for n, a in calls if n in ("narrow-run", "wide-run"))
    actual = args[2] if name == "narrow-run" else args[2]["routing_logits"]
    assert (
        actual is logits
        and actual.data_ptr() == pointer
        and actual.dtype == torch.bfloat16
    )


@pytest.mark.parametrize(
    "name",
    [
        "q38_mk_moe_q35_tp4.cu",
        "q38_mk_moe_q35_sf_pack_tp4.cu",
        "q38_mkm_wide.cu",
    ],
)
def test_actual_raw_family_declares_bf16_router_abi(name):
    path = Path(experts.__file__).parents[1] / "kernels/jit/csrc/moe" / name
    assert re.search(
        r"const\s+uint16_t\s*\*\s*logits;\s*//\s*\[M,\s*512\]\s*bf16 router logits",
        path.read_text(),
    )


@pytest.mark.parametrize("rows", [1, 4, 12, 16])
def test_new_smoke_uses_raw_bf16_and_strict_top_ten(rows):
    path = Path(__file__).parents[1] / "smoke/qwen3_8_experts.py"
    spec = importlib.util.spec_from_file_location("expert_smoke_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logits = module.router_logits(rows, 512, torch.device("cpu"))
    assert (
        logits.dtype == torch.bfloat16
        and logits.shape == (rows, 512)
        and logits.is_contiguous()
    )
    values, indices = logits.topk(10, dim=-1)
    assert torch.equal(indices, torch.arange(10).expand(rows, -1))
    assert torch.all(values[:, :-1] > values[:, 1:])
