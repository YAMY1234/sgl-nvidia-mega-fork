# SPDX-License-Identifier: Apache-2.0
"""BF16 shared gate/up in the FN router grid. Numeric qualification is separate."""

from functools import cache
from pathlib import Path
import torch
from sgl_nvidia_mega.common import jit
from . import router

SOURCE = Path(__file__).parent / "src/router_shared.cu"


def shared_bytes(shape, capacity):
    # Preserve router layout; BF16 shared weights and reserved token plane.
    return (
        (8 + capacity) * (shape.hidden * 2 + 16)
        + 2 * 16 * 8 * 4
        + (8 + capacity) * (shape.hidden * 2 + 16)
        + 4 * 16 * 8 * 4
        + 24
    )


@cache
def _kernel(device, shape, capacity):
    major, minor = torch.cuda.get_device_capability(device)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        f"q38_router_shared_m{capacity}",
        f"sm_{major}{minor}a",
        router.options(shape, capacity),
    )
    return jit.load(
        cubin, "q38_router_shared_bf16", (256, 1, 1), shared_bytes(shape, capacity)
    )


def build(device, shape):
    for cap in sorted({router.max_m_bucket(m) for m in range(1, shape.max_rows + 1)}):
        _kernel(device, shape, cap)


def router_shared(x, weight, scale, shared_weight, *, shape):
    m = x.shape[0]
    n = shared_weight.shape[0]
    if not 1 <= m <= 8:
        raise ValueError(f"BF16 shared router currently covers 1..8 rows, got {m}")
    logits = torch.empty((m, shape.experts), dtype=torch.bfloat16, device=x.device)
    packed = torch.empty((m, shape.hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m, shape.hidden // 16), dtype=torch.uint8, device=x.device)
    gate_up = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    p = router.Params(
        x=x.data_ptr(),
        weight=weight.data_ptr(),
        global_scale=scale.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=packed.data_ptr(),
        x_sf=sf.data_ptr(),
        num_tokens=m,
        mode=1,
        sh_weight=shared_weight.data_ptr(),
        sh_out=gate_up.data_ptr(),
        sh_n=n,
    )
    jit.launch(
        _kernel(x.device.index, shape, router.max_m_bucket(m)),
        (shape.experts // 8, 1, 1),
        [p],
        torch.cuda.current_stream().cuda_stream,
    )
    return (logits, packed, sf), gate_up
