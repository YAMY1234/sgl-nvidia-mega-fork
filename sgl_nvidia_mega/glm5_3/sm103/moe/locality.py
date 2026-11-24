# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Die-local (locality-domain) form of the small-M MoE engine, host side.

On a GPU with two locality domains (two dies, each with its own HBM), memory from
``cudaMalloc`` is interleaved over both dies, so about half of the weight bytes an SM
streams cross the die-to-die link. With ``SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_LOCALIZE=1``:

* **weights** (``Banks``): each MoE layer's routed FC1 weights and scales (the server's
  tensors, re-bound in place: same shape, dtype and layout), FC2 records and shared
  pack live in VMM allocations made on a locality domain (``cuMemCreate`` with
  ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``): the first half of each tensor
  (experts 0..127; shared quarters 0, 1) on domain 0, the second half on domain 1, at
  the allocation granularity (a page straddling the half goes to the domain holding
  most of its bytes). One virtual range per tensor, so every kernel address and the
  server's own MoE path are unchanged. The banks are made before the KV cache is sized
  and the server's original FC1 storage is freed, so the KV sizing sees the net use.
* **SM map** (``probe``): the domain of each ``%smid``, by green contexts per domain
  (``cuDevSmResourceSplit``, TRT-LLM's method) and by pointer chasing over per-domain
  buffers (DeepGEMM's method); both must agree where both are defined.
* **placement**: how many of the engine's N co-resident 4-CTA clusters land on domain
  0 (a cluster launch of the engine's shape), the engine's n_0 virtual slots.
* **locality blocks** (``LocBlocks``): one 2 KiB block per layer (``constants.LOC_*``):
  the claim word, n_0 per grid size and the ``%smid`` -> domain table.

A device with one locality domain, a driver or bindings without the locality API, or a
probe that fails or disagrees: the engine stays stock (logged)."""

from __future__ import annotations

import ctypes
import logging
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .constants import LOC_BYTES, LOC_DIE, LOC_N0

logger = logging.getLogger(__name__)

LOCALIZE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_LOCALIZE"
MAX_SMID = LOC_BYTES - LOC_DIE  # %smid table entries (1024)
CLUSTER = 4
PROBE_SMEM = 223664  # the four-request engine's shared memory: one CTA per SM


def requested() -> bool:
    return os.environ.get(LOCALIZE_ENV, "").strip() not in ("", "0")


class Unavailable(RuntimeError):
    """This device or driver cannot run the die-local form."""


# ------------------------------------------------------------------ driver glue
def _cu():
    import cuda.bindings.driver as cu

    return cu


def _ck(ret, what: str):
    if isinstance(ret, tuple):
        err, vals = ret[0], ret[1:]
    else:
        err, vals = ret, ()
    if int(err) != 0:
        raise Unavailable(f"{what}: {err}")
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


PROBE_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
__device__ __forceinline__ u32 smid_() { u32 r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u64 clk_() { u64 r; asm volatile("mov.u64 %0, %%clock64;" : "=l"(r) :: "memory"); return r; }
__device__ __forceinline__ u32 ld_cg(const u32* p) {
  u32 r; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
extern "C" __global__ void smid_probe(u32* out) { if (threadIdx.x == 0) out[blockIdx.x] = smid_(); }
// one CTA per SM (large shared memory): cycles per dependent hop over CHUNKS chunks of
// this SM's slot (128-byte hops, a 32-hop ring per 4 KiB chunk)
extern "C" __global__ void chase(const u32* buf, u32* out, u32 chunks, u32 hops) {
  extern __shared__ u32 pad[];
  if (threadIdx.x) return;
  const u32 s = smid_();
  if (s >= 1024u) return;
  u32 best = 0xffffffffu;
  for (u32 c = 0; c < chunks; ++c) {
    const u32* ch = buf + (u64)(s * chunks + c) * 1024u;
    u32 idx = ld_cg(ch);
    if (idx == 0xfffffffeu) pad[0] = idx;
    const u64 t = clk_();
    for (u32 h = 0; h < hops; ++h) idx = ld_cg(ch + idx);
    if (idx == 0xfffffffeu) pad[0] = idx;
    const u32 dt = (u32)((clk_() - t) / hops);
    best = dt < best ? dt : best;
  }
  out[s] = best;
}
// the engine's launch shape (4-CTA clusters, one CTA per SM): each CTA's %smid
extern "C" __global__ void cluster_probe(u32* out) {
  extern __shared__ u32 pad[];
  if (threadIdx.x == 0) { pad[0] = 0; out[blockIdx.x] = smid_(); }
}
"""


def _nvrtc(arch: str) -> bytes:
    from cuda.bindings import nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(PROBE_SRC.encode(), b"moe_loc.cu", 0, [], [])
    if int(err):
        raise Unavailable(f"nvrtcCreateProgram: {err}")
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
    (cerr,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(cerr):
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        nvrtc.nvrtcDestroyProgram(prog)
        raise Unavailable(f"NVRTC {arch}: {log.decode(errors='replace')[-800:]}")
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin


class _Module:
    def __init__(self, cubin: bytes):
        cu = _cu()
        self.cu = cu
        self.mod = _ck(cu.cuModuleLoadData(cubin), "module")

    def fn(self, name: str, smem: int = 0):
        cu = self.cu
        f = _ck(cu.cuModuleGetFunction(self.mod, name.encode()), name)
        if smem:
            _ck(
                cu.cuFuncSetAttribute(
                    f,
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem,
                ),
                f"{name} smem",
            )
        return f

    def launch(self, f, grid, block, args, stream, smem=0, cluster=0):
        cu = self.cu
        holders = []
        for a in args:
            holders.append(
                ctypes.c_uint64(a) if isinstance(a, int) else a
            )  # pointers as u64, scalars pre-wrapped
        arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.addressof(h) for h in holders]
        )
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = stream
        if cluster:
            at = cu.CUlaunchAttribute()
            at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = (
                cluster,
                1,
                1,
            )
            cfg.attrs = [at]
            cfg.numAttrs = 1
        else:
            cfg.numAttrs = 0
        _ck(cu.cuLaunchKernelEx(cfg, f, ctypes.addressof(arr), 0), "launch")

    def unload(self):
        self.cu.cuModuleUnload(self.mod)


class _Raw:
    """``__cuda_array_interface__`` of raw device bytes (for ``torch.as_tensor``)."""

    def __init__(self, ptr: int, size: int):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
        }


# ------------------------------------------------------------------ banks
class Banks:
    """VMM allocations homed on locality domains, one virtual range per tensor."""

    def __init__(self, device_index: int, ndom: int = 2):
        cu = _cu()
        self.cu, self.dev, self.ndom = cu, int(device_index), ndom
        self.loc_type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        self.gran = max(
            int(
                _ck(
                    cu.cuMemGetAllocationGranularity(self._prop(d), flag),
                    "granularity",
                )
            )
            for d in range(ndom)
        )
        self.maps: List[Tuple[int, int, list]] = []
        self.bytes_by_die = [0] * ndom

    def _prop(self, d: int):
        cu = self.cu
        p = cu.CUmemAllocationProp()
        p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        p.location.type = self.loc_type
        p.location.id = self.dev | (d << 8)
        return p

    def page_dies(self, nbytes: int, split: int) -> List[int]:
        """Domain of each page: 0 if most of its bytes lie below ``split``."""
        g = self.gran
        pages = (nbytes + g - 1) // g
        return [0 if split - p * g >= g // 2 else 1 for p in range(pages)]

    def alloc(self, nbytes: int, dies: List[int]) -> int:
        cu, g = self.cu, self.gran
        size = len(dies) * g
        va = int(_ck(cu.cuMemAddressReserve(size, g, cu.CUdeviceptr(0), 0), "reserve"))
        handles = []
        try:
            p = 0
            while p < len(dies):
                q = p
                while q < len(dies) and dies[q] == dies[p]:
                    q += 1
                run = (q - p) * g
                h = _ck(cu.cuMemCreate(run, self._prop(dies[p]), 0), "cuMemCreate")
                handles.append(h)
                _ck(cu.cuMemMap(cu.CUdeviceptr(va + p * g), run, 0, h, 0), "map")
                self.bytes_by_die[dies[p]] += run
                p = q
            acc = cu.CUmemAccessDesc()
            acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id = self.dev
            acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ck(cu.cuMemSetAccess(cu.CUdeviceptr(va), size, [acc], 1), "access")
        except Exception:
            self._release(va, size, handles)
            raise
        self.maps.append((va, size, handles))
        return va

    def ordinal(self, va: int) -> int:
        cu = self.cu
        return int(
            _ck(
                cu.cuPointerGetAttribute(
                    cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                    cu.CUdeviceptr(va),
                ),
                "ordinal",
            )
        )

    def tensor(self, nbytes: int, split: Optional[int] = None) -> torch.Tensor:
        """uint8 [nbytes] on the banks: bytes below ``split`` (default: half) homed on
        domain 0, the rest on domain 1 (page granular). Checked by the pointer
        attribute at the first and last page."""
        split = nbytes // 2 if split is None else split
        dies = self.page_dies(nbytes, split)
        va = self.alloc(nbytes, dies)
        for p in (0, len(dies) - 1):
            got = self.ordinal(va + p * self.gran)
            if got != dies[p]:
                raise Unavailable(
                    f"page {p} reports locality domain {got}, wanted {dies[p]}"
                )
        return torch.as_tensor(_Raw(va, nbytes), device=f"cuda:{self.dev}")

    def _release(self, va, size, handles):
        cu = self.cu
        cu.cuMemUnmap(cu.CUdeviceptr(va), size)
        for h in handles:
            cu.cuMemRelease(h)
        cu.cuMemAddressFree(cu.CUdeviceptr(va), size)

    def free_all(self) -> None:
        torch.cuda.synchronize(self.dev)
        for va, size, handles in self.maps:
            self._release(va, size, handles)
        self.maps.clear()
        self.bytes_by_die = [0] * self.ndom

    def total(self) -> int:
        return sum(self.bytes_by_die)


# ------------------------------------------------------------------ topology
@dataclass
class Topology:
    ndom: int
    sms: int
    sm_to_die: List[int]  # by %smid, -1 where unknown
    sms_per_die: List[int]
    green: Dict = field(default_factory=dict)
    chase: Dict = field(default_factory=dict)
    placement: Dict[int, Dict] = field(default_factory=dict)  # N -> probe stats
    n0: Dict[int, int] = field(default_factory=dict)  # N -> die-0 clusters
    granularity: int = 0

    def summary(self) -> Dict:
        return {
            "domains": self.ndom,
            "sms": self.sms,
            "sms_per_die": self.sms_per_die,
            "green": self.green,
            "chase": {k: v for k, v in self.chase.items() if k != "cycles"},
            "placement": self.placement,
            "n0": self.n0,
            "granularity": self.granularity,
        }


def domain_count(device_index: int) -> int:
    cu = _cu()
    _ck(cu.cuInit(0), "cuInit")
    dev = _ck(cu.cuDeviceGet(device_index), "device")
    attr = getattr(
        cu.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None
    )
    if attr is None:
        raise Unavailable("cuda-bindings without CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT")
    return int(_ck(cu.cuDeviceGetAttribute(attr, dev), "locality domain count"))


def _split_group(cu, dev, d: int, supplemental: bool, sm_count: int):
    F = cu.CUdevSmResourceGroup_flags
    flags = int(F.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID)
    if supplemental:
        flags |= int(F.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_SUPPLEMENTAL_SMS)

    def attempt(count):
        sms = _ck(
            cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
            "dev resource",
        )
        p = cu.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        p.smCount, p.flags, p.localityDomainId = count, flags, d
        ret = cu.cuDevSmResourceSplit(1, sms, 0, [p])
        if int(ret[0]) != 0:
            return None, 0
        grp = ret[1][0] if isinstance(ret[1], (list, tuple)) else ret[1]
        return grp, int(grp.sm.smCount)

    grp, n = attempt(0)  # discovery mode
    if (grp is None or n == 0) and not supplemental:
        for count in range(sm_count, 0, -1):
            grp, n = attempt(count)
            if grp is not None:
                break
    return grp, n


def _green_map(cu, dev, ndom: int, sm_count: int, cubin: bytes) -> Dict:
    """TRT-LLM's method: a green context per domain (with its supplemental SMs) runs
    an ``%smid`` probe over many CTAs."""
    prev = _ck(cu.cuCtxGetCurrent(), "ctx")
    sets, per = [], []
    try:
        for d in range(ndom):
            # with the supplemental SMs where the driver accepts it (a 13.4 user-mode
            # driver refused it on GB300: T95), else the stable domain only (the
            # pointer chase places the remaining SMs)
            sgrp, stable = _split_group(cu, dev, d, False, sm_count)
            grp, full = _split_group(cu, dev, d, True, sm_count)
            if grp is None or full == 0:
                grp, full = sgrp, 0
            seen = set()
            if grp is not None:
                desc = _ck(cu.cuDevResourceGenerateDesc([grp], 1), "desc")
                green = _ck(
                    cu.cuGreenCtxCreate(
                        desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
                    ),
                    "green ctx",
                )
                try:
                    _ck(cu.cuCtxSetCurrent(_ck(cu.cuCtxFromGreenCtx(green), "ctx")), "set")
                    st = _ck(
                        cu.cuGreenCtxStreamCreate(
                            green, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
                        ),
                        "green stream",
                    )
                    m = _Module(cubin)
                    f = m.fn("smid_probe")
                    blocks = 8 * sm_count
                    buf = _ck(cu.cuMemAlloc(4 * blocks), "alloc")
                    host = (ctypes.c_uint32 * blocks)()
                    try:
                        for _ in range(8):
                            m.launch(f, blocks, 32, [int(buf)], st)
                            _ck(cu.cuStreamSynchronize(st), "sync")
                            _ck(cu.cuMemcpyDtoH(ctypes.addressof(host), buf, 4 * blocks), "dtoh")
                            seen.update(int(v) for v in host)
                    finally:
                        cu.cuMemFree(buf)
                        m.unload()
                        cu.cuStreamDestroy(st)
                finally:
                    cu.cuGreenCtxDestroy(green)
            sets.append(seen)
            per.append({"domain": d, "stable": stable, "with_supplemental": full, "seen": len(seen)})
    finally:
        _ck(cu.cuCtxSetCurrent(prev), "restore ctx")
    allsm = set().union(*sets) if sets else set()
    overlap = sum(len(a & b) for k, a in enumerate(sets) for b in sets[k + 1 :])
    sm_to_die = [-1] * MAX_SMID
    for d, s in enumerate(sets):
        for x in s:
            if x < MAX_SMID:
                sm_to_die[x] = d
    return {
        "per_domain": per,
        "partition_ok": overlap == 0 and len(allsm) == sm_count,
        "overlap": overlap,
        "sm_to_die": sm_to_die,
    }


def _chase_map(banks: Banks, mod: _Module, sm_count: int, device: torch.device) -> Dict:
    """DeepGEMM's method: per-SM dependent-load latency over a buffer on each domain;
    an SM's domain is the near one (far / near >= 1.25 required)."""
    chunks = 4
    nbytes = MAX_SMID * chunks * 4096
    stream = torch.cuda.current_stream(device).cuda_stream
    f = mod.fn("chase", PROBE_SMEM)
    ring = torch.zeros(1024, dtype=torch.int32, device=device)
    ring[0::32] = ((torch.arange(32, device=device) + 7) % 32 * 32).int()
    cyc = []
    tensors = []
    for d in range(2):
        t = banks.tensor(nbytes, split=nbytes if d == 0 else 0)
        tensors.append(t)
        t.view(torch.int32).view(-1, 1024).copy_(ring.expand(MAX_SMID * chunks, 1024))
        out = torch.zeros(MAX_SMID, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        best = None
        for _ in range(3):
            out.zero_()
            mod.launch(
                f,
                sm_count,
                32,
                [t.data_ptr(), out.data_ptr(), ctypes.c_uint32(chunks), ctypes.c_uint32(31)],
                stream,
                smem=PROBE_SMEM,
            )
            torch.cuda.synchronize(device)
            v = out.cpu()
            best = v if best is None else torch.minimum(best, torch.where(v > 0, v, best))
        cyc.append(best.tolist())
    sm_to_die = [-1] * MAX_SMID
    ratios = []
    for s in range(MAX_SMID):
        a, b = cyc[0][s], cyc[1][s]
        if a <= 0 or b <= 0:
            continue
        near, far = (0, b / a) if a <= b else (1, a / b)
        ratios.append(far)
        if far >= 1.25:
            sm_to_die[s] = near
    del tensors
    return {
        "sm_to_die": sm_to_die,
        "measured": sum(1 for s in range(MAX_SMID) if cyc[0][s] > 0),
        "clear": sum(1 for x in sm_to_die if x >= 0),
        "far_near_median": round(statistics.median(ratios), 3) if ratios else None,
        "far_near_min": round(min(ratios), 3) if ratios else None,
        "cycles": cyc,
    }


def placement(mod: _Module, sm_to_die: List[int], nclusters: int, device, rounds=16) -> Dict:
    """Clusters per domain for the engine's launch shape (N 4-CTA clusters of 256
    threads, one CTA per SM), over ``rounds`` launches."""
    stream = torch.cuda.current_stream(device).cuda_stream
    f = mod.fn("cluster_probe", PROBE_SMEM)
    out = torch.zeros(CLUSTER * nclusters, dtype=torch.int32, device=device)
    counts, mixed, unknown = [], 0, 0
    for _ in range(rounds):
        out.fill_(-1)
        mod.launch(f, CLUSTER * nclusters, 256, [out.data_ptr()], stream, smem=PROBE_SMEM, cluster=CLUSTER)
        torch.cuda.synchronize(device)
        sm = out.view(nclusters, CLUSTER).cpu().tolist()
        n0 = 0
        for row in sm:
            dies = {sm_to_die[s] if 0 <= s < MAX_SMID else -1 for s in row}
            if len(dies) != 1:
                mixed += 1
            d = min(dies)
            if d < 0:
                unknown += 1
            n0 += d == 0
        counts.append(n0)
    hist = Counter(counts)
    return {
        "n0_hist": dict(sorted(hist.items())),
        "n0": hist.most_common(1)[0][0],
        "mixed_clusters": mixed,
        "unknown_clusters": unknown,
        "rounds": rounds,
    }


def probe(device: torch.device, grids: Tuple[int, ...]) -> Topology:
    """The device's locality topology and the engine's per-domain placement for each
    grid size in ``grids``; raises ``Unavailable`` when the die-local form cannot run."""
    cu = _cu()
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    device = torch.device("cuda", index)
    ndom = domain_count(index)
    if ndom != 2:
        raise Unavailable(f"{ndom} locality domain(s)")
    dev = _ck(cu.cuDeviceGet(index), "device")
    props = torch.cuda.get_device_properties(device)
    sms = props.multi_processor_count
    cubin = _nvrtc(f"sm_{props.major}{props.minor}a")
    green = _green_map(cu, dev, ndom, sms, cubin)
    banks = Banks(index, ndom)
    mod = _Module(cubin)
    try:
        chase = _chase_map(banks, mod, sms, device)
        gmap, cmap = green["sm_to_die"], chase["sm_to_die"]
        both = [s for s in range(MAX_SMID) if gmap[s] >= 0 and cmap[s] >= 0]
        agree = sum(1 for s in both if gmap[s] == cmap[s])
        green_s = {k: v for k, v in green.items() if k != "sm_to_die"}
        green_s["agree_with_chase"] = f"{agree}/{len(both)}"
        if agree != len(both) or not both:
            raise Unavailable(f"SM maps disagree: green vs chase {agree}/{len(both)}")
        # green (with the supplemental SMs) where defined, the chase elsewhere
        sm_to_die = [gmap[s] if gmap[s] >= 0 else cmap[s] for s in range(MAX_SMID)]
        known = sum(1 for x in sm_to_die if x >= 0)
        if known < sms:
            raise Unavailable(f"domain known for {known} of {sms} SMs")
        topo = Topology(
            ndom=ndom,
            sms=sms,
            sm_to_die=sm_to_die,
            sms_per_die=[sum(1 for x in sm_to_die if x == d) for d in range(ndom)],
            green=green_s,
            chase=chase,
            granularity=banks.gran,
        )
        for n in sorted(set(grids)):
            pl = placement(mod, sm_to_die, n, device)
            topo.placement[n] = pl
            topo.n0[n] = pl["n0"]
            if pl["mixed_clusters"] or pl["unknown_clusters"]:
                raise Unavailable(f"cluster placement at N={n}: {pl}")
            if not 3 <= pl["n0"] <= n - 3:
                raise Unavailable(f"N={n}: {pl['n0']} clusters on domain 0")
    finally:
        mod.unload()
        banks.free_all()
    return topo


# ------------------------------------------------------------------ locality blocks
class LocBlocks:
    """One ``LOC_BYTES`` block per layer: zero claim word, n_0 per grid size and the
    ``%smid`` -> domain table (unknown SMs as domain 0)."""

    def __init__(self, topo: Topology, nlayers: int, device: torch.device):
        blk = torch.zeros(LOC_BYTES, dtype=torch.uint8)
        for n, n0 in topo.n0.items():
            blk[LOC_N0 + n] = n0
        blk[LOC_DIE : LOC_DIE + MAX_SMID] = torch.tensor(
            [max(0, d) for d in topo.sm_to_die], dtype=torch.uint8
        )
        self.blocks = blk.repeat(nlayers, 1).to(device).contiguous()

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.blocks[i]


# ------------------------------------------------------------------ relocation
def relocate_param(holder, name: str, banks: Banks) -> int:
    """Re-bind ``holder.<name>`` (a contiguous parameter) to a copy on the banks (same
    shape and dtype; first half on domain 0); returns its bytes."""
    p = getattr(holder, name)
    t = p.data
    if not t.is_contiguous():
        raise Unavailable(f"{name} is not contiguous")
    nbytes = t.numel() * t.element_size()
    b = banks.tensor(nbytes)
    b.copy_(t.reshape(-1).view(torch.uint8))
    p.data = b.view(t.dtype).view(t.shape)
    return nbytes


def to_banks(t: torch.Tensor, banks: Banks) -> torch.Tensor:
    """A contiguous copy of ``t`` on the banks (first half on domain 0)."""
    t = t.contiguous()
    b = banks.tensor(t.numel() * t.element_size())
    b.copy_(t.reshape(-1).view(torch.uint8))
    return b.view(t.dtype).view(t.shape)


def describe(topo: Topology) -> str:
    return (
        f"{topo.ndom} locality domains, SMs per domain {topo.sms_per_die}, "
        f"green/chase agree {topo.green.get('agree_with_chase')}, "
        f"far/near {topo.chase.get('far_near_median')}, "
        f"clusters on domain 0 {topo.n0}, granularity {topo.granularity >> 20} MiB"
    )



# ------------------------------------------------------------------ schedule model
def virtual_slot(die: int, rank: int, n0: int, nclusters: int) -> Tuple[int, int]:
    """The engine's claim rule: domain-local rank -> virtual slot (vd, vr)."""
    nd = n0 if die == 0 else nclusters - n0
    if rank < nd:
        return die, rank
    return 1 - die, nclusters - 1 - rank


def slot_list(vd: int, vr: int, n0: int, nclusters: int, sw: int, d0: int, d1: int):
    """Python model of ``engine16.table_local``: the virtual slot's unit list, each entry
    ("sh", quarter) or (domain, unit index in that domain's class-ordered list)."""
    u = (4 * d0, 4 * d1)
    m = (n0, nclusters - n0)
    cap = (u[0] + u[1] + 4 * sw + nclusters - 1) // nclusters
    x = tuple(max(0, u[d] + 2 * sw - m[d] * cap) for d in (0, 1))
    md, ud, xd, uo, xo = m[vd], u[vd], x[vd], u[1 - vd], x[1 - vd]
    own = ud - xd
    tot = own + xo
    nr = md - 2
    lim = min(tot, nr * cap)
    out = []
    if vr >= 2:
        idxs = range(vr - 2, lim, nr)
    else:
        out.append(("sh", 2 * vd + vr))
        idxs = range(lim + vr, tot, 2)
    for idx in idxs:
        out.append((vd, idx) if idx < own else (1 - vd, uo - xo + idx - own))
    return out


def check_schedule(
    nclusters: int, n0: int, sw: int, max_units: int = 17, max_d: int = 128
) -> Dict:
    """Every (d0, d1) with d0 + d1 <= ``max_d`` distinct experts (d0 of them in domain
    0): every unit of both domains and the four shared quarters in exactly one slot
    list, no list longer than ``max_units``; also the claim rule's exact cover for every
    physical split."""
    worst = 0
    for d0 in range(max_d + 1):
        for d1 in range(max_d + 1 - d0):
            seen = Counter()
            for vd in (0, 1):
                for vr in range(n0 if vd == 0 else nclusters - n0):
                    lst = slot_list(vd, vr, n0, nclusters, sw, d0, d1)
                    worst = max(worst, len(lst))
                    seen.update(lst)
            want = {("sh", q) for q in range(4)}
            want |= {(0, i) for i in range(4 * d0)} | {(1, i) for i in range(4 * d1)}
            if set(seen) != want or any(v != 1 for v in seen.values()):
                raise AssertionError(f"cover N={nclusters} n0={n0} d=({d0},{d1})")
    if worst > max_units:
        raise AssertionError(f"N={nclusters} n0={n0}: a slot list of {worst} > {max_units}")
    for a0 in range(nclusters + 1):
        slots = Counter()
        for die, cnt in ((0, a0), (1, nclusters - a0)):
            for r in range(cnt):
                slots[virtual_slot(die, r, n0, nclusters)] += 1
        want = {(0, r) for r in range(n0)} | {(1, r) for r in range(nclusters - n0)}
        if set(slots) != want or any(v != 1 for v in slots.values()):
            raise AssertionError(f"claim cover N={nclusters} n0={n0} a0={a0}")
    return {"max_units": worst}
