# FN MoE slots

Opt in with `SGLANG_NVIDIA_MEGAKERNEL_SLOTS=route,mkm`. `route_shared` adds the BF16 shared gate/up projection to the router grid; it is a separate numeric slot. All slots default off.

Dimensions come from `hf_config.text_config`, and rows from the build context. The current router supports H2560/E512/top10. The current SF-MKM supports TP4 logical intermediate160/native ModelOpt storage192, up to7 rows; the served one-request verify uses4 rows. Unsupported shapes return config/build reasons. The storage shard is the native192-row checkpoint slice, not a160-row logical slice followed by padding.

The MoE step uses model-owned modules and weights. It never reads ForwardView metadata or a private backend. It preserves the main-stream producer, the alternate stream's actual wait, the native shared gate/add and true join. The ordinary path reduces slots0→9 using separate FP32 multiply/add and one BF16 store. The deferred path returns the native finalize handoff. Native `ffn_boundary.exit(...).finish(...)` still owns communication and HC. The special single-row native decode communicator remains native.

`route_shared` uses original BF16 shared weights, full-K BF16 MMA and native activation/down/gating. It does not convert these weights to FP8 or use the failed last-cluster shared-drain prototype. Its gate/up output is held to FP64 on the same BF16 operands: mean relative-L2 error ≤1.05 times native BF16, maximum absolute error ≤2 times native.

CPU checks: `pytest test/unit/test_qwen3_8_spec.py test/unit/test_qwen3_8_moe.py`, plus the qwen3_8 cases of `test/unit/test_contract.py`. Compile the router buckets, MKM and finalize with `common.jit.compile_source(..., arch="sm_103a")` on CPU; copy its cache to the GPU worker. GPU compilation is forbidden in the fixture probe.

The GPU test `test/kernels/test_qwen3_8_moe.py` takes `FN_MOE_FIXTURE` (one rank's real layer/router/cases tensor file), `FN_PREPARED` (precompiled artifacts/native.json), `FN_SGLANG` (the exact4517 Python source root), and `SGLANG_CACHE_DIR` (the prepared cache). `FN_MOE_RANK` defaults to0. Both the CPU and GPU probe assert the imported SGLang source root. It compares IDs, BF16 routing weights, canonical expert rows and final output against freshly executed native FlashInfer, and replays changed inputs.

Qualification: route+mkm passed four independent TP4-shaped ranks, M1/M4×3 real inputs, strict native output parity, nine changed-input replays per shape/rank and rank3 memcheck. This is an isolated kernel qualification; model smoke and same-node TP4 Static parity remain separate. Shared gate/up passed four independent ranks, M1/M4×3 real inputs, its1.05/2.0 FP64 gate and rank3 memcheck. Shared-tail graph/model/performance gates remain separate. The shared GPU test additionally takes `FN_SHARED_FIXTURE` (the original BF16 shared weight fixture) and its precompiled shared-router cache.
