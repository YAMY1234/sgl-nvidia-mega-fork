# Ordinary MoE merge producer prototype

This unregistered entry is not built by `runtime.build`, not selected by any slot, and does not replace the native FFN exit. The original route/MKM/finalize qualification bundle remains unchanged.

The CPU builder supplies the exact `fn_hop.cuh` from peer branch `qwen38-fn-mega-hop@7e84fdf`. The launch takes contiguous BF16 expert rows `[M,10,H]`, BF16 routing weights `[M,10]`, native shared-down BF16 `[M,H]`, native GATE_ONLY FP32 gates `[M]`, an allocated BF16 local debug tensor `[M,H]`, and `fnhop::Site` by value. Grid is `(ceil(H/512), M)`, block `(128,1,1)`. Each thread owns four consecutive columns; every column is written and pushed exactly once.

The caller must complete the original routed/shared joins before launch and retain all inputs/mailbox through consumption. `pdl_wait` precedes all dependency reads and the stage read. This entry does not trigger dependents early; a dependent consumer can wait for full producer completion. The native ordinary arithmetic includes the routed BF16 cut, FP32 gate/shared/routed FMA, and final BF16 cut. Mailbox push canonicalizes signed zero using the helper; local debug retains the original bits.

Deferred finalize is excluded. Native AR/HC ownership stays with the original exit until a complete boundary adapter is separately qualified. Required future gates: GATE_ONLY FP32/native merge parity; complete fresh producer parity; changed-input graph replay and stage reuse/skew; sanitizer; then service boundary ownership and complete-chain performance. CPU compilation alone qualifies none of those.

CPU compilation passed in the 1028 image (Python3.12/CUDA13.0, sm103a): nvcc compiled this entry with unmodified header SHA256 `a00856404b55e0eed18253908c13e35f2cec13a397b57289132a21817538ce72`; Triton3.8 compiled the exact SGLang4517 native gate body in both GATE_ONLY and ordinary merge variants (BLOCK4096,16 warps, PDL, default FP fusion). This is compilation evidence only, not GPU arithmetic or producer qualification.
