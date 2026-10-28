# Isolated FN HC hop

This is a primitive qualification, not an enabled executor slot. It replaces
neither the native stage lifecycle nor the rank-320 HC down/up projections.

The producer publishes its complete local BF16 output after the real shared
branch join. The consumer sums ranks in FP32 order 0 through 3, rounds the block
to BF16, applies the HC gate FMA and rounds the combined residual to BF16, then
computes each branch's RMS and Gemma normalization. The reduced block and
combined residual must match the native reference bitwise; normalized output
must satisfy the native-relative FP64 error gate.

`fn_hop.cuh` owns a separate three-stage multicast mailbox. Its Site and control
state must not alias FlashInfer or custom-allreduce state. Every generation has
exactly one producer per rank/token/fragment and exactly one consumer. Local PDL
does not replace peer readiness polling. Fixed warp participation masks must
survive the divergent polling loop.

For an MoE producer, routed finalize must round to BF16 before the ordinary
shared-gate FMA, followed by the local BF16 output cut. The deferred path has a
different rounding contract and remains native. Publishing routed-only output
before shared merge is invalid.

The finite test in `test/kernels/fn_hop` uses CPU-built SM103a cubins, rebuilds
the native FI reference from the installed image, and performs no GPU JIT.
Timing reports consume-only and independently charged publish+consume. A fresh
same-generation publish precedes every consume-only measurement. The performance
gate requires at least 0.9 microseconds saved against the same-method native
reference in both hot and cold modes; profile absolute timings are not used as
microbenchmark thresholds. Primitive qualification does not establish service
gain or authorize executor integration.
