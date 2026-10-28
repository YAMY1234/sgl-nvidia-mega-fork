FN GDN uses the generic `ForwardView` and the active backend returned by
`get_attn_backend().linear_attn_backend`. Head counts come from HF `text_config`;
verify rows come from the build's `PackageContext.tokens_per_request`.

- `gdn_verify`: native Triton convolution, ReplaySSM, native sigmoid norm/output.
- `gdn_pdl`: the same arithmetic with a gate-safe producer wait. This is a
  bitwise fallback, not the performance competitor of in-tree e5 MK.
- `gdn_mk`: mature e5 BF16 projection/conv plus the numeric four-warp full-head
  core and fused sigmoid norm. It owns
  its original PDL entry trigger/wait and preserves native commit/track. Four
  verify rows/request are qualified. Out projection, AR and HC stay native.

Initial qualification on the 1015 image: four independent GPU seeds, 120/120 tests,
local heads4/12, M4/M8/M12, contiguous/gapped projection views, convolution
output/active/window and ReplaySSM output/rings bitwise (BV64/NW1 and BV32/NW2,
with/without PDL and gate wait). Numeric qualification: four seeds M4/M4/M8/M12,
FP64 single-step and 2048-step recurrence (5120 accepted tokens), sampled every
256 steps. Across raw O, sigmoid norm and tracked state, maximum relative-L2
error ratio to native is1.035356 and maximum absolute-error ratio1.535666.
Standard1.05/2.0 tolerances apply. Both BF16 lanes use the native compact commit;
FP64 keeps ideal unrounded state. These are primitive gates; plugin service
parity/smoke and performance remain pending.

Mature front port:24/24 GPU cases against frozen e5 source, M4/M8/M12 with
and without bias, int32/int64 indices, strided windows, three changed-input/state
graph replays per case. The same BF16 mature front is retained for the MK slot;
the new N32 front was stopped after failing its cold-L2 mixer speed gate. Integrated87153be CPU gate:
FN spec12, own GDN8 (including native/numeric GDN + MoE build/plan combinations),
qwen3_8 contract30 and entrypoint1 pass. Combined wiring tests mock GPU resource
construction; they do not claim model execution or service parity.

N32 research is not registered as a slot and has no numeric tolerance claim.
The mature BF16 front remains part of `gdn_mk`.

Requalification on the 1028 image (Python3.12, FlashInfer0.7.2,
sglang-kernel0.4.10) passed with unchanged runtime code:

| Gate | Result |
|---|---|
| FN CPU units / FN contract | 33 / 38 passed |
| Native conv / ReplaySSM / PDL | 120/120 bitwise |
| Mature BF16 front | 24/24 bitwise, including three changed-input graph replays |
| MK FP64 single-step and 2048-step tracks | All standard1.05/2.0 bounds passed |

The maximum MK/native FP64 error ratios were1.036255 relative-L2 and1.690461
max-absolute, across raw O, sigmoid norm and tracked state. The four numeric
arms covered M4/M4/M8/M12 and5120 accepted tokens each. These new CPU-generated
input streams are not compared numerically with the earlier GPU-generated
streams. All GPU kernels were loaded from same-image CPU-precompiled cubins
or C-export objects; no GPU compilation was allowed. The core retained206
registers,1152 static shared bytes and zero local bytes.

Service qualification is separate: judge parity only on short windows of at
most64 steps; use parity-disabled natural-AL GSM32 against a matching all-off
service for longer requests. Native/native control also differs on long
parity windows because the harness does not restore every in-place state.
