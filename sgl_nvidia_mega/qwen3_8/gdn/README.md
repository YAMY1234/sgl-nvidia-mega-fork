FN GDN uses the generic `ForwardView` and the active backend returned by
`get_attn_backend().linear_attn_backend`. Head counts come from HF `text_config`;
verify rows come from the build's `PackageContext.tokens_per_request`.

- `gdn_verify`: native Triton convolution, ReplaySSM, native sigmoid norm/output.
- `gdn_pdl`: the same arithmetic with a gate-safe producer wait. This is a
  bitwise fallback, not the performance competitor of in-tree e5 MK.
- `gdn_mk`: numeric e5 four-warp full-head core and fused sigmoid norm. It owns
  its original PDL entry trigger/wait and preserves native commit/track. Four
  verify rows/request are qualified. Out projection, AR and HC stay native.
- `gdn_front`: the new BF16 N32 projection/conv; not yet available.

Native primitive qualification: four independent GPU seeds, 120/120 tests,
local heads4/12, M4/M8/M12, contiguous/gapped projection views, convolution
output/active/window and ReplaySSM output/rings bitwise (BV64/NW1 and BV32/NW2,
with/without PDL and gate wait). Numeric qualification: four seeds M4/M4/M8/M12,
FP64 single-step and 2048-step recurrence (5120 accepted tokens), sampled every
256 steps. Across raw O, sigmoid norm and tracked state, maximum relative-L2
error ratio to native is1.035356 and maximum absolute-error ratio1.535666.
Standard1.05/2.0 tolerances apply. Both BF16 lanes use the native compact commit;
FP64 keeps ideal unrounded state. These are primitive gates; plugin service
parity/smoke and performance remain pending.
