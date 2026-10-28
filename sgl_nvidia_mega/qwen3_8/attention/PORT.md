# FN qsa_core slot

Based on plugin f8f99a7 and admitted FN implementation2a92c89d.

The target-verify one-request M4 slot uses native index Q/K preparation and
compression, admitted B score and b76 exact selection, then132 selected-KV
tcgen05 partial CTAs and the PDL merge/gate. The consumer waits before reading
partial results. Mature BF16 qkv/o_proj and the original AR/HC remain native.
M1 and draft-extend paths and their seed/cache lifetime are untouched.

Metadata comes from get_attn_backend() resolved to full_attn_backend; generic
ForwardView.metadata is None and is never used. KV uses view.kv_pool.pool.
Dimensions come from hf_config.text_config, rows/request from PackageContext.
Unsupported dimensions return config_disable_reasons; unsupported runtime
metadata/modes use the original model self_attention before any state writes.

454089: four rank-local224K cases plus derived8K, N5, original2a92 vs port
bitwise, including cache writes. 454345: against exact main4517 native sparse
backend+native pack/gate and FP64, relL2 error ratio .814905-.921637, maxabs
.513759-1.000000. Standard1.05/2 tolerances pass. 8K is explicitly a derived
fixture using real TP4 tensors, not a real8K model capture. The standalone
native decode chain measured11.62-11.94us versus core8.53-8.68us.

Service parity awaits the main4517 flash_mla image fix. These primitive tests
do not establish plugin service performance or GSM quality. New mka and
selected4-token TMA remain unregistered experiments, not part of this slot.
