# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and is turned on with SGLang's `--enable-nvidia-megakernel` server flag. The package registers each model's spec with SGLang's megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, `sglang.srt.megakernels` before sgl-project/sglang#45389). With the flag, SGLang checks the setup at argument resolution, before the weights load. After the model runner has prepared a covered model (before the KV cache is sized), SGLang builds the megakernel and replaces that model instance's `forward`, which runs each covered step itself and leaves every other step to the model's own forward. Every rank then logs `NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103); 148-SM B300 (SM103) with `sm148` | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request); on B300, speed was measured at one request only, serving passed a GSM8K canary at client concurrency 12 and 16 with decode CUDA graphs up to batch 16 (DSpark block 5), and larger batches and other block sizes rely on the kernel tests |
| DeepSeek-V4.1-Flash | GB300 (SM103) | 8, with the MoE at TP4 x EP2 (two trays, one NVLink domain) | The same verify widths and the DSpark draft step; served and measured at one request (DSpark block 5); larger batches rely on the world-8 kernel tests. See [TP8 + EP2](#deepseek-v41-flash-at-tp8--ep2) |
| Qwen3.5-397B-A17B (NVFP4) | GB300 (SM103) | 8 (two trays, one NVLink domain) | MTP target verify steps of 7 rows (batch 1) |
| Kimi K3 (packed MXFP4) | GB300 (SM103, 152 SMs); 148-SM B300 (SM103) and B200/GB200 (SM100) with `sm148` | 8 | Decode rows 1–8, 16, 32 and 64; fixed q4–q8 verification within 64 token rows. Admission and tested configurations differ; see [`spec.py`](sgl_nvidia_mega/kimi_k3/spec.py). The 148-SM admission has no TP8 serving validation yet; the same gate was validated on Kimi K3.1 at TP16 on GB200 |
| Kimi K3.1 (packed MXFP4) | GB300 (SM103) | 16 (one NVLink domain); 8 | DFlash target verify steps of 8 rows a request (7 drafted tokens and the anchor) at 1 to 8 requests, without DCP or with DCP equal to TP, also as the decode role of a PD-disaggregated deployment; the DFlash2 draft decoder layers at one request. Served and measured end to end at TP16, one request. See [Kimi K3.1 on GB300](#kimi-k31-on-gb300) |
| GLM-5.3, GLM-5.2 (NVFP4) | B300 / GB300 (SM103), Rubin (SM107) | 8 (one node, or two trays of one NVLink domain); 4 (one GB300 tray) | TP8: MTP target verify steps of 4 to 16 rows (batch 1 to 4 at 4 rows a request, batch 1 to 2 at 6); TP4: verify steps of up to 48 rows at 4 to 6 rows a request (batch 1 to 8 at 6); the NextN draft step in its captured graphs |

For DeepSeek-V4.1-Flash, larger verify steps, prefill and extend run
SGLang's standard path. Qwen3.5 also offers the explicit GDN prefill slot below. Their draft specs provide the DSpark proposal and the
Qwen3.5 EAGLE entries, respectively. For Kimi K3, the plugin covers the 93-layer
language model (24 MLA and 69 KDA layers) at the listed widths; every other call
runs the standard path. For GLM-5.3, larger verify steps, prefill and extend run
SGLang's standard path, and the NextN draft step runs on its draft spec.

GLM-5.2 has GLM-5.3's architecture and the same value of every config field the
GLM-5.3 specs pin (`HF_CONFIG` in `glm5_3/__init__.py`), so the same specs serve it,
and what this README says of GLM-5.3 holds for GLM-5.2. On `nvidia/GLM-5.2-NVFP4`
(8 x B300) the megakernel engages, a test-only step-by-step comparison with SGLang's
own target verify and draft steps shows no difference, and GSM8K-200 (strict match)
scores 0.975 with the megakernel and 0.960 without it.

The Kimi K3 adapter is experimental. Initial ordinary-decode validation used
context length 8192 on GB300/TP8: every listed CUDA-graph width and padding
boundary passed dummy-weight
capture/replay, B1 and B16 passed stock-versus-plugin real-weight canaries, and a
specialized-route/captured-replay canary passed, and the GSM8K quality protocol
matched its frozen stock-path reference score. This is bounded functional and
quality evidence; it does not claim performance, unlisted configurations or production
readiness.

Earlier matched real-weight functional checks covered ordinary decode and
five-draft/q6 DSpark at context length 262144, with DCP off and stock draft
execution. Inputs span 32768, 131072, 240000 and 255872 tokens, plus a
262000-input/128-output deployment-edge request. Long B1 and mixed-B2 requests,
short B4 controls, chunked prefill, page boundaries, retirement, request-slot
reuse and full decode graph replay passed the predefined numerical checks.
Long-input output tokens matched stock; short controls can differ within the
numerical bounds. This is not global bitwise equivalence, an AgentX benchmark,
long-context task-quality validation or a performance claim. Other fixed draft
lengths have component coverage, not equivalent long-context serving evidence.

These results belong to the revisions tested, not automatically to every later
composition. Combined default-OFF ordinary-decode and fixed-q6 checks passed
within the recorded scopes, including near-cap requests, mixed lengths, reuse
and graph replay. A separate registered-default width-64 control matched the
published and private implementations exactly for all 64 sequences and 4,096
selected-token log probabilities. That control does not establish native-stock
accuracy or resolve the five numerical failures in the broader configured-width
experiment. Those failures, including at the already-admitted width 64, are not
waived by component tests or the default-OFF additional-width API. These are
bounded results, not a blanket numerical or performance PASS.

Optional prefetch, embedding and selected-width preparation have bounded
correctness evidence and remain dormant. The TokenSpeed MLA alternative has
matched ordinary-decode and fixed-q6 evidence at TP8, DCP off and context length
8192; TRTLLM remains the default. DCP resources and execution interfaces are
included for explicit integration, with matched ordinary-decode and fixed-q6
evidence at TP8 and context length 8192, including native acceptance, state
commit, hidden-state handoff and graph replay. The DCP adapters require additional
framework interfaces; stock-framework imports and explicit unsupported-activation
rejection are separate CPU contracts, not GPU qualification. No DCP feature
admission is added to the registered spec, and no DCP long-context, optional
combined-ON, default activation or performance claim follows from these checks.

Enabling the plugin does not itself enable DSpark: ordinary decode remains
ordinary decode; five-draft/q6 is the qualified speculative configuration when
DSpark is explicitly configured. An admitted shape is not a claim that every
batch, context and optional-provider combination has passed full-model testing.

### What a covered DeepSeek-V4.1-Flash step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm (`hop_a`, `hop_a_batch2`) | 1–16 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| MoE all-reduce + mHC post + next norm (`hop_b`, `hop_b_wide`) | 1–48 | Hop B: the top-k finalize and the shared-expert add on a Lamport mailbox, whose consumer runs the mHC post, combine and RMSNorm; past 8 rows it writes the next input's MXFP8 scales per 8-row tile, so the next layer's `wqkv_a` runs on the low-latency GEMM. Hop A's widths use hop A's mailbox, the wider ones a second one |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`, `gate_wide`) | 1–48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Shared expert gate_up (`shared_ll`) | 1–48 | FlashInfer's low-latency MXFP8 GEMM with 8-row tiles on the alternate stream from the layer's fork, after a one-pass MXFP8 quant into its row-tile layout (bit for bit the shared linear's own quant); SGLang's clamped SwiGLU and `down_proj` follow |
| Replicated embedding (`embed`) | 1–48 | SGLang's fused lookup on the whole table, all-gathered once at build; no all-reduce |
| Indexer top-k (`attn_indexer`) | 30–48 (batch 5–8) | The decode indexer's top-k in the plugin's forward, reading SGLang's paged indexer metadata and writing its selection into SGLang's buffer; the index query and head weights run on the layer's second stream, under the compressor. At 6–24 rows, chunked metadata or without paged metadata SGLang's indexer runs |
| Engram prefetch (`engram`) | 1–8 (batch 1) | Every Engram layer's lookup and WKV quarter on a side stream; their two all-reduces on the main stream (SGLang's TP group) |
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod the TP size), which sums the partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 70 slots run (`runtime.SHIPPED_SLOTS`): the ones above, the attention chain, hop B's triggers, the DSpark draft step's slots (`draft_hop` among them, which acts at TP8 only), and the 30 slots in the first table below (`runtime.DEFAULT_ON_SLOTS`). Each of those 30 measured faster served at batch 1 (a 95 % interval below zero) and passed the accuracy checks; the ones marked **numerics change** are not bitwise with the path they replace (`runtime.NUMERICS_CHANGE_SLOTS`) and are checked with evals instead. On a default start one of them with nothing to serve stays off (and `attn_q_cluster` and `wo_b_publish_split` stay off at TP4, where they measured slower), and the startup summary names it under "default slots left off". `markov_shard_logits` and `moe_balanced` act only when SGLang asks for them through a switch that is not in SGLang main yet (see the table); without it they leave the step unchanged.

`SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) replaces the default list, for turning slots off, tests and bisecting; empty runs every step function's stock code. To turn one default slot off, list the default slots without it, for example:

```bash
export SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(x for x in s if x != 'wo_b_publish_split'))")
```

To run only the bitwise default slots, also leave out every name in `runtime.NUMERICS_CHANGE_SLOTS`. A slot listed explicitly must build, and a listed slot whose prerequisite is missing refuses the start.

Default-on slots (DeepSeek-V4.1-Flash):

| Slot | Numerics | What it does | Origin |
| --- | --- | --- | --- |
| `gate_route` | bitwise | The router's top-k and the routed pair's route record in one cluster launch at 1-8 rows, target and draft | own |
| `gate_route_quant` | bitwise | With `gate_route`: that launch also writes the routed input's MXFP8 quant (FlashInfer's linear-layout `mxfp8_quantize` bit for bit), so the MoE half forks no routed-quant stream and the top-k launches as the router's programmatic dependent | own |
| `gate_route_early` | bitwise | With `gate_route`: SGLang's tensor-core router GEMM, rebuilt from its `tiny_mma` kernel on SGLang's own schedule, releases the top-k once its weight loads are issued instead of after its cluster reduction; checked against `tiny_gemm_bf16` at build. The top-k then releases GEMM1 after the row's top-k, not right after its own wait, so GEMM1's waiting CTAs no longer take the SMs ahead of the layer's shared expert (`SGL_NVIDIA_MEGA_GATE_ROUTE_EARLY_RELEASE`: 1 after the wait, 2 at completion, 0 at entry) | SGLang kernel; release point as TensorRT-LLM's router GEMM |
| `gemm2_pipe` | bitwise | The routed GEMM2, target and draft, on a pipelined prelude build of the same kernel | derived from muLLM |
| `shared_fold`, `shared_fold_quant`, `shared_fold_early_release` | bitwise | With `shared_ll`: the shared expert's gate_up epilogue writes the clamped SwiGLU and `down_proj`'s MXFP8 input, and releases `down_proj`'s GEMM at its loads | own |
| `hop_mbar` | bitwise | The Lamport consumers' RMSNorm sums cross the cluster by `st.async` onto mbarriers | own |
| `attn_indexer_narrow`, `attn_indexer_fused` | bitwise (ties: in-order pick) | Slot `attn_indexer` at the swap-AB widths too, and the candidate consumers' logits + top-k at 1-8 rows in one cluster launch | own |
| `prologue_fork`, `hc_coeff_fused` | bitwise | The Engram hashes forked at the step start; the prenorm's mHC coefficients in one launch | own |
| `attn_q_norm_gemm` | bitwise | q_norm + MXFP8 quant inside the low-latency wq_b GEMM at 1-8 rows where no indexer reads q_lora | own |
| `attn_compressor_priority` | bitwise | The compressor layers' compressor on a stream at the device's highest priority | derived from muLLM |
| `markov_lowlat` | bitwise | The draft's Markov chain with converged pollers and acquire loads in its exchange, the direct exchange to 4 rows at TP4 and 2 at TP8 | own; acquire idiom as muLLM's |
| `markov_tail` | bitwise | The Markov chain also advances the draft-noise offset, and writes the verify ids `[anchor \| draft tokens]` when SGLang hands them over (an SGLang-side switch that is not upstream yet; without it only the offset add goes) | own |
| `draft_stage0_table` | bitwise | The first draft stage's query and K/V rows gathered in one launch from a per-token table, filled by the live kernels at the first eager forward and checked bit for bit per width; about 1.1 GiB per GPU at TP8 and about 20 s at startup | design derived from TensorRT-LLM |
| `engram_publish` | bitwise | With the Engram prefetch, each Engram layer's WKV columns go into every rank's mailbox as tagged words and the main stream collects them at the layer, in place of the all-reduce of zero-padded rows at layer 1 | derived from muLLM |
| `attn_indexer_logits_q6` | bitwise | With `attn_indexer`: the full paged index logits of 5-7 rows a request (the candidate source and full top-k layers) on DeepGEMM's paged MQA logits kernel with all of a request's rows in one Q block, so the index K streams once per step instead of once per 4 rows; the 6-row, page-128 variant is built at load, any other shape keeps SGLang's call | adapted from DeepGEMM |
| `gemm2_reg_pool` | bitwise | With `gemm2_pipe`: that GEMM2, target and draft, built at 96 registers per thread (`__maxnreg__`) instead of 128, so a co-running kernel's CTA can land on its SM | own; register pool as muLLM's |
| `markov_shard_logits` | bitwise | When SGLang's accept reads only each rank's own draft-logit columns (it passes `step_logits_shard_only` to the draft sampler; not in SGLang main yet), the Markov chain keeps only those columns: no peer staging stores, no publishing fence, no epilogue copy. Without it the chain is built as before | own; sharded draft logits as muLLM's |
| `target_lm_head_xload` | bitwise | With `target_lm_head_mxfp8`: that stream brings the rows of x into shared memory as 16-byte loads all issued before their stores (rows not 16-byte aligned keep the 8-byte loop), so its first MMAs wait one L2 round trip for x instead of several | own |
| `swapab_h8`, `wo_a_g1` | **numerics change** at TP8 | At attention TP8: swap-AB on the 8 local heads and the fused wo_a on the 1 local group, target and draft, in place of SGLang's FlashMLA and einsum (bitwise with the TP4 kernels on the same heads, not with SGLang's TP8 path); no effect at TP4 | own |
| `target_lm_head`, `target_lm_head_mxfp8` | **numerics change** | The target's LM head at 1-8 rows on the low-M kernel, streaming an MXFP8 copy of the vocabulary shard (E4M3 codes, UE8M0 scales of 32 columns, quantized at load) | own |
| `attn_q_cluster` | **numerics change** | q_lora norm + MXFP8 quant + wq_b in one cluster launch at 1-8 rows where no indexer reads q_lora (a different RMS order, an FP8 `mma.sync` k-block fold); a default start takes it at attention TP8 only (8 local heads; at TP4 it measured +21 us per step slower), listed it runs at TP4 too | derived from muLLM |
| `wo_b_publish_split` | **numerics change** | wo_b at 1-8 rows as a cluster split-K FP8 projection whose epilogue publishes into hop A's mailbox (and, with `draft_hop`, the draft hop's), in place of the publishing low-latency GEMM (a different FP32 summation order); a default start takes it at attention TP8 only (1024 wo_b inputs per rank; at TP4 it measured +94 us per step slower), listed it runs at TP4 too | derived from muLLM |
| `draft_shared_ll` | **numerics change** (draft only) | The draft stages' shared expert on the low-latency GEMM with the folded epilogue, as `shared_ll` runs the target's; the draft's proposals may change, the target's verify does not | design as muLLM's draft MoE |
| `moe_balanced` | **numerics change** | At attention TP8 + MoE TP4 x EP2 with SGLang's balanced expert storage (every rank holds its MoE-TP slice of all 384 routed target experts, about 47 GB more weights per GPU; an SGLang-side switch that is not in SGLang main yet): the route record deals each step's distinct experts, in id order, to the two expert-parallel groups in turn, so each group runs half of them whatever the routing (the experts' partial sums meet at hop B in another grouping). Without that storage it has nothing to serve and a default start leaves it off | rule as muLLM's balanced expert groups |

Opt-in slots (run only when listed):

| Slot | Numerics | What it does | Why it is not on by default |
| --- | --- | --- | --- |
| `attn_core_lean` | **numerics change** | At attention TP8, 1-8 rows: a one-launch sparse attention core (cluster merge with the sink and the inverse RoPE in the kernel) in place of swap-AB + combine, target and draft, on SGLang's `v41_new` KV layout (V4.1 KV encodings, P in BF16, its own split and merge order); needs `swapab_h8` and `wo_a_g1`; derived from muLLM | It needs `SGLANG_DSV4_KV_LAYOUT=v41_new`, a server-wide SGLang setting the plugin does not choose. Under `v41_new` swap-AB declines, so TP4 and verify widths above 8 rows run SGLang's attention, which is much slower there; set the variable only for attention-TP8 batch-1 serving |
| `draft_final_collapse` | bitwise | With `draft_hop`: the last draft stage's MoE post also writes the head's mHC collapse | No resolved served gain |
| `draft_stage0_first` | bitwise | With `draft_stage0_table`: the first draft stage's attention is issued from the token ids ahead of the embedding, whose lookup, hc broadcast and the stage's mHC coefficients run on a side stream joined at the stage's attention post | No resolved served gain alone; on current SGLang the draft's metadata preparation still binds the first stage |
| `hop_final_collapse` | bitwise at TP4; **not bitwise at TP8** | The last layer's hop B also writes the head's mHC collapse, so the head runs only its RMSNorm | At attention TP8 + EP2 it changed one DSpark proposal in 4183 verify steps: target outputs identical, one prompt took one extra verify step |
| `attn_cluster_merge`, `attn_index_q_first` | bitwise | Swap-AB's tile merge in the CTA cluster; the index query ahead of wq_b | Small gains not yet confirmed served: `attn_cluster_merge` -3.3 us per step in a 2 + 2-launch A/B; `attn_index_q_first` faster in the one-GPU index-chain bench, not resolved served |
| `attn_wo_a_fp8`, `attn_index_weights_bf16` | bitwise (`attn_wo_a_fp8`: where its FP8 recovery is exact, checked per layer) | wo_a streamed as E4M3 (32x32 UE8M0 scales, half the bytes) and widened exactly in shared memory; BF16 index head weights made beside the head weights, not cast before the logits | In the full batch-1 stack at attention TP8 + MoE TP4 x EP2 (real-text prompts, temperature 1, real routing, acceptance forced to 3.51), server step p50: `attn_wo_a_fp8` -13.1 us [-15.5, -10.7], `attn_index_weights_bf16` -7.7 us [-10.2, -5.1] (one launch per arm, pass-level interval); alone on an earlier stack the gain was not resolved |
| `lm_head_pinned_tactic` | untuned tactics | Both low-M LM heads on one fixed tactic per row count | Measurement aid |
| `draft_lm_head_mxfp8` | **measurement only** | With `draft_lm_head`: the draft's LM head on the MXFP8 copy | The checkpoint ships one BF16 `head.weight` shared by the target and the draft, and InferenceX's review rules forbid a draft head below the checkpoint's precision (`plan.MEASUREMENT_ONLY_SLOTS`) |
| `hop_b_quad_trigger`, `hop_b_cap16`, `draft_embed` | bitwise | Hop B's consumer launched right after the finalize-publish's wait; hop B's row tiles at 9-16 rows only; the draft's tokens from the target's gathered table | No served gain shown (`hop_b_cap16`: +16 us on the batch-1 verify graph) |

Listing `sm148` admits the 148 SMs of a B300 instead of the 152 required otherwise.

Qwen3.5 (`qwen3_5/runtime`) runs the same way: per layer the stage boundary (`boundary/`), the GatedDeltaNet or the attention (`gdn/`, `attention/`) and the MoE (`moe/`), each with a stock implementation that reproduces SGLang's verify step bit for bit. It reads the running forward's metadata and pools through SGLang's EAGLE binding (`sglang.srt.kernel_plugins.nvidia.eagle_binding`). With the megakernel enabled, its shape-bounded bitwise paths run by default. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` overrides that policy for testing and bisecting; an empty value runs the stock paths.

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route_topk`, `mkm_m32` … `mkm_m512` | 1–896 | Materialized stock routing and the shape-selected fused wide routed-expert engines |
| `hop`, `hop_fp8` | 1–15 | The layer boundary's fused all-reduce + residual add + RMSNorm, also writing the next static-FP8 input |
| `multi_request` | 2–128 requests | The target graphs keep every offered request count instead of padding to a coarser width |
| `draft_loop`, `draft_extend`, `draft_multi`, `draft_embed` | 1–128 requests | EAGLE draft entries, exact sharded top-1 and embedding exchange; the pre-FC norms write directly into the FC input |

The remaining kernel families are opt-in through that one slot policy. These change the arithmetic within the tolerances `spec.py` declares:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route`, `route_gemm`, `mkm`, `mkm_wide` | 1–32 | Fused router fronts and the small routed-expert engines |
| `gdn_verify`, `gdn_pdl`, `gdn_conv` | 7–896 | Explicit numeric verify chain and dependent variants; FP8 norm uses the v2 relative-L2 bound of 2e-2 |
| `gdn_core` | 7–224 | ReplaySSM verify, gated norm and FP8 output in one launch |
| `gdn_conv_core` (with `gdn_core`, `gdn_conv`) | 29–224 | `gdn_conv`'s conv inside `gdn_core`'s launch, bit for bit the two launches |
| `hop_mid` (with `hop`) | 16–28 | Extends the one-shot boundary mailbox through four requests instead of returning those widths to the stock boundary |
| `gdn_push` (with `gdn_verify`, `hop`) | 7 | The GatedDeltaNet `out_proj` as the hop's GEMM, whose epilogue pushes the partial into the hop's mailbox |
| `gdn_front`, `gdn_trt_front` | 7–28 | Fused GatedDeltaNet projections and conv before the dependent verify |
| `route_shared` (with `route`) | 1–8 | The shared expert's static-FP8 gate/up in the router front's launch, so the shared expert on the alternate stream waits for that launch only |
| `route_lin`, `skv2`, `mka`, `mka_push` | shape-bounded | Router/attention overlap, split-KV attention and fused attention boundaries |
| `lowm_in`, `lowm_out`, `lowm_se`, `lowm_push` | shape-bounded | Validated low-row projection and shared-expert launches |
| `lowm_ba` (with `gdn_verify`; TP8, TEP8) | 84–112 | GatedDeltaNet `in_proj_ba` on one low-M launch whose cluster splits K, in place of SGLang's linear (cuBLAS); numeric (`runtime.NUMERICS_CHANGE`), with no bound stated in `spec.py` |
| `draft_moe`, `draft_q1`, `draft_xchg`, `draft_hop`, `draft_fin`, `draft_front` | shape-bounded | Fused MTP draft kernels and boundaries |
| `draft_skv2` (bitwise) | 1 request | The draft extend's KV-length graph variants, in which `draft_skv2_cp` serves one long request; below its length the extend's rows run the stock decode |
| `bt_tail` (bitwise) | 16–703 | The layer and MoE inputs' fused all-reduce + residual add + RMSNorm past the hop's rows, on FlashInfer's balanced (BT) protocol with its last launch replaced by one that also writes the next static-FP8 input; FlashInfer's outputs bit for bit. With `bt_tail_early` / `bt_tail_early_fin` the MoE input's / layer input's last launch releases its dependents at its wait. It needs the FlashInfer its launches were checked against (flashinfer-python 0.7.4, `FLASHINFER_SOURCES`) and a BF16-output workspace; with another FlashInfer, or at a width where either point does not route the three-launch BT (with the TP8 / TEP8 routes, the layer input below 24 rows), it stays stock |
| `gemm_ar` (bitwise, with `bt_tail`; TP8, TEP8) | 33–128 (none at TP4: the kernel takes K up to 1024, TP4's `out_proj` / `o_proj` have 2048) | The mixer's `out_proj` / `o_proj` runs inside the MoE input's all-reduce launch on `bt_tail`'s point: the FP8 projection, a two-shot all-reduce, the residual add, the norm and the MoE's static-FP8 input in one launch, at the BT path's rounding points (cuBLAS + FlashInfer's BT outputs bit for bit). A projection the kernel refuses, or one whose all-reduce does not run on `bt_tail`, runs on cuBLAS |
| `fc2ar` (bitwise, with `bt_tail` and `moe_engine`; TEP8) | 35–128 | The MoE engine writes its FC2 rows into a Lamport buffer and releases the next boundary point early; that point's one-shot all-reduce polls the rows while the engine runs and folds the finalize, the residual add, the norm and the next static-FP8 input in one launch, at the BT path's rounding points (FlashInfer's BT outputs bit for bit). Capture-mode steps only; a width without a BT route stays on BT. The Lamport buffer is made at load, for the widest covered call at FlashInfer's widest FP4 `tile_n`, and every `tile_n`'s call shares it |
| `draft_select` (with `draft_extend`) | 1–128 requests | The draft extend's selected rows in one row-copy launch at TP4, TP8 and TEP8; bitwise `torch.index_select` on in-range indices |
| `gumbel_fast` (with `sharded_gumbel`) | sampled verify rows | `sharded_gumbel`'s noise drawn four per Philox call (another stream, the same law), as one CUDA launch where the rows are aligned quads |
| `verify_head` (with `sharded_gumbel`) | 70–128 | The vocabulary-sharded verify's LM head on the tensor cores in place of the GEMM, through SGLang's `verify_shard_head` hook; numeric (`runtime.NUMERICS_CHANGE`), with no bound stated in `spec.py` |
| `verify_head_keys` (with `verify_head`, `gumbel_fast`) | 70–128 | The head also leaves the T = 1 and greedy keys of its logits, which `gumbel_fast`'s launch takes for rows at temperature 1 or 0 |

The MTP draft's norms in `draft_hop` and `draft_fin` take the rounding of SGLang's
GemmaRMSNorm (`gemma_fused_add_rmsnorm`, multiplier 1 + the raw weight in fp32), as the
stock draft does: the post-attention norm on the draft hop (the add-norm consume), the
final norm after a deferred MoE finalize on the hop, and, past the hop at TEP8, both norms
after SGLang's MNNVL CuTe DSL workspace, which then only sums. Past that workspace's LL
and BT routes (`draft.layer.STOCK_SUM_MAX_ROWS`, 703 rows), which add the ranks' rows in
rank order, those steps run SGLang's all-reduce and GemmaRMSNorm.

Opt-in variants of the CP slots (`skv2_cp`, `draft_q1_cp`, `draft_skv2_cp`) at TP8 and TEP8; their kernel tests in `test/kernels` compare the outputs with the default build's, byte for byte:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `cp_smv2`, `cp_ptxas_o1` (with a CP slot) | as its CP slot | Every CP-lite partial built with softmax v2 (on unmasked tiles the row max from `tcgen05.ld.red`, then each 32-column chunk reloaded from tensor memory and exponentiated two lanes per `fma.rn.f32x2` / `add.rn.f32x2`) / with ptxas `-O1` |
| `cp_merge_plain_wait` (with `skv2_cp`) | `skv2_cp`'s multi-request verify | The multi-request target verify's exact per-head CP-lite merge polls its sources with the plain merge's volatile-load wait instead of the per-thread spin |
| `cp_q_push` (with `skv2_cp` and `mka`) | `skv2_cp`'s multi-request verify | The front's q writer stores the FP8 q rows to the CP-lite group's members as the q gather's low-latency words, so the ungated exact call's gather only waits for them |
| `cp_qw_entry` (with `cp_q_push`) | `skv2_cp`'s multi-request verify | The pushing q writer releases its dependents at its entry, so the gather's wait half and the partial launch under it; each waits before reading its outputs |
| `cp_merge_cg` (with `cp_q_push`) | the CP slots' early merges | Every early per-head CP-lite merge reads its lengths, epochs, the step's q / K / V and the gate through L2 (`ld.global.cg`) |
| `cp_lam_q` (with `cp_q_push`) | `skv2_cp`'s multi-request verify | The pushed q words move as Lamport words (plain 32-bit data in a sentinel-filled region of their own, one writer a word; the wait half stores the sentinel back over the words it polled) instead of 8-byte low-latency words |
| `cp_q_gather` (with `cp_q_push` and `cp_lam_q`) | `skv2_cp`'s multi-request verify through 63 rows | The pushing q writer also polls the group's Lamport q words of each row it pushed and lands CP-lite's `q_all`, prefix lengths and epochs itself, so the ungated exact call launches no gather and its partial launches under the writer |
| `cp_o_push` (with `cp_q_push` and `cp_merge_plain_wait`) | `skv2_cp`'s multi-request verify, up to 64 requests | The fused CP-lite partial writes each finalized (row, head) partial and its LSE into the per-head merges' mailboxes as low-latency words and releases the merge once its stream is done; the merge only polls every source's words in one pass, with its dependency wait moved to its end |

The tuned Qwen3.5 TP8/TEP8 serving policy explicitly includes `hop_mid` whenever it
includes `hop`.  It remains opt-in rather than part of the bitwise default policy;
an explicit `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` value replaces, rather than extends,
that default list.

The TP8 plan (`topology.plan`, aggregated and PD decode) also runs `route_lin`
whenever the policy holds the slots it runs through (`route`, `mkm`, `hop`,
`gdn_push`, `mka`, `mka_push`, and `lowm_se` with `route_shared`): its router logits
stay within 1.05x stock's RMS error against FP64 on the same quantized operands.
No router operand is rounded to BF16, and the logits' RMSNorm scale comes from the
FP32 pre-norm row with each hidden partial's largest BF16 rounding residual per
32-channel group added back (one 4-byte cell per rank, row and group in the router
mailbox). Its packing checks each rank's own weights (a BF16 router, an FP16 scale
for the folded `out_proj` core, BF16-exact gamma times that scale); when any rank
declines, every rank keeps the stock router and logs a warning naming each declining
rank's reason.
The same TP8 plan adds `mkm_wide_bal` for `mkm_m256` / `mkm_m512`: balanced
CTA assignment at 129–512 verify rows, with bitwise-identical outputs.
TEP8, DEP8, TP2 and TP4 run exactly the requested slots their topology supports.

`gdn_conv_core` runs `gdn_conv`'s causal conv update inside `gdn_core`'s launch
(`sm103/gdn/core.py` with `conv`), with the arithmetic of SGLang's Triton conv, so its
conv output, rolling conv states, windows and the core's outputs equal the two launches'
bit for bit. The last CTA of a key head to read the prior q / k state rolls it, through
int32 counters per request and key head of each GDN layer (made at build), so a layer's
launches must not overlap: there is one verify stream, and the spec does not accept
SGLang's batch overlap. The launch releases `out_proj` at its entry at every width.

With `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` and no slot list (the measured
stack), TP8 and TEP8 on GB300 (compute capability 10.3, 152 SMs) run the measured plan
(`runtime.MEASURED_PLAN`): the shipped slots plus the 31 opt-in slots of
`runtime.MEASURED_OPT_IN` (not `route_gemm` or `draft_xchg`), the BT tail with its
two early releases (`runtime.MEASURED_BT_TAIL`; stock without flashinfer-python 0.7.4) and
the CP-lite variants of `runtime.MEASURED_CP_LITE`, the draft slots of
`runtime.MEASURED_DRAFT` and the verify keys provider's options `runtime.MEASURED_HOOKS`
(`gumbel_fast`, `verify_head`, `verify_head_keys`), and at TEP8 also `gemm_ar` and
`fc2ar` on the BT tail's points (`runtime.MEASURED_BT_AR`). The topology filters and
the TP8 plan then apply as for a slot list. The plan also runs options, which are no
slots (`runtime.MEASURED_OPTIONS`; the draft's are in the table below): `gdn_value_split`,
where `gdn_core` runs each
(request, value head) of the TP8 shard's 8 value heads on a cluster of two CTAs at
7–112 rows (the norm's sums are combined in the one-CTA order, so the same bits), and
`gemm_tactics`, where the FP8 projection GEMM of `in_proj_qkvz` and `qkv_proj` takes a
cuBLASLt algorithm per verify width from a table keyed by topology, GPU and cuBLASLt
version (`gemm_tactics.py`). Each table algorithm must equal FlashInfer's heuristic
algorithm bit for bit on a check GEMM of the layer's weights, else the width runs that
heuristic algorithm; another cuBLASLt version, or a FlashInfer whose cuBLAS FP8 runner
lacks a name the module reads, runs FlashInfer's BMM; each with one WARNING. Unlike the
rest of the build, a width resolves at its first call outside a CUDA graph capture, and
each rank resolves on its own (a rank whose check fails runs the heuristic algorithm,
whose bits the others' algorithm equals). Per row, of
those it turns on, `runtime.MEASURED_BITWISE` (`mkm`, `mkm_wide`, `mkm_wide_bal`,
`gdn_pdl`, `gdn_conv`, `gdn_conv_core`, `mka_push`, `bt_tail`,
`bt_tail_early`, `bt_tail_early_fin`, `gemm_ar`, `fc2ar`, `cp_smv2`, `cp_ptxas_o1`,
`cp_merge_plain_wait`, `cp_q_push`, `cp_qw_entry`, `cp_merge_cg`, `cp_lam_q`,
`cp_q_gather`, `cp_o_push`, `draft_skv2`, `draft_select`, `gdn_value_split`, `gemm_tactics`,
`gdn_prefill_front`, `gdn_prefill_track_snapshot`, `moe_prefill_input_nvfp4`,
`moe_prefill_add_norm`, `moe_prefill_noclone`, `prefill_gate_fp8`, `prefill_act_fp8`,
`gdn_prefill_norm_fp8`, `prefill_norm_fp8` and the bitwise draft rows below) keep the
step's bytes given the slots they need, and
`runtime.NUMERICS_CHANGE` change the arithmetic; each has a row in
`spec.NUMERIC_TOLERANCES` (empty where no bound is stated) or a tolerance claim in the
draft spec, except `hop_mid` (`runtime.NUMERICS_UNBOUNDED`).
Numerics exceptions (`spec.NUMERIC_EXCEPTIONS`): at TP8 / TEP8 (CP 4 x 4 query heads a
rank, CP-lite's LL path), `skv2_cp`'s target verify keeps two E4M3 terms
of P from one request (BF16 + residual partials)
through 64 (FP16 partials). At 65–128 requests, on CP-lite's two-slot
plans, it keeps one E4M3 term of P and BF16 partials. At TP4 (CP 2 x 8 query heads a
rank, whose rows do not fit the LL words) every width runs CP-lite's fold path as
before, with BF16 partials. In both, the exception declares a single request's error
unbounded (`worst_request`).
Every other topology or GPU runs the shipped slots, and an explicit slot list keeps the
policy above. The startup log names the plan's key and its slots. Where the measured plan
runs a CP slot, every CP-lite partial on the one-slot balanced plans also builds with a
lean fixed phase and an LSE-first combine, on the one-request-per-lane plan (up to 32
requests) with a per-CTA plan cache, and with `cp_o_push` reads its push epochs right after
its dependency wait (`q35_cp_fused.set_cp_phase`): data movement and issue order only,
compared byte for byte with the default build in `test/kernels`; elsewhere it builds as
before.

The rows that join the shipped slots per key (`runtime.MEASURED_PLAN`);
`test/unit/test_qwen3_5_plan_snapshots.py` pins each key's resolved slots, and that
the megakernel alone or an explicit slot list resolves every key as upstream's build:

| Key (GB300) | Rows that join the shipped slots |
| --- | --- |
| `tp8` | `MEASURED_OPT_IN`, `MEASURED_BT_TAIL`, `MEASURED_CP_LITE`, `MEASURED_DRAFT`, `MEASURED_HOOKS` and the options `MEASURED_OPTIONS`; the TP8 plan adds `route_lin` and `mkm_wide_bal` |
| `tep8` | The same and `MEASURED_TEP8`, `MEASURED_BT_AR`, `MEASURED_DRAFT_TEP8` and the options `MEASURED_TEP8_OPTIONS`, less `EP_STOCK_SLOTS` (expert parallelism keeps SGLang's MoE there) |
| `dep4` | `MEASURED_DEP4`; DEP4 declares none of the shipped slots |
| `p_tp1_pp2` | `MEASURED_PREFILL` (`gdn_prefill_front`, `gdn_prefill_track_snapshot`, `moe_prefill_input_nvfp4`, `moe_prefill_add_norm`, `moe_prefill_noclone`, `prefill_gate_fp8`, `prefill_act_fp8`, `gdn_prefill_norm_fp8`, `prefill_norm_fp8`, `attn_prefill_qk_rows`), the prefill slots declared at the key |
| Any other key or GPU | None: the shipped slots (the prefill body seam is still offered: see below) |

At TEP8 on GB300 the measured plan also runs the routed body's own rows
(`runtime.MEASURED_TEP8`; TEP8 only, each with the slots it needs). As with the other
opt-in slots, an explicit slot list that names them runs them at TEP8 on any GPU the
spec admits.

| Slot | Rows | Kernel |
| --- | --- | --- |
| `moe_engine` | every TEP8 width | The routed body on a persistent NVFP4 MoE engine (FC1, SwiGLU, NVFP4, FC2 in one launch) in place of FlashInfer's two GEMMs; the same FC2 rows bit for bit |
| `moe_ring` (with `moe_engine`, `route_lean`) | 64–128 | The engine streams the first FC1 stages of the experts the layer used at its previous call before its dependency wait; bitwise |
| `route_big_gemm` (numeric) | 33–192 (from 17 with `route_big_mid`) | Route-big's router logits from a split-K tcgen05 GEMM in a fixed, batch-independent sum order |
| `route_lean` | 17–448 | Route-big's lean variant: the routing cluster's counts by bulk copies onto mbarriers, the engine released at route-big's start; bitwise |
| `moe_engine_route` (with `moe_engine`, `route_big_gemm`) | 33–63 | The router GEMM keeps each token's top-10 per expert tile and the engine writes the routing tables itself, in place of route-big; bitwise |
| `route_big_mid` (numeric) | 17–32 | Route-big writes the routed body's tables at tier M's widths in place of route_topk's per-token launch, bit for bit on the same logits; with `route_big_gemm` the router GEMM runs from 17 rows |
| `lowm_se_tep8` (numeric, with `lowm_se`) | 33–128 | `lowm_se`'s two low-M launches for the shared expert at the routed body's widths |
| `lowm_se_wide` (numeric, with `lowm_se_tep8`) | 129–160 | The same launches over a 128-row and a 32-row chunk; each row the 128-row launch's bit for bit |
| `lowm_se_coresident` (with `lowm_se_tep8`) | 84–128 | The shared expert's gate_up releases its down launch at its end instead of after its weight loads; bitwise |

The MTP draft's own rows of the measured plan (slots of `runtime.MEASURED_DRAFT` at TP8
and TEP8 and `runtime.MEASURED_DRAFT_TEP8` at TEP8; options, which are no slots, of
`runtime.MEASURED_OPTIONS` and, TEP8 only, `runtime.MEASURED_TEP8_OPTIONS`):

| Row | Kind | Rows | What it runs |
| --- | --- | --- | --- |
| `embed_local` | option, bitwise | every verify width | The verify's embedding rows from a replicated BF16 table (the vocabulary-parallel shards all-gathered at build) in one gather launch, in place of the vocabulary-parallel lookup and its all-reduce; steps aside under `SGLANG_ENABLE_EMBED_REPLICATION` |
| `draft_embed_local` (with `draft_loop`) | slot, bitwise | 1–128 requests | The draft's embedding rows from the same table (one table a rank when the draft shares the target's embedding), in place of `draft_embed`'s exchange; steps aside as `embed_local` |
| `draft_fc_input` (with `draft_embed_local`) | slot, numeric | 1–128 requests | The fc input's table gather and both pre-fc Gemma RMSNorms in one launch |
| `draft_top1_cfg` | option, bitwise | 2–128 requests | `draft_multi`'s sharded top-1 sends its row keys with plain stores to every peer's buffer and shortens its last CTA's chain; the same tokens |
| `draft_head_lowm` (with `draft_multi`) | slot, numeric | 1–8 requests | The multi-request steps' LM head on the low-M GEMM at the widths where it beat cuBLAS in a build-time race (rank 0's picks on every rank) |
| `draft_head_keys` (with `draft_head_lowm`) | option, numeric | 9–16 requests | The multi-request steps' LM head on the tensor-core head with the row argmax in its epilogue (no logits written) and one key exchange over unicast stores, in place of the cuBLAS head and `draft_multi`'s sharded top-1; the same tokens on the same bf16 logits |
| `draft_router_lowm` | slot, numeric | 1–16 rows | The draft MoE block's router GEMM on the low-M GEMM at the widths its build-time race took |
| `draft_moe_engine` | TEP8 option, numeric | 1–32 rows | The draft's routed BF16 experts in one persistent launch (routing in every CTA's prologue, FlashInfer's Renormalize arithmetic) in place of FlashInfer's `trtllm_bf16_moe`, through the draft MoE block's deferred forward |
| `draft_moe_wide` (with `draft_moe_engine`) | TEP8 option, numeric | 33–128 rows | The engine's wide build: four token words a local expert, segments of 64 rows |
| `draft_shared_lowm` | TEP8 option, numeric | 2–16 rows | The draft's shared expert on two low-M launches (gate_up with the SwiGLU in its epilogue, then down) |
| `draft_fc_tp` (with `draft_fc_input`) | TEP8 slot, numeric | 1–128 rows | The steps' and extends' fc as the column-parallel fc plus one all-reduce and the layer's input norm (the draft hop's add-norm, else SGLang's MNNVL CuTe DSL sum) |
| `draft_fc_push` (with `draft_fc_tp`) | TEP8 option, numeric | the hop's widths | The fc window GEMM's epilogue pushes its columns into the draft hop, whose consume reads each column from its owner's slot |
| `draft_oproj_push` (with `draft_hop`) | option, numeric | 2–16 requests | The multi-request steps' o_proj on the low-M GEMM pushing its partial into the draft hop, so the publish launch goes |
| `draft_early_hops` | option, bitwise | 1–16 rows | The fused fc input, the draft hop's layer-input consume and its FFN-input consume release the next GEMM (low-M fc window, TGV qkv, low-M router) at entry; launch order only |
| `draft_q_push` (with `draft_front`, `draft_multi` and `cp_q_push`) | slot, bitwise | the multi-request calls CP-lite takes | The draft front's q writer pushes the steps' and extends' FP8 q rows to the CP-lite group as the q gather's words, so CP-lite's gather only waits; the same `q_all` bytes |
| `draft_q_gather` (with `draft_q_push` and `cp_lam_q`) | slot, bitwise | the multi-request calls CP-lite takes | The pushing draft front also polls the group's Lamport q words of each row it pushed and lands CP-lite's `q_all`, prefix lengths and epochs itself, so the call launches no gather and its partial launches under the front |
| `draft_cp` (with a CP slot) | slot, numeric | 2–64 requests | SGLang's draft worker offers its TRT-LLM MHA decode and draft-extend calls on the FP8 KV cache through its `draft_attention` hook; they run as CP-lite's precise call (two-term E4M3 P, BF16 + residual partials), and the stock attention past 64 requests. Where it runs, the megakernel draft's own CP-lite calls (`draft_q1_cp`, `draft_skv2_cp`, `draft_multi`) take the same precise call and cutover; without it they run as before |
| `draft_gate_fold` (with `draft_multi` and `draft_cp`) | slot, bitwise | 2–64 requests | The precise multi-request call's per-head merge applies the attention output gate in `fused_sigmoid_mul`'s arithmetic, so no gate launch follows; the same bytes |
| `draft_o_push` (with `draft_multi` and `draft_cp`) | slot, bitwise | 2–64 requests | The precise multi-request call's fused partial pushes its (O, LSE) and residual words into the members' per-head merge mailboxes, and the merge, still launched after its wait, only polls them; the same words |
| `draft_merge_early` (with `draft_o_push`) | option, bitwise | 2–64 requests | That polling merge runs its causal block and polls under the partial, its dependency wait moved to its end (before the gate read and every store), and the partial releases it after its stream; launch order only |
| `draft_hilo_trig` (with `draft_merge_early`) | option, bitwise | 2–64 requests, draft steps (1 row a request) | On the draft steps that early merge also releases its dependents at entry, so the draft o_proj (with `draft_oproj_push`, the draft hop's pushing GEMM) loads its weights under the merge; each reads the merge's output only after its own dependency wait; the draft extend's merge keeps the early form without it; launch order only |

With the plan's option `gdn_commit`, the TP8 or TEP8 build offers SGLang's
`SupportsRecurrentCommit`: after each chain verify it runs SGLang's ReplaySSM commit
(`gdn/commit.py`, the cursor advance and the conv rollback as SGLang's), with the
circular ring's compact commit as a CUDA kernel (`sm103/gdn/commit.py`). Its
checkpoints can differ in rounding from SGLang's Triton commit's (numeric, in
`runtime.NUMERICS_CHANGE`). It takes the pools at the first covered step: the compact
BF16 ring of length 16 with int32 cursors; on other pools it logs why and SGLang
commits as without it.

DEP4 (TP4, attention DP4, MoE EP4) is admitted only under the measured stack, which
lifts the `moe_a2a_backend` and `flashinfer_allreduce_fusion_backend` pins and keeps the
`flashinfer_trtllm` runner pin; the megakernel alone refuses it as before. On GB300 its
plan (`runtime.MEASURED_DEP4`) runs `gdn_verify` and `gdn_core` with three numeric rows
of its own, and the rest of the step (MoE, its exchange, the boundary and the draft)
stays SGLang's:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `gdn_core_tc` (with `gdn_core`) | 7–896 | `gdn_core`'s launch on tcgen05 (`sm103/gdn/core_tc.py`): persistent CTAs over (request, value head) pairs |
| `gdn_commit_fold` (with `gdn_core_tc`) | 7–896 | That launch first folds a flush row's committed ring entries into its checkpoint; the commit after the verify (`SupportsRecurrentCommit`) then only advances the ring cursors and rolls back the conv state. It runs only with `--disable-radix-cache`, without unified memory or the parity check, and where every offered verify width runs `gdn_core`; it is decided before the build compiles. A retracted request is not backed up to the host (`SupportsStateSlotReset.backs_up_state`) |
| `gathered_cdf` | gathered verify rows | Temperature-only rows of the gathered verify logits draw their target tokens by inverse CDF (`sm103/gathered_cdf.py`) through SGLang's gathered-logits sampler policy: the same law as the stock sampler on another random stream. Its kernels and seed counter are built at load |

TEP8's routed body takes its NVFP4 MoE tactic per token bucket from a fixed table
(`moe/tactics.py`) on FlashInfer 0.7.1, SM103 and the routed experts' MoERunner
signature; any other configuration or bucket asks the autotuner, and the log names
each width's source. `SGLANG_NVIDIA_MEGAKERNEL_MOE_TACTICS` switches the table; unset,
it follows `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`.

The router-only `route`, `route_topk` and `route_gemm` slots support TP2, TP4, TP8 and TEP8 with the TRTLLM runner and no all-to-all. PD decode TP2 uses the same local row counts: batch 6/10 gives 42/70 verify rows. These fronts leave the expert GEMMs with the stock runner, so TP2 needs no fused expert engine. `route_shared` keeps its TP8/TEP8 projection bounds; DEP8 keeps stock MoE dispatch and routing.

`mka` uses the fused projection kernels through 14 verify rows on TP8 and 8 on
TP4. At wider rows, and at every TP2 width, it keeps SGLang's projection GEMMs
and fuses Q/K normalization, RoPE and the FP8 KV write, then the output gate
and static FP8 quantization. TP2 has 16 query heads and one KV head per rank;
TP4 has 8/1 and TP8 has 4/1. The same front path serves PD decode TP2 and DEP8 (32/2 per rank).
`multi_request` coverage is required for more than one request. `skv2` keeps
its one-request, one-KV-head scope; `skv2_cp` additionally requires replicated
KV heads across ranks.

The sampling and acceptance after the verify stay SGLang's own code. `gdn_prefill_split_conv` is an explicit bitwise slot for ordinary text prefill on TP1/PP4 with the data-only prefill binding: it splits the native conv by Q/K/V channels above 8192 channels, retaining native recurrence, norm and state writes. Unsupported layouts use SGLang; an empty slot policy calls the original prefill forward directly. The measured stack also admits the prefill role at TP1/PP2 (key `p_tp1_pp2`, `spec.pd_reasons`). On GB300 its measured plan (`runtime.MEASURED_PREFILL`) runs `gdn_prefill_front`, declared at TP1/PP2 only: on SGLang's FlashInfer prefill route (`--linear-attn-prefill-backend flashinfer`), each GDN layer's causal conv, Q/K/V split, Q/K L2 norm and gates come from one CUDA launch and two small Triton launches (`sm103/gdn/prefill_front.py`) with the bytes of SGLang's `causal_conv1d_fn`, `gdn_prefill_qkv_prepare_fwd` and `fused_gdn_gating`; the layer's state read, the FlashInfer chunked prefill, the state gather and write-back, the prefix-track copies (with SGLang's one-launch state copies where its predicates take them), the gated norm and the output projection keep SGLang's order (`gdn/prefill_fi.py`). Under the measured stack the spec offers SGLang's prefill body seam (`prefill_graph`, `SupportsPrefillBody`), so the slot runs in the layer module's forward that SGLang's breakable prefill CUDA graphs capture per token bucket (`--cuda-graph-backend-prefill breakable`), or eagerly without them: each GDN layer's core is a graph break, as SGLang's own linear-attention core is, and runs on the live batch's real rows; the rest of the body is captured as SGLang's. The build checks SGLang's seams and every GDN layer's shape and weights, compiles the CUDA kernel and runs the Triton launches once; a layer it cannot take keeps SGLang's forward for the whole prefill. A step on another prefill kernel, a multi-item or strided state layout or recomputed checkpoints, and a layer whose tensors the front does not take, run SGLang's own core there before any state is written, each reason logged once. The plan's `gdn_prefill_track_snapshot` (bitwise, in the front's extend) has the FlashInfer chunked prefill store only the checkpoints the prefix-track copies read: FlashInfer's `chunk_gated_delta_rule` takes a destination row per checkpoint entry (`checkpoint_state_indices`, -1 skips the store), so the tracked requests' entries land in a buffer of one row each, which SGLang's track copy reads, and no other entry is written (`gdn/fi_track_snapshot.py`). The kernel and the states it stores are SGLang's call's; FlashInfer runs destination rows on its native (non context-parallel) kernels only, and the slot runs on the SM100 family only, so a step whose batch FlashInfer's own heuristic sends to its context-parallel kernel keeps SGLang's all-row checkpoints. The FlashInfer files the slot rests on are pinned by sha256 (flashinfer-python 0.7.4); another FlashInfer keeps SGLang's checkpoints, with one startup line. The slot-mode kernel is its own FlashInfer build: it compiles on the first step that runs it, as the measured stack runs it, and the slot adds no prewarm. With the body offered, SGLang waits for a layer-wise host load-back to finish before a prefill graph replays, where its stock body waits per layer. The spec decides the offer at import, before the role, the key and the GPU are known, so under the measured stack every process offers the body (the decode builds and every other key plan none): any of them that replays prefill graphs (a TP1/PP4 prefill, an aggregated server with breakable prefill graphs) waits that way too. What the wait costs on prefills with host-loaded prefix hits is not measured. In that body, slot `moe_prefill_input_nvfp4` (declared at TP1/PP2 only) serves the MoE layers on SGLang's CuTe DSL NVFP4 MoE (`--moe-runner-backend flashinfer_cutedsl`, no all-to-all, a static activation scale): the post-attention Gemma fused add-RMSNorm (FlashInfer's `FusedAddRMSNormKernel`, which it extends, `sm103/moe/prefill_input_nvfp4.py`) also writes the routed experts' NVFP4 input with `fp4_quantize`'s bytes, and the routed experts take it in place of their own quantization (`moe/prefill.py`); the norm output and the residual keep the stock bytes. The build checks every MoE layer's FFN input path and block (a layer it cannot take keeps SGLang's, with one startup line) and compiles the kernel; a batch the mirrored path does not cover runs SGLang's FFN input. Slot `moe_prefill_add_norm` (declared at TP1/PP2 only, in the same body) leaves each covered MoE block's shared-expert gated add (SGLang's Triton `fused_gate_sigmoid_mul_add`) to the next layer's input norm on the stage: the block launches the same Triton kernel in its `GATE_ONLY` mode with the stock launch shape, and the next layer's input Gemma fused add-RMSNorm (`sm103/moe/prefill_output_add_norm.py`, which extends FlashInfer's `FusedAddRMSNormKernel`) does the gated add in its load step; the norm output and the residual keep the stock bytes. A block defers only where its FFN exit passes the output through unchanged and the next layer takes it on the mirrored attention input; anything else (the stage's last layer, a capture, another batch) completes the gated add with the stock kernel first. Slot `moe_prefill_noclone` (declared at TP1/PP2 only, in the same body) gives a covered block's shared expert the block's input itself in the dual-stream order (under CUDA graph capture), where SGLang gives it a clone: the clone guards a routed path that writes into its input, and the CuTe DSL runner's standard path, which the build requires, only reads it. In that body, the measured plan's static FP8 hand-offs (`runtime/prefill_fp8.py`) have the op before a ModelOpt static per-tensor FP8 linear write that linear's E4M3 input, which the linear takes through its prequantized-input path instead of quantizing it again; each writes the bytes of SGLang's `static_quant_fp8` on the op's stock output, so the linear's output is the stock one: `prefill_gate_fp8`, a full-attention layer's output gate writing `o_proj`'s input (`sm103/attention/gate_quant.py`: the verify step's gate kernel on SGLang's launch of the gate); `prefill_act_fp8`, an MoE layer's shared expert (or a dense MLP) writing `down_proj`'s input from its SiLU-and-mul (`sm103/moe/prefill_act_fp8.py`, built through SGLang's JIT loader on SGLang's activation header; SGLang's MoE block calls the shared expert itself, so the build gives such an MLP the package's `Qwen2MoeMLP.forward`, which takes the fused path only while the body runs); `gdn_prefill_norm_fp8`, a GDN layer's gated RMSNorm writing `out_proj`'s input in the front's step (`gdn/norm_fp8.py`: the verify step's gated-norm kernel on SGLang's prefill launch, with an E4M3 store that keeps the BF16 store's register layout); `prefill_norm_fp8`, a layer's input Gemma add-RMSNorm also writing the static FP8 input of `in_proj_qkvz` or `qkv_proj` (`sm103/gemma_norm_fp8.py`: FlashInfer's CuTe DSL fused add-RMSNorm, the kernel SGLang's `GemmaRMSNorm` runs, with an E4M3 epilogue; the build gives the norm and its linear the package's forwards: the norm's runs the dual kernel where `GemmaRMSNorm._forward_impl` runs the fused add-RMSNorm alone, as on the call SGLang's attention read makes, `norm(x, residual, None)`, reading the linear's input scale at the call, and the linear takes the E4M3 copy only when its input is the norm's output itself, unmodified since; where `moe_prefill_add_norm` does the previous block's gated add in that norm, its kernel writes the copy, `sm103/moe/prefill_output_add_norm.py`'s E4M3 epilogue). A layer whose linear is not a static per-tensor FP8 linear keeps SGLang's forward, logged once at the build. The plan also runs `attn_prefill_qk_rows`, declared at TP1/PP2 only: each full-attention layer's Q / K GemmaRMSNorm + partial RoPE runs over tiles of 8 (token, head) rows of one kind, Q and K in one launch (`sm103/attention/qk_norm_rope_rows.py`), where SGLang's `fused_qk_gemma_rmsnorm_rope_gate` runs one program per (token, head), and `fused_sigmoid_mul` reads the output gate in place as a strided view of the QKV projection instead of SGLang's copy (where `prefill_gate_fp8` took the layer, its FP8 gate reads that view and writes `o_proj`'s input); the rest of the layer is SGLang's, in its order (`attention/prefill_rows.py`). It is numeric (`runtime.NUMERICS_CHANGE`): only each head's FP32 sum of squares runs in another order, so an element can round to the neighbouring BF16 value; given Q and K, the gated output's bytes are SGLang's. If the tiles do not take one full-attention layer's norms or head shape, or the model is not BF16, every full-attention layer keeps SGLang's for the whole prefill, with one startup line. Either slot runs the prefill body on its own; without the front, the GDN layers run SGLang's layer forward there. With the megakernel alone or an explicit slot list, TP1/PP4 stays the only prefill shape. In an SGLang that has the switches, `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1` runs the sampling and acceptance in CUDA graphs, and `SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1` captures the GatedDeltaNet extend inside the breakable prefill CUDA graphs.

GLM-5.3 (`glm5_3/`) follows the same layout: the executor's layer loop runs the hops between the sublayers (`boundary/`), the attention (`attention/`) and the MoE (`moe/`), then the logits (`logits/`), each with a stock path that reproduces SGLang's verify step bit for bit, and CuTe DSL kernels (`sm103/`) where a slot serves its width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| Attention (`attention`) | 4–16 | The q_a / kv_a projection on a small-M GEMM (with the indexer's key / head-gate projection, up to 8 rows), the MLA prologue (norms, q_b, W_UK absorption, RoPE and FP8 query, the MLA and index-K cache rows), the indexer query, the sparse MLA over the fused top-k page table and the value up-projection; the indexer's top-k stays SGLang's |
| Routed and shared experts (`moe`) | 4–16 | One small-M NVFP4 engine launch for the routed experts and the BF16 shared expert, for 1, 2 or 3–4 groups of up to 4 rows (a request's rows with 4 draft tokens; 6 rows run as 4 + 2, 12 as 4 + 4 + 4) |
| All-reduce + residual + RMSNorm (`hops`) | 4–16 | One hop per tensor-parallel sum over symmetric memory: `o_proj` with the router on MoE layers, the MoE finalize folded into the MLP-output all-reduce, and the final norm |
| Shared-top-k q_a / kv_a (`qkv_a_small_m_shared`) | 4–8 | The small-M projection on the layers that reuse the previous layer's top-k (one packed weight copy each) |
| Deferred input norm (`dnorm`) | 4–8 | Those layers' input RMSNorm applied in the projection's epilogue from the fold hop's block sums (moves the rounding points; its tolerances are in `glm5_3/spec.py`) |
| NextN draft step (`draft`) | every captured draft-decode and draft-extend graph | The draft's embedding gather + eh norms in one kernel on the replicated table (any row count), the column-sharded `eh_proj` (1–4 rows, and 8, 12 and 16 with 4 draft tokens or 6 and 12 with 6; gathered by the hops' all-reduce on the target's mailbox), the distributed top-1 in place of the logits all-gather (up to 8 rows), and the draft layer's MLA prologue (up to 16 rows); each kernel's other row counts run SGLang's code |
| NextN draft-extend row select (`draft_rowselect`) | every captured draft-extend graph | After the draft layer's attention (which writes every row's KV and indexer key), the MLP, its all-reduce and the final norm run on the rows the draft's LM head reads (`select_index`, one per request) instead of every verify row; the kept rows are the full-row stream's bit for bit, the draft MoE then picks its kernels by the smaller row count (tolerances in `glm5_3/draft.py`) |

At TP4 (one GB300 tray, 16 attention heads per rank) the verify step's groups are composed per width from kernel units (`glm5_3/ll/`; `glm5_3/units.py` orders them per step, `glm5_3/registry.py` composes them, `glm5_3/runtime/composed.py` runs them in the executor's layer loop): the sparse MLA as flash decoding over the top-k rows (`ll.sparse_mla_fd`, up to 7 rows a wave), the attention front on the low-latency fused-A GEMM and a one-launch MLA KV half (`ll.attention`), the small-M BF16 GEMMs on PDL split-K and TGV (`ll.gemm.<role>`), the all-reduce + residual + RMSNorm hops on a plugin-owned Lamport buffer with the MoE finalize folded in (`ll.hops`), FlashInfer's TRT-LLM FP4 MoE with that deferred finalize (`ll.moe`), and in the NextN draft the row-parallel `eh_proj` summed in the target's hop (`ll.draft_eh`). Each step runs the first unit that covers its width, SGLang's step body elsewhere; a width no unit covers runs SGLang's forward, and the startup log names why ("stock at N rows: ..."). Each unit is a slot (`runtime.LL_SLOTS`); `ll.moe.alt_stream_priority` runs only when listed. A unit that covers a width but fails to build stops the server on every rank, naming its slots.

The TP4 split-K GEMMs (`ll.gemm`) run FlashInfer's split-K kernel (`mm_bf16(..., tactic=)`) when the installed FlashInfer has it with the synchronization fixes (`SplitKTactic.SYNC_FIXED`); otherwise they run the plugin's vendored copy (`ll/kernels/cute/splitk.py`). The startup log names the source. `SGLANG_NVIDIA_MEGAKERNEL_GLM_GEMM_SPLITK` switches it: `auto` (the default), `flashinfer` (FlashInfer's kernel even without the fixes; the server stops if FlashInfer has no such API) or `vendored`.

With the megakernel on, all shipped slots run; none needs to be listed. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting only; a list without a slot another needs (`dnorm` needs `qkv_a_small_m_shared` and `hops`, `qkv_a_small_m_shared` needs `attention`, `draft_rowselect` needs `draft`) stops the server at startup, naming it. Slot `draft` needs embedding replication, which the flag turns on (SGLang sets `SGLANG_ENABLE_EMBED_REPLICATION` when it is unset), greedy draft acceptance and an SGLang whose EAGLE draft graph runners run a draft spec's capture check; the server stops at startup, naming each one that is missing (an explicit `SGLANG_ENABLE_EMBED_REPLICATION=0` too).

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step. Exceptions, each made at a width's first eager forward (normally SGLang's eager warm-up run before it captures the width), after the KV cache is sized, and never under CUDA-graph capture: Qwen3.5's TEP8 routed body (`moe/body.py`), each width's runner, tactic, routing tables and body workspace, and with slot `moe_engine` the engine's buffers, call state and ring records (the engine's pieces refuse to be made under capture); the router-GEMM compiles of slots `route_big_gemm` and `moe_engine_route` (a width first seen under capture does not take these slots; `route_big_gemm`'s then runs SGLang's router GEMM); slot `moe_engine_route`'s per-CTA tables (refused under capture); and `gemm_tactics`' per-width algorithm check, which runs at a width's first call outside capture (its paragraph above).
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** DeepSeek-V4.1-Flash waits on another GPU have
  a deadline below SGLang's watchdog. A rank that misses it faults at address
  0x0, so no token from a failed step is returned. Kimi K3 validates SGLang's
  finite build deadline and compiles it into all six reachable TP8 remote-wait
  paths; expiry likewise stores through address 0x0 and traps. Qwen3.5's hop
  compiles the build deadline in, and CP-lite's waits (the CP slots' q gathers,
  including the q writer's, their flag barriers and merges) take the same
  deadline from each wait's start; the waits of slots `bt_tail`, `gemm_ar` and
  `fc2ar` have none, as FlashInfer's BT owner reduce at the same points has none.
- **No silent fallback.** With `--enable-nvidia-megakernel`, an unsupported setup stops the server at startup, before the weights load, listing every reason; a model the loaded weights do not fit stops every rank after the load. A Qwen3.5 slot whose packing the loaded weights do not fit runs stock on every rank instead, with the reason in the startup log; where the check reads rank-local weights (`route_lin`), the ranks agree first and each logs the decline as a warning.

## Quick Start

### Requirements

- An SGLang with the megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, or `sglang.srt.megakernels` before sgl-project/sglang#45389; `--enable-nvidia-megakernel`); Qwen3.5 also needs `sglang.srt.kernel_plugins.nvidia.eagle_binding`; GLM-5.3 also needs its DSA binding view (`ForwardView.dsa` in `sglang/srt/kernel_plugins/nvidia/binding.py`, the SGLang PR filed with this model), and refuses at startup without it; with that PR the flag also turns on the embedding replication slot `draft` needs (`FLAG_SETS_ENV` in `sglang/kernels/plugin_kernels/nvidia/spec.py`)
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.3), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- DeepSeek-V4.1-Flash: 4 x GB300, or 4 x 148-SM B300 (SM103) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`; at TP8 + EP2, 8 x GB300 on two trays of one NVLink domain
- Qwen3.5: 8 x GB300, two trays of one NVLink domain
- Kimi K3: 8 x GB300 (SM103, 152 SMs), or 8 x 148-SM B300 (SM103) or B200 (SM100) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (for Kimi K3 it only lowers the SM floor), BF16, context length up to 262144 within the
  native model limit, TP8, DP1 and DCP off,
  FP8 E4M3 KV cache with page size 64, `trtllm_mla` for prefill and decode,
  `flashinfer_mxfp4` with default precision, and full decode CUDA graphs
- GLM-5.3, GLM-5.2: 8 x B300, or 8 GPUs over two trays of one NVLink domain (GB300 NVL72, Rubin VR NVL72), or 4 x GB300 (TP4)
- Kimi K3 DSpark requires the target-verification interfaces in
  [SGLang #45406](https://github.com/sgl-project/sglang/pull/45406). The tested
  configuration uses `RadixArk/Kimi-K3-DSpark` with five draft proposals (q6);
  the draft model and acceptance remain on SGLang's stock path.
- Kimi K3 checks for CuTe DSL 4.8.0, a coherent FlashInfer Python/cubin/JIT-cache
  wheel trio and the private ABI used by its prepared kernels. The general
  dependency list above is not a promise of compatibility with arbitrary newer
  versions; see `sgl_nvidia_mega/kimi_k3/compat.py`.

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

DeepSeek-V4.1-Flash:

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --engram-config '{"placement": "cpu"}' \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

`SGLANG_SPEC_TP_SYNC` and `--engram-config` are required by the DeepSeek-V4.1 spec; without them the server stops at startup, naming each.

### DeepSeek-V4.1-Flash at TP8 + EP2

On two GB300 trays of one NVLink domain the spec admits one more layout: attention TP8 with the MoE at TP4 x EP2 (`--tp 8 --ep-size 2`). Each EP half holds 192 of the 384 routed experts and the shared expert is split over all 8 ranks, as SGLang loads them; every other (tp, ep) pair, TP4 over several nodes, a non-trivial expert placement and `SGLANG_SHARED_EXPERT_TP1` are refused at startup. The recommended launch, on each tray (`--node-rank 0` and `1`):

```bash
SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1 \
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 8 --ep-size 2 --nnodes 2 --node-rank 0 --dist-init-addr <tray-0 host>:<port> \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --engram-config '{"placement": "cpu"}' \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

The default slot list serves this layout; `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` turns on SGLang's own DSpark runtime optimizations for megakernel serving (fused accept tail, deferred draft KV store and others in SGLang's `environ.py`); the TP8 numbers in this README were measured with it. Each rank keeps its own Engram host table (`per_rank`), as at TP4.

`swapab_h8` and `wo_a_g1` change the arithmetic against stock SGLang at TP8, which runs the decode attention as FlashMLA with the 8 local heads padded to 64 and wo_a as inverse RoPE, a BF16 einsum and wo_b's MXFP8 quantize. The two slots run TP4's kernels instead: swap-AB on the 8 local heads and the fused wo_a on the 1 local group, each bit for bit the TP4 kernel on the same heads or group. The fused wo_a writes the same MXFP8 codes as stock TP8 at 1 and 6 rows; at 16 rows 0.012 % of codes move by at most one eighth of a block step. GSM8K with the two slots matched stock TP8. Without them TP8 still serves, on SGLang's FlashMLA and einsum, at about 4.0 ms per verify step instead of 3.2 ms (one request, DSpark acceptance length forced to 3.51).

For attention-TP8 serving at batch 1 the opt-in `attn_core_lean` saves about 130 us per verify step more. It needs SGLang's `v41_new` KV layout, a server-wide setting the plugin does not choose, so add both to the launch above:

```bash
SGLANG_DSV4_KV_LAYOUT=v41_new \
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(s + ('attn_core_lean',)))") \
```

Under `v41_new` swap-AB declines (it reads only the V4 layout), so verify steps above 8 rows run SGLang's attention and are much slower; do not set it for batched serving or at TP4.

SGLang fuses its all-reduce + mHC posts only at TP2 and TP4. At TP8 the target's posts run on the plugin's hops (hop A, hop B and the two-shot) and the draft's posts on a Lamport mailbox of their own (slot `draft_hop`, shipped; it builds nothing at TP4). At startup the draft logs each requested draft slot as built, declined (with the reason) or not needed.

Qwen3.5, on each of the two trays (`--node-rank 0` and `1`):

```bash
python -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --tp 8 --nnodes 2 --node-rank 0 --dist-init-addr <tray-0 host>:<port> \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN --speculative-num-steps 6 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 7 \
  --attention-backend trtllm_mha \
  --linear-attn-decode-backend triton \
  --enable-linear-replayssm-spec --mamba-ssm-dtype bfloat16 \
  --moe-runner-backend flashinfer_trtllm \
  --flashinfer-allreduce-fusion-backend cutedsl \
  --enable-nvidia-megakernel
```

GLM-5.3 or GLM-5.2 (NVFP4) on 8 x B300, with an NVFP4 GLM-5.3 or GLM-5.2 checkpoint (for example `nvidia/GLM-5.2-NVFP4`) whose config has the shapes in `glm5_3/__init__.py` (`HF_CONFIG`; the spec checks them at startup):

```bash
python -m sglang.launch_server \
  --model-path <NVFP4 GLM-5.3 or GLM-5.2 checkpoint> \
  --trust-remote-code \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 --page-size 64 \
  --attention-backend dsa --dsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm \
  --disable-shared-experts-fusion \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-nvidia-megakernel
```

Every GLM-5.3 slot runs with `--enable-nvidia-megakernel` alone; no slot list and no environment variable are needed. The GLM-5.3 NextN draft step (slot `draft`) needs embedding replication, which the flag turns on: SGLang sets `SGLANG_ENABLE_EMBED_REPLICATION` when it is unset and logs it, and keeps a value you set (`0` stops the server at startup, naming it). The draft step also needs greedy draft acceptance (no `--speculative-use-rejection-sampling`, SGLang's default); without it the server stops at startup, naming it. SGLang's defaults give the rest of the environment the spec needs; a setup it does not serve stops at startup, naming each reason. Serving with HiCache (`--enable-hierarchical-cache`) is supported: the covered steps read only the running requests' device KV pages. At TP4 so is the decode role of a PD-disaggregated deployment (`--disaggregation-mode decode`): the KV arrives through the transfer backend before a request's first verify step, outside the covered step; the prefill role runs without the flag.

On two trays (GB300 NVL72, Rubin VR NVL72), launch the same command on each tray with `--nnodes 2 --node-rank <0|1> --dist-init-addr <tray-0 host>:<port>`. The covered steps need SGLang's FlashInfer all-reduce + RMSNorm fusion at verify sizes, which needs NVLink multicast; on nodes without it, set `SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION=1`: the covered steps then run anyway, the stock residual steps on SGLang's unfused all-reduce and slot `hops` on its own symmetric-memory all-reduce.

The GLM-5.3 megakernel serves MTP with 3 draft steps (`--speculative-num-steps 3 --speculative-num-draft-tokens 4`, as above) or 5 (`--speculative-num-steps 5 --speculative-num-draft-tokens 6`) at TP8, and 2 to 5 draft steps at TP4 (`--tp-size 4`); any other draft length stops the server at startup, naming the served values.

At TP8 the GLM-5.3 megakernel covers verify steps of up to 16 rows: 1 to 4 requests with 4 draft tokens, 1 or 2 with 6 (at TP4 up to 48 rows); larger steps run SGLang's path while its derived weight copies (about 20 GiB per GPU, logged at startup) still take KV-cache memory, so it helps little on high-concurrency servers, and the server warns when `--max-running-requests` is set above 8.

### Kimi K3.1 on GB300

The Kimi K3.1 megakernel (`sgl_nvidia_mega/kimi_k31`) runs the DFlash target verify
step and, behind `SGLANG_K31_DRAFT_MK`, the DFlash2 draft's decoder layers. It needs
an SGLang with the Kimi K3.1 host bindings (sgl-project/sglang#48414); on an older
SGLang its spec refuses every configuration with that reason. It builds on its own
copy of the Kimi K3 base (`kimi_k31/k3` and its kernels in `kimi_k31/sm103/k3`), so
`sgl_nvidia_mega/kimi_k3` and the Kimi K3 path are unchanged.

`kimi_k31/k3/default_scope.py` turns a set of tuned options on by default only in
the configuration they were measured in: TP16, no DCP, 8 draft tokens and
`--max-running-requests 1`. Elsewhere an unset option keeps its other default, and
a set environment variable always wins.

### Kimi K3 optional startup providers

The hybrid runtime and DSpark interfaces are upstream in SGLang through
[#45406](https://github.com/sgl-project/sglang/pull/45406), merge
`aa3e34aaebf880d72482cea74dbbfef54da728e2`. API v2 is required. The optional
interfaces below do not expand the registered hardware, model or width guards.

Native SGLang embedding and no cross-layer weight prefetch remain the defaults.
For experiments, set either of these plugin-owned switches before launching
the server with `--enable-nvidia-megakernel`:

| Environment variable | Unset or `0` | `1` |
| --- | --- | --- |
| `SGLANG_KIMI_K3_MEGAKERNEL_PREFETCH` | No optional prefetch preparation/resources | Prepare cross-layer weight prefetch using the existing selective TP8 token-row policy |
| `SGLANG_KIMI_K3_MEGAKERNEL_EMBEDDING` | Native embedding; no optional gather preparation/resources | Prepare the specialized gather for supported native embedding layouts; retain native fallback otherwise |

Use the same settings on every tensor-parallel rank. Only `0` and `1` are
accepted; other values fail during build, before optional resource preparation.
The plugin reads these switches at build time, not in the decode/verification
loop. Restart the server to change them; they are not live graph toggles.
Supplied `inputs_embeds`, vocabulary sharding and native communication semantics
remain unchanged. Neither option enables DCP, alternate MLA or wider batch
admission.

Each option has bounded ordinary-decode and q6 serving correctness evidence.
The prefetch serving screen and embedding component measurements did not
establish a useful benefit in their tested scopes; neither justifies default
activation. Both remain OFF. Their separate proofs do not qualify both options
together or establish arbitrary workload, long-context ON or performance coverage.

### Kimi K3 optional capture-width build API

`sgl_nvidia_mega.kimi_k3.spec.build_with_capture_widths` is an explicit
adapter-facing alternative to `SPEC.build`. The registered build remains
unchanged; no CLI or environment setting automatically enables this policy.

Separately, [SGLang #49376](https://github.com/sgl-project/sglang/pull/49376)
automatically bounds `BuildContext.widths` by the resolved request limit for
eligible fixed-q6/TP8/DCP-off configurations before calling `SPEC.build`.
That policy retains reachable specialized eager widths, keeps full preparation
when a safe bound is unavailable, and does not invoke this optional API.

Call the alternative on every TP rank, passing the usual model, HF config and
`BuildContext`, plus keyword `capture_token_rows`. `None` preserves the existing
width policy. A tuple selects baseline token rows `1–8,16,32,64` plus the supplied
capture rows. Sizes must be positive integers; duplicates are deduplicated,
and sizes above capacity are excluded. An empty tuple selects the baseline.
Use actual configured capture rows, not a dense capacity range inferred from
`BuildContext.widths`. For q6, request batches 1/2/4 correspond to token rows
6/12/24. Verification rounds up to whole requests, excludes rounded rows above
64 and excludes ordinary-only baseline rows incompatible with its query width.
This does not add ordinary decode to a verification-only provider.

The API checks cross-rank agreement before preparation and snapshots the selected
widths into a new build context. All ranks must call the same build API; this is
not a live toggle. It does not implicitly enable another provider. The prefetch
and embedding environment settings above are still read independently; alternate
MLA and DCP remain unselected.
Normal SGLang startup continues to use `SPEC.build`; an adapter must explicitly
choose the alternative when activating it later.

CPU selection/forwarding tests do not establish GPU correctness or performance.
Wider full-model numerical checks retain unresolved failures; this API is
experimental and default-OFF, not qualification or a fix for those failures.

### Kimi K3 dormant attention and DCP interfaces

These lower-level interfaces retain tested implementations for explicit future
integration. They are not server switches and do not change `SPEC.build`.

| Capability | Plugin interface | Registered default / integration boundary |
| --- | --- | --- |
| Alternate MLA | `providers.build_resources(..., mla_core="tokenspeed")` | `mla_core="trtllm"`; an adapter must select and prepare the alternative before graph capture |
| DCP8 resources | `providers.build_resources(..., mla_core="tokenspeed", dcp=dcp_resources.DcpBuildOptions())` | `dcp=None`; requires the matching native TP/DCP groups before preparation |
| DCP ordinary decode | `dcp_execution.PreparedDcpHybrid` | Requires the separate SGLang DCP binding extension and its owned pools/buckets |
| DCP fixed-q6 verification | `dcp_verify_execution.PreparedDcpVerify` | Requires the separate SGLang DCP verification binding extension, including native acceptance and state commit |

Stock SGLang at the merge above does not contain those DCP extensions. Importing
the plugin does not require them; attempting the unsupported DCP handoff rejects
explicitly rather than substituting the ordinary pool layout. Callers own native
metadata preparation, graph-safe updates and resource lifetimes. The plugin's
readiness/ownership checks must not be bypassed. These are startup integration
choices, not live backend or graph toggles.

TokenSpeed selection requires `tokenspeed-mla==2.0.2`; DCP requires this selector
explicitly, not the default TRTLLM core. Supplying `dcp` alone is rejected.

The bounded attention/DCP results described above do not qualify arbitrary
combinations with prefetch, embedding, additional widths or long-context DCP.
Their dormant inclusion does not enable them or imply a performance benefit.

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # registers the specs with SGLang's megakernels framework
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   ├── deepseek_v41/
│   │   ├── spec.py          # the spec: the setups it serves and its build
│   │   ├── plan.py          # which kernel fills each slot, per width
│   │   ├── runtime/         # the build, the executor and its per-step state, the embedding
│   │   ├── boundary/        # the mHC between sublayers
│   │   ├── attention/       # the attention step
│   │   ├── moe/             # the MoE step
│   │   ├── engram/          # the Engram layers
│   │   ├── logits/          # the head and the logits
│   │   └── sm103/           # kernels and the build-time runner; no SGLang imports
│   ├── qwen3_5/              # spec, plan, runtime, boundary, GDN, attention, MoE and SM103 kernels
│   ├── glm5_3/            # GLM-5.3 and GLM-5.2: the same layout (spec, plan, runtime, boundary, attention, moe, logits, sm103)
│   ├── kimi_k3/              # the Kimi K3 spec, adapter and executor
│   │   └── sm103/            # GB300 CuTe DSL kernels and their launchers
│   └── kimi_k31/             # the Kimi K3.1 spec, verify/decode bindings and the DSA verify step
│       ├── draft/            # the DFlash2 draft spec
│       ├── k3/               # the Kimi K3 base the K3.1 megakernel builds on
│       └── sm103/            # GB300 kernels: the draft layers, DSA JIT kernels and the K3 base kernels
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # plugin contracts and setup checks; selected CUDA metadata/graph cases
    ├── kernels/           # legacy standalone kernel tests; outside required contribution CI
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

The contribution suite consists of unit tests and E2E tests. Each supported
model/hardware configuration needs an accuracy test and a performance test
against the same native SGLang baseline. See the
[contributor guide](CONTRIBUTING.md#required-tests) for the comparison protocol
and the distinction between required coverage and the initial CI rollout.

```bash
python3 -m pytest test/unit -v -rfE
```

GPU E2E test definitions live in this repository and run on SGLang's GPU CI
infrastructure. A maintainer requests them with `run-ci-sglang` on the Mega PR.
The initial workflow covers DeepSeek-V4.1-Flash on four GB300 GPUs; other
model/hardware combinations require their own workflow coverage.

Additional manual test coverage:

The DeepSeek-V4.1-Flash TP8 + EP2 kernel tests (`test_deepseek_v41_moe_ep.py`, `test_deepseek_v41_hop_b.py`, `test_deepseek_v41_hops.py`, `test_deepseek_v41_draft_hop.py`, and `test_deepseek_v41_draft.py -k ranks`) run on both trays under a two-node torchrun (`--nnodes 2 --nproc-per-node 4 -m pytest <file> -s`).

Qwen3.5 runs TP8, which spans two GB300 trays: its e2e tests read a server launched on them through `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` and the files its ranks log to (`SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS`, comma-separated).

GLM-5.3 GPU and end-to-end suites are not yet part of the initial CI workflow.
The GLM-5.3 CPU config and SGLang contract checks are included in `test/unit`.
Kimi K3 full-serving qualification is separate from these unit tests; the
standalone DeepSeek-V4.1-Flash kernel tests remain outside required contribution CI.

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md): merge prerequisite SGLang bindings
through normal SGLang CI first, then validate the Mega PR through the label-based
integration flow. The guide defines unit/E2E coverage, matched accuracy and
performance comparisons, and the evidence to include in a PR.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
