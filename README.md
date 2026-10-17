# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** is the monokernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for NVIDIA GPUs. It holds the kernels of a decode or speculative-verify step that only make sense inside a fused step: kernels that hand their outputs to each other through mailboxes and programmatic dependent launch instead of through SGLang's per-op boundaries.

The library implements SGLang's monokernel interface (`sglang.srt.monokernels`) and loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point), which registers each model's spec under the monokernel backend `nvidia`. With `--monokernel-backend nvidia`, every step that a spec serves runs on its kernels, and every other step runs SGLang's standard path. Without the flag, nothing in this package is imported beyond its entry point.

> **Experimental:** `--monokernel-backend` is an experimental SGLang feature. The kernels, the models and setups they cover, and the plugin interface may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103) | 4 | DSpark verify steps of 1 to 48 rows (batch 1 to 8 at 6 rows a request), and the DSpark draft stages |

Steps past 48 rows, prefill and extend steps run SGLang's standard path.

### What a routed step runs

A routed step is SGLang's own step, rebuilt from SGLang's modules, with three slots filled by this package:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm | 1–8 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| Router top-k | 1–8, 48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Routed experts | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |

Every other kernel is SGLang's. The kernels stay separate launches chained with programmatic dependent launch: on GB300 a kernel boundary with PDL costs less than an exchange between all blocks of one persistent grid.

### Design

- **Built at startup.** The mailbox, the kernels for every routed width and their tactics are allocated and compiled when the server starts, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe routing.** Each step is routed by its CUDA graph key alone, so every rank makes the same decision and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** Every wait on another GPU has a deadline below SGLang's watchdog. A rank that misses it faults at address 0x0, so no token from a failed step is returned.
- **In-place weights.** The kernels read the weights SGLang loaded; nothing is copied.

## Quick Start

### Requirements

- SGLang with the monokernel interface (`sglang.srt.monokernels`); tested with SGLang `b62a48ee15b`
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.2), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- 4 x GB300

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --monokernel-backend nvidia
```

If the model, GPU or configuration is not supported, the server refuses to start and lists every reason.

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   └── deepseek_v41/
│       ├── spec.py        # target model: spec and refusals
│       ├── draft.py       # DSpark draft model: spec
│       ├── plan.py        # which kernel fills each slot, per width
│       ├── forward.py     # the routed target and draft steps (SGLang glue)
│       └── sm103/         # kernels and the build-time runner; no SGLang imports
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # CPU: the specs, the slot plan and the SGLang interface contract
    ├── kernels/           # GPU: each kernel against SGLang's or FlashInfer's op, fail-stop
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

```bash
pytest test/unit               # CPU only
pytest test/kernels            # 4 x GB300
pytest test/e2e                # 4 x GB300, with DEEPSEEK_V41_FLASH_MODEL_PATH set
```

## Contributing

Open a pull request against `main`. The PR title follows `[Type] scope: title` (see `.github/pull_request_template.md`), and `pre-commit run --all-files` must pass.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
