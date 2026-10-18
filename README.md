# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and needs no change to SGLang. With `SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1` it registers two hooks: after SGLang loads a covered model, it checks the setup and builds the megakernel (before the KV cache is sized); around the model's `forward`, it runs each covered step itself and leaves every other step to SGLang. Without the variable it registers nothing.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103) | 4 | DSpark target verify steps of 1 to 48 rows (batch 1 to 8 at 6 rows a request) |

Larger verify steps, prefill, extend and the DSpark draft run SGLang's standard path.

### What a covered step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`) | 1–8, 48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |

`SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) selects the slots; empty runs every step function's stock code.

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** Every wait on another GPU has a deadline below SGLang's watchdog. A rank that misses it faults at address 0x0, so no token from a failed step is returned.
- **No silent fallback.** With the switch on, an unsupported setup stops the server at startup on every rank, listing every reason.

## Quick Start

### Requirements

- SGLang main; tested with SGLang `5cbe2371a53`
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
SGLANG_ENABLE_NVIDIA_MEGAKERNEL=1 \
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=moe,gate \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5
```

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # the switch and the hooks it registers
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   └── deepseek_v41/
│       ├── requirements.py  # the setups it serves; the startup refusals
│       ├── plan.py          # which kernel fills each slot, per width
│       ├── runtime/         # the hooks, the executor and its per-step state, the embedding
│       ├── boundary/        # the mHC between sublayers
│       ├── attention/       # the attention step
│       ├── moe/             # the MoE step
│       ├── engram/          # the Engram layers
│       ├── logits/          # the head and the logits
│       └── sm103/           # kernels and the build-time runner; no SGLang imports
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # CPU: the plugin, the setup checks, the step loop and the SGLang contract
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
