# RunPod Serverless Codebase Reference

Real codebase walkthrough: FLUX image-generation handler. Traces how model identity, framework primitives, determinism utilities, and dimension guards compose into a working serverless endpoint.

## `FluxPipeline` and `pipe`

`FluxPipeline` is a `diffusers` class that bundles every component needed for FLUX image generation into a single object:

| Component | Role |
|---|---|
| Text encoder | Converts prompt string → embedding tensor |
| Transformer | Denoising backbone (the bulk of parameters) |
| VAE | Decodes latent tensor → pixel image |
| Scheduler | Controls the denoising step sequence |

```python
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
```

`pipe` holds this assembled object in memory. Calling `pipe(prompt=..., height=..., width=..., generator=...)` runs the full end-to-end pipeline: encode text → iterative denoising → decode latents → return `PIL.Image`.

## `torch` role in the handler

PyTorch shows up in four specific ways in a serverless handler:

```python
# 1. Check hardware availability before loading
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# 2. dtype: bfloat16 halves VRAM vs FP32, more numerically stable than float16
pipe = FluxPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)

# 3. Move model weights to GPU
pipe.to("cuda")

# 4. Disable gradient tracking during inference (saves memory, speeds up forward pass)
with torch.no_grad():
    output = pipe(...)
```

`torch.bfloat16` is preferred over `float16` for diffusion models because it has the same exponent range as FP32 — it can represent very small and very large values without underflow/overflow during denoising math.

## `torch.Generator` and deterministic generation

```python
generator = torch.Generator("cuda").manual_seed(seed)
output = pipe(prompt=prompt, generator=generator, ...)
```

How it works:
- The diffusion pipeline starts from a random noise tensor sampled from a Gaussian distribution
- `torch.Generator` is a stateful PRNG; `manual_seed(seed)` sets its initial state
- With the same seed, the same initial noise tensor is produced → identical output image
- Without a fixed seed, each run draws different noise → different image

**When to fix vs randomize:**
- Expose `seed` to API callers for reproducibility / debugging
- Randomize (omit generator) for variety in production when callers don't care

## `_clamp_to_multiple`

FLUX and most diffusion models require image dimensions divisible by 8 (sometimes 64) due to tensor stride alignment in the VAE encoder/decoder downsampling layers. This helper enforces that constraint:

```python
def _clamp_to_multiple(value: int, multiple: int, max_val: int) -> int:
    # Step 1: cap at max to avoid OOM
    value = min(value, max_val)
    # Step 2: ensure at least one multiple (avoids zero-dimension tensor)
    value = max(multiple, value)
    # Step 3: floor to nearest multiple (e.g. 1023 → 1016 for multiple=8)
    return (value // multiple) * multiple
```

Annotated example for `_clamp_to_multiple(1023, 8, 2048)`:
1. `min(1023, 2048)` → `1023` (no cap needed)
2. `max(8, 1023)` → `1023` (already above floor)
3. `(1023 // 8) * 8` → `127 * 8` → `1016`

**Why each constraint:**
- `max_val=2048`: caps resolution to prevent OOM on typical serverless GPUs
- `max(multiple, ...)`: prevents a 0×0 tensor if caller passes `height=0`
- Floor division: CUDA convolution kernels crash or produce garbage output on misaligned tensor shapes

## Dynamic hardware scaling

Two `pipe` configuration methods for adapting to available VRAM:

### `enable_model_cpu_offload()`
```python
pipe.enable_model_cpu_offload()
```
- Shuttles individual model components (text encoder, transformer, VAE) between CPU RAM and GPU on demand
- Only the component currently needed occupies VRAM; others stay in RAM
- **Use when**: single GPU with limited VRAM; acceptable throughput tradeoff; safety net to avoid OOM

### `device_map="balanced"` (via accelerate)
```python
from accelerate import init_empty_weights
pipe = FluxPipeline.from_pretrained(..., device_map="balanced")
```
- `accelerate` library analyses model layer sizes and distributes them across all visible GPUs
- Each GPU holds a slice of layers; forward pass passes activations between GPUs
- **Use when**: multiple GPUs available and you want to maximise throughput with full parallel execution

| Method | VRAM needed | Multi-GPU | Throughput | Best for |
|---|---|---|---|---|
| `enable_model_cpu_offload()` | Low | No | Lower | VRAM-constrained single GPU |
| `device_map="balanced"` | Full model spread | Yes | Higher | Multi-GPU deployment |

## Serverless memory persistence

How the model-loading strategy maps to RunPod Serverless execution:

### Cold start path (first request on a new worker)
```
Network Volume (weights on disk)
  → System RAM  (CPU reads + deserializes .safetensors)
  → PCIe bus    (DMA transfer, pinned memory)
  → VRAM        (GPU now owns weights, ready for inference)
```
Cold start is the expensive path. See `05-Serverless-Deep-Dive.md` for FlashBoot and `min_workers` strategies to reduce frequency.

### Warm request path (subsequent requests, same worker)
- Model weights remain in VRAM — no reload
- Handler receives JSON job, runs `pipe(...)`, returns image bytes
- Latency is dominated by diffusion steps, not loading

### Parallel worker scaling
- Each new worker RunPod spins up = a new independent container with its own GPU
- That worker runs its own cold start: Network Volume → RAM → VRAM
- **Workers do not share VRAM**: VRAM is physically on-chip; there is no cross-GPU shared pool at the hardware level
- 1 worker = 1 container = 1 dedicated GPU (contiguous VRAM blocks required by CUDA cannot be split across physical cards)

**Consequence**: scaling from 1 → N concurrent requests means N independent cold starts happening in parallel, not N requests sharing one already-loaded model.
