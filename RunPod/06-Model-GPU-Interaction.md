# Model-GPU Interaction

## VRAM Formula

```
VRAM required = (params × bytes_per_param) + 20% KV cache overhead
```

Examples:
- 7B FP16: `7 × 10⁹ × 2 bytes = 14 GB + ~3 GB overhead = ~17 GB`
- 70B INT4: `70 × 10⁹ × 0.5 bytes = 35 GB + ~7 GB overhead = ~42 GB`
- 13B INT8: `13 × 10⁹ × 1 byte = 13 GB + ~3 GB overhead = ~16 GB`

KV cache overhead scales with:
- Batch size × sequence length × num_heads × head_dim × 2 (K + V) × 2 (bytes for FP16)
- Long sequences or large batches can exceed the 20% estimate

## Two Phases of Inference

### Prefill (Prompt Processing)
- Reads input tokens, builds KV cache
- **Compute-bound**: throughput scales with FLOPs
- High parallelism — all input tokens processed simultaneously
- Metric: prefill latency (ms)

### Decode (Token Generation)
- Generates one token at a time, reading entire KV cache per step
- **Memory-bandwidth-bound**: throughput scales with memory bandwidth
- Sequential — can't parallelize across tokens
- Metric: tokens/second

Implication: for long generation tasks, GPU memory bandwidth matters more than raw compute.

## Memory Bandwidth Table and Impact on Decode

| GPU | Mem BW | Relative decode speed |
|---|---|---|
| RTX 4090 | ~1.0 TB/s | 1× baseline |
| A100 PCIe 80GB | ~1.9 TB/s | ~1.9× |
| A100 SXM 80GB | ~2.0 TB/s | ~2.0× |
| H100 SXM | ~3.35 TB/s | ~3.35× |
| H200 | ~4.8 TB/s | ~4.8× |
| B200 | ~8.0 TB/s | ~8.0× |

For a 7B FP16 model: RTX 4090 ≈ 80 tok/s, H100 SXM ≈ 250+ tok/s decode.

## Quantization Formats Comparison

| Format | VRAM vs FP16 | Speed | Quality | Notes |
|---|---|---|---|---|
| FP16 | 1× (baseline) | 1× | Baseline | Standard, full quality |
| INT8 | ~0.5× | ~1.0-1.2× | ~99% | bitsandbytes, minimal quality loss |
| GPTQ | ~0.25-0.5× | 0.8-1.0× | ~97-99% | Post-training quantization, static |
| AWQ | ~0.25× | ~1.5-2× | ~98-99% | Activation-aware, better than GPTQ |
| AWQ + Marlin | ~0.25× | ~3-4× | ~98-99% | Marlin kernel: 741 tok/s reported |
| GGUF Q4_K_M | ~0.25× | CPU-optimized | ~97% | Best for CPU/hybrid (llama.cpp) |
| GGUF Q8_0 | ~0.5× | CPU-fast | ~99% | Near-lossless on CPU |

AWQ with Marlin kernel = fastest GPU inference per VRAM dollar.
GGUF Q4_K_M = best for Mac Mini / CPU hybrid (not RunPod GPU).

## Tensor Parallelism with vLLM

Split model across multiple GPUs:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4  # splits across 4 GPUs
)
```

Requirements:
- GPUs must be in the same machine (not distributed across nodes for vLLM TP)
- For efficient TP: GPUs need NVLink (SXM variants)
- PCIe-connected GPUs can TP but bandwidth bottleneck limits speedup

## NVLink vs PCIe for Multi-GPU

| | NVLink (SXM) | PCIe |
|---|---|---|
| Bandwidth | ~600 GB/s bidirectional | ~64 GB/s (PCIe 4.0 x16) |
| Latency | Low | Higher |
| Shared memory address space | Yes (via NVLink) | No |
| Tensor parallelism scaling | Near-linear to 8 GPUs | Significant bottleneck at 4+ GPUs |
| Cost | A100 SXM, H100 SXM | A100 PCIe, RTX cards |

NVLink allows GPU peer-to-peer memory access without CPU involvement — critical for activations/gradients passing in tensor parallel training.

## How NVLink Shared Memory Works

With NVLink, CUDA can allocate unified virtual addresses across GPUs:
- GPU 0 can read from GPU 1's memory directly at ~NVLink bandwidth
- No CPU bounce, no PCIe hop
- Enables efficient all-reduce (gradient sync) and KV cache sharing in TP
- NVSwitch (in DGX systems) provides full mesh — any GPU to any GPU at full bandwidth

Without NVLink (PCIe only): all inter-GPU comms go through CPU/host memory, limiting TP efficiency.

## Practical Model-to-GPU Assignment (with Quantization)

| Model | Strategy | GPU | Est. VRAM Used |
|---|---|---|---|
| 7B | FP16 | RTX 4090 | ~16 GB |
| 7B | INT4 AWQ | RTX 3090 | ~5 GB |
| 13B | FP16 | A40 / L40S | ~28 GB |
| 13B | INT4 AWQ | RTX 4090 | ~8 GB |
| 30B | FP16 | A100 80GB | ~65 GB |
| 30B | INT4 AWQ | A40 48GB | ~20 GB |
| 70B | FP16 | H100 NVL 94GB / H200 | ~140 GB |
| 70B | INT4 AWQ | A100 80GB | ~40 GB |
| 70B | INT4 AWQ TP=2 | 2× A40 | ~20 GB each |
