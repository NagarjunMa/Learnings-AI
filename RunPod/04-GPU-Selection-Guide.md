# GPU Selection Guide

## Full GPU Catalog

| GPU | VRAM | Architecture | Approx $/hr (Secure) | Notes |
|---|---|---|---|---|
| RTX 3090 | 24 GB | Ampere | $0.44 | Good for 7B INT4/INT8; no ECC |
| RTX 4090 | 24 GB | Ada Lovelace | $0.74 | Best perf/$ for inference at 24GB |
| A40 | 48 GB | Ampere | $0.79 | Data center, ECC, 48GB VRAM |
| A6000 | 48 GB | Ampere | $0.76 | Similar to A40, workstation GPU |
| L40S | 48 GB | Ada Lovelace | $1.14 | Ada + data center + ECC; inference sweet spot |
| A100 PCIe 40GB | 40 GB | Ampere | $1.64 | Data center, ECC, NVLink optional |
| A100 PCIe 80GB | 80 GB | Ampere | $1.89 | 80GB VRAM, PCIe bandwidth |
| A100 SXM 80GB | 80 GB | Ampere | $2.09 | NVLink, 2TB/s bandwidth — best multi-GPU |
| H100 PCIe | 80 GB | Hopper | $2.99 | Transformer Engine, FP8 |
| H100 SXM | 80 GB | Hopper | $3.49 | NVLink + NVSwitch, 3.35TB/s bw |
| H100 NVL | 94 GB | Hopper | $3.99 | Larger VRAM than standard H100 |
| H200 | 141 GB | Hopper | $4.99 | HBM3e, 4.8 TB/s bandwidth |
| B200 | 192 GB | Blackwell | ~$8.00+ | Latest gen, 8 TB/s bandwidth |

## VRAM Requirements by Model Size and Precision

Rule of thumb: `params (billions) × bytes_per_param + ~20% KV cache overhead`

| Model Size | FP16 (2B/param) | INT8 (1B/param) | INT4 (~0.5B/param) |
|---|---|---|---|
| 7B | ~15 GB | ~8 GB | ~4-5 GB |
| 13B | ~28 GB | ~14 GB | ~7-8 GB |
| 30B | ~62 GB | ~31 GB | ~16 GB |
| 70B | ~140 GB | ~70 GB | ~37 GB |

Add 15-25% for KV cache at typical sequence lengths (2048 tokens).

## GPU-to-Model Matching

| Model | Precision | Fits on |
|---|---|---|
| 7B | FP16 | RTX 4090 (24GB), A40, L40S |
| 7B | INT4 | RTX 3090/4090 (24GB) easily |
| 13B | FP16 | A40 (48GB), L40S (48GB) |
| 13B | INT4 | RTX 4090 (24GB) with room |
| 30B | FP16 | A100 80GB, H100 |
| 30B | INT4 | A40, L40S (48GB) |
| 70B | FP16 | H100 NVL (94GB) or 2×A100 80GB |
| 70B | INT4 | A100 80GB, H100 PCIe |
| 70B | FP16 | H200 (141GB) single GPU |

## Memory Bandwidth Comparison

Memory bandwidth dominates decode throughput (token generation speed).

| GPU | Mem BW | Impact |
|---|---|---|
| RTX 4090 | ~1.0 TB/s | Good for 7B, moderate for 13B |
| A100 PCIe 80GB | ~1.9 TB/s | Solid production inference |
| A100 SXM 80GB | ~2.0 TB/s | ~5% faster than PCIe on decode |
| L40S | ~0.86 TB/s | Lower BW than A100 despite Ada arch |
| H100 SXM | ~3.35 TB/s | 1.7× A100 on decode throughput |
| H200 | ~4.8 TB/s | Best single-GPU decode speed |
| B200 | ~8.0 TB/s | Next gen, best available |

Key: for decode-heavy workloads (long generation), BW matters more than FLOPs.

## A100 SXM vs PCIe

| | A100 PCIe 80GB | A100 SXM 80GB |
|---|---|---|
| Memory bandwidth | ~1.9 TB/s | ~2.0 TB/s |
| NVLink | Optional (via NVSwitch) | Yes, native |
| NVLink bandwidth | ~600 GB/s | ~600 GB/s |
| TDP | 300W | 400W |
| Cost/hr | ~$1.89 | ~$2.09 |
| Multi-GPU scaling | PCIe limited (~64 GB/s) | NVLink scales well |
| Use for TP | Suboptimal | Preferred |

SXM = Socket Module — physically different form factor, sits in NVSwitch backplane.

## Training vs Inference vs Fine-tuning GPU Choice

**Training (full fine-tune)**
- Need max VRAM for model + optimizer states (Adam = 3× model size)
- 70B model full fine-tune: ~600GB+ → requires H100 cluster
- Preferred: A100 SXM, H100 SXM (NVLink for tensor parallelism)

**Inference**
- Need VRAM for model weights + KV cache
- BW-limited during decode → H100/H200 over A100 for throughput
- Cost-optimized: RTX 4090 (24GB INT4) or L40S (48GB)

**LoRA / QLoRA fine-tuning**
- Only trains adapters (~1-10M params), base model in INT4/INT8
- 7B QLoRA: fits on single RTX 4090 (24GB)
- 70B QLoRA: fits on A100 80GB INT4 with room

## RTX 4090 vs A100 Decision Matrix

| Factor | RTX 4090 | A100 80GB |
|---|---|---|
| VRAM | 24 GB | 80 GB |
| Cost/hr | ~$0.74 | ~$1.89 |
| ECC memory | No | Yes |
| Data center grade | No (consumer) | Yes |
| LoRA 7B INT4 | Yes, plenty | Overkill |
| 13B FP16 inference | No (too small) | Yes |
| Long-context KV cache | Fills fast | Much more headroom |
| Use for | Dev, 7B inference, fine-tuning | Production 13B+, training |

## L40S vs A100 for Inference

| | L40S | A100 80GB PCIe |
|---|---|---|
| VRAM | 48 GB | 80 GB |
| Architecture | Ada Lovelace | Ampere |
| Memory BW | ~0.86 TB/s | ~1.9 TB/s |
| FP16 FLOPs | ~366 TFLOPS | ~312 TFLOPS |
| Cost/hr | ~$1.14 | ~$1.89 |
| INT8/FP8 | Ada INT8 fast | Good INT8 |
| Best for | Compute-bound models (prefill) | BW-bound decode throughput |

L40S is better if compute (prefill latency) is the bottleneck.
A100 is better if decode throughput (tokens/sec) is the priority.
