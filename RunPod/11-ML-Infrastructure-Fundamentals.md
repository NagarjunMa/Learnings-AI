# ML Infrastructure Fundamentals

Bottom-up overview of the full ML infrastructure stack: from transistors and SIMD pipelines up through CUDA kernels, frameworks, model files, the loading pipeline, inference loop, and production deployment architecture.

## Why GPU over CPU

| Processor | Design goal | Core count | Core type |
|---|---|---|---|
| CPU | Complex sequential logic | 8–64 | Powerful, out-of-order, branch prediction |
| GPU | Parallel arithmetic throughput | Thousands | Simple, in-order, SIMD |

AI workloads are dominated by matrix multiplication (attention, feed-forward layers): the same multiply-accumulate operation repeated billions of times on large tensors. CPUs waste most of their transistor budget on control logic that never fires during tensor math. GPUs are the sweet spot because every transistor contributes to arithmetic throughput.

## GPU hardware components

| Component | What it is | Why it matters |
|---|---|---|
| CUDA Cores | General-purpose FP32/INT32 arithmetic units | Handles all non-matrix ops |
| Tensor Cores | 4×4 matrix multiply in one clock cycle | Accelerates Transformer attention; 10–20× faster than CUDA cores for matmul |
| VRAM | On-chip HBM (High Bandwidth Memory) | Sets model capacity ceiling — weights must fit here to run |
| Memory Bandwidth | GB/s throughput from VRAM to cores | Bottleneck during autoregressive decode (not compute, but data movement) |

Cross-ref: VRAM sizing formulas and bandwidth numbers → `06-Model-GPU-Interaction.md`.

## SIMD architecture

**Single Instruction, Multiple Data**: every CUDA core in a Streaming Multiprocessor (SM) executes the *same* instruction simultaneously on *different* data elements.

```
SM (Streaming Multiprocessor)
├── Warp Scheduler
├── 32 CUDA Cores  ← Warp 0: all execute "multiply A[i] × B[i]" in lockstep
├── 32 CUDA Cores  ← Warp 1: ...
└── Shared Memory / L1 Cache
```

- **Warp**: 32 threads that execute in lockstep; the atomic unit of GPU scheduling
- **Block**: group of warps sharing L1/shared memory
- **Grid**: all blocks launched for one kernel call

**Why ideal for tensor ops**: a matrix row × column dot product is the same multiply-add instruction on every element pair — perfect SIMD alignment.
**Why poor for branchy logic**: if half the threads take a different branch (`if condition`), the other half stall — called warp divergence.

## CUDA kernels

A CUDA kernel is a small C++ function tagged `__global__` that the GPU executes in parallel across thousands of threads simultaneously.

```cpp
__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread computes one output element
    float sum = 0;
    for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
    C[row*N+col] = sum;
}
```

PyTorch and TensorFlow do not expose kernels directly — they compile each layer operation (matmul, softmax, LayerNorm) into pre-written CUDA kernels and dispatch them to the GPU. The user writes Python; the framework emits CUDA.

**Kernel launch overhead**: each dispatch has ~5–10 µs CPU overhead. For small tensors, this overhead dominates. For large tensors (e.g. 4096×4096), compute time dominates — which is why batching matters.

## PyTorch vs TensorFlow

| Dimension | PyTorch | TensorFlow |
|---|---|---|
| Graph type | Dynamic (define-by-run) — graph built as Python runs | Static — graph compiled before execution |
| C++ backend | ATen tensor library | XLA compiler |
| Custom ops | Easy — just write Python/C++ | Harder — requires graph-compatible ops |
| Preferred for | Research, custom architectures, fine-tuning | Production serving, mobile export, TPU |
| End result | Both compile down to CUDA kernels on NVIDIA GPUs | |

Both ecosystems ultimately call the same CUDA libraries (cuBLAS, cuDNN) for the heavy math. The framework is the translator between Python semantics and GPU instructions.

## Model anatomy

Three file types that must all be present to run a model:

```
model-name/
├── model.safetensors          # (or multiple shard files: model-00001-of-00003.safetensors)
├── config.json                # Architecture specification
├── tokenizer.json             # Vocabulary + merge rules
└── tokenizer_config.json      # Tokenizer settings
```

| File | Contains | Why needed |
|---|---|---|
| `.safetensors` / `.bin` shards | Learned weight tensors — the actual parameters from training | Without weights the architecture is an empty shell |
| `config.json` | `num_layers`, `hidden_dim`, `num_attention_heads`, `vocab_size`, etc. | Framework needs this to instantiate the correct class before loading weights |
| `tokenizer.json` + vocab | Token ↔ ID mappings, BPE merge rules | Converts raw text to input IDs and output IDs back to text |

All three are downloaded together from HuggingFace Hub with `snapshot_download()` or `from_pretrained()`.

## Model loading pipeline

Step-by-step path from storage to inference-ready GPU:

```
1. Disk (SSD / Network Volume)
   └─ OS reads .safetensors bytes; deserializes tensor metadata + raw data into CPU memory

2. System RAM
   └─ Full weight tensors live here in FP32 or the stored dtype
   └─ Bottleneck: disk I/O speed (NVMe ~7 GB/s, Network Volume ~1–2 GB/s)

3. PCIe Bus
   └─ DMA (Direct Memory Access) transfer using pinned/page-locked memory
   └─ Pinned memory bypasses OS virtual memory paging → faster DMA
   └─ PCIe 4.0 x16: ~32 GB/s theoretical; real-world ~20 GB/s
   └─ Bottleneck for very large models (70B+ = 140 GB of data to move)

4. VRAM
   └─ Weights resident on GPU; CUDA kernels can now read them
   └─ model.to("cuda") or device_map triggers this transfer
```

Typical time breakdown for a 7B parameter FP16 model (~14 GB):
- Disk → RAM: 2–10 seconds (SSD vs. network)
- RAM → VRAM: ~1 second (PCIe limited)
- Total cold start: 3–15 seconds

## Autoregressive inference loop

Language models generate one token per forward pass:

```
Input tokens: ["The", "sky", "is"]
         ↓
Forward pass 1 → logits → sample → "blue"
         ↓
Forward pass 2 (inputs: ["The", "sky", "is", "blue"]) → "and"
         ↓
... repeat until <EOS> or max_length
```

**KV Cache**: on each forward pass, the model computes Key and Value tensors for every token in the context. Without caching, these are recomputed from scratch on every pass. With KV cache, they are stored in VRAM and reused:

```
VRAM usage during generation:
  Model weights:  fixed (e.g. 14 GB for 7B FP16)
  KV Cache:       grows linearly with sequence length
                  ≈ 2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes
```

Long sequences become memory-expensive not because the model grows, but because the KV cache grows. Cross-ref: VRAM formula with KV buffer → `06-Model-GPU-Interaction.md`.

## Quantization libraries

(Quantization formats and bit-width comparison → `06-Model-GPU-Interaction.md`. This covers the Python library side.)

### AutoAWQ — production pre-quantization
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Calibrate and quantize to disk (run once)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("mistral-7b-awq-4bit")
```
- Quantization happens offline; the saved model loads fast every time
- Best for: production deployments where startup speed matters

### BitsAndBytes — load-time quantization
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config)
```
- Quantization happens at load time on every cold start — adds seconds
- Best for: experimentation, prototyping, fine-tuning with QLoRA

| Library | When quantized | Load speed | Use case |
|---|---|---|---|
| AutoAWQ | Offline, once | Fast | Production serving |
| BitsAndBytes | At load time | Slower | Prototyping / QLoRA |

## Fitting large models: CPU offload and MoE

When a model is too large for available VRAM:

### CPU Offloading
```python
pipe.enable_model_cpu_offload()
# or
model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
```
- Layers not currently computing stay in CPU RAM; only the active layer is in VRAM
- Reduces effective VRAM requirement to roughly the size of the largest single layer
- Throughput drops significantly due to PCIe round-trips per layer

### Mixture of Experts (MoE)
```
Standard dense model:  all N layers active per token → full VRAM occupied
MoE model (e.g. Mixtral 8×7B):  router selects 2 of 8 expert layers per token
  → Only 2/8 expert weights need to be in VRAM at any one time
```
- Active parameter count per forward pass is a fraction of total parameters
- Mixtral 8×7B: 47B total params, ~13B active per token — fits on a single A100
- No code change required; the routing is internal to the model architecture

## MIG vs CUDA MPS

Two strategies for running multiple models or workloads on a single GPU:

### MIG — Multi-Instance GPU
```bash
# Enable MIG mode on GPU 0
nvidia-smi -i 0 -mig 1

# Create a 3g.40gb instance (half an A100 80GB)
nvidia-smi mig -cgi 3g.40gb -C
```
- **Hardware-level** partitioning: each MIG slice gets its own isolated VRAM, L2 cache, and compute
- Available only on: A100, H100, A30 (enterprise cards)
- Zero interference between slices — strict SLA isolation guaranteed
- Use when: serving multiple independent models or tenants on one expensive GPU

### CUDA MPS — Multi-Process Service
```bash
# Start the MPS daemon
nvidia-cuda-mps-control -d
```
- **Software-level** time-slicing: multiple processes share CUDA cores and L2 cache dynamically
- Works on: any NVIDIA GPU including consumer RTX 4090
- Processes submit kernels to a shared command queue; MPS serializes them
- Risk: under heavy load, one process can starve others → latency spikes

| Feature | MIG | CUDA MPS |
|---|---|---|
| Isolation level | Hardware | Software |
| Hardware required | A100 / H100 only | Any NVIDIA GPU |
| VRAM isolation | Yes — hard partitioned | No — shared |
| Interference risk | None | Possible under contention |
| Best for | Strict SLA, multi-tenant | Utilization on consumer hardware |

## Hybrid training workflow (Mac Mini + RunPod)

Split workloads to match each machine's strengths:

### Mac Mini / Apple Silicon role
- **Framework**: MLX (Apple's native ML framework; uses Unified Memory — no PCIe copy between CPU and GPU)
- **Tasks**: data labeling, dataset cleaning, tokenization, small validation runs
- Unified Memory (up to 192 GB on M3 Ultra) is efficient for data-heavy preprocessing but slow for large matrix math vs. dedicated VRAM

### RunPod role
- **GPU**: A100 80GB or L40S for training throughput
- **Tasks**: the actual training loop — forward pass, loss calculation, backpropagation, weight updates
- Network Volume: mount dataset once, reuse across runs without re-upload

### LoRA — Low-Rank Adaptation
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
# Only LoRA adapter weights (~1–10 MB) are trained; base model weights are frozen
```
- Inserts small trainable matrices (rank-16) alongside frozen base weights
- Reduces trainable parameters by 99%+ → proportional VRAM and compute reduction
- Trained adapter is saved separately and merged at inference time

### Full workflow
```
1. Mac Mini: label forms, export dataset as JSONL to Network Volume
2. RunPod: mount Network Volume, load base model, apply LoRA config
3. RunPod: run training loop (N epochs), checkpoint adapter weights to Network Volume
4. Mac Mini: test adapter locally with MLX merge; validate outputs
5. RunPod (serverless): deploy merged model for production inference
```

## RunPod framework stack

Every component in the stack and how they communicate:

```
┌─────────────────────────────────────────────────┐
│  Your Python code (handler.py)                   │
│  runpod.serverless.start({"handler": handler})   │
└───────────────────┬─────────────────────────────┘
                    │ Python function calls
┌───────────────────▼─────────────────────────────┐
│  PyTorch                                         │
│  • Translates tensor ops → CUDA kernel calls     │
│  • ATen C++ backend dispatches to cuBLAS/cuDNN   │
└───────────────────┬─────────────────────────────┘
                    │ CUDA kernel launches (PTX bytecode)
┌───────────────────▼─────────────────────────────┐
│  NVIDIA CUDA Runtime + cuBLAS / cuDNN            │
│  • Manages kernel scheduling on SMs              │
└───────────────────┬─────────────────────────────┘
                    │ ioctl / driver calls
┌───────────────────▼─────────────────────────────┐
│  NVIDIA Container Toolkit (libnvidia-container)  │
│  • Passes /dev/nvidia* devices into container    │
│  • Mounts driver libraries from host into image  │
└───────────────────┬─────────────────────────────┘
                    │ Device passthrough
┌───────────────────▼─────────────────────────────┐
│  Host OS + NVIDIA GPU Driver                     │
│  • Owns physical GPU hardware                    │
│  • Exposes /dev/nvidia0, /dev/nvidiactl, etc.    │
└─────────────────────────────────────────────────┘
```

**Why driver passthrough matters**: Docker normally isolates containers from host hardware. NVIDIA Container Toolkit creates a controlled pass-through that lets the container's CUDA runtime talk to the host driver without needing a separate driver inside the image. The GPU driver version on the host sets the maximum CUDA version available in any container on that machine.

## FastAPI + vLLM production architecture

```
Client
  │ HTTP POST /generate
  ▼
FastAPI
  • Route matching, auth middleware, request validation
  • Pydantic model parses request body
  • Calls vLLM engine (subprocess or OpenAI-compatible API)
  • Streams response via Server-Sent Events (SSE)
  │
  ▼
vLLM
  • Continuous batching: fills GPU with requests from multiple clients simultaneously
  • PagedAttention: manages KV cache as paged virtual memory — no fragmentation
  • Dynamic memory paging: evicts cold KV pages when VRAM pressure rises
  │
  ▼
GPU (VRAM)
  • Model weights loaded once at startup
  • KV cache pages allocated / freed per request
```

SSE streaming loop in FastAPI:
```python
from fastapi.responses import StreamingResponse

async def generate_stream(prompt: str):
    async for token in vllm_engine.generate(prompt):
        yield f"data: {token}\n\n"

@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(generate_stream(request.prompt), media_type="text/event-stream")
```

**vLLM continuous batching vs. naive batching**: naive batching waits for a full batch before starting. Continuous batching inserts new requests into in-flight batches between decode steps — GPU utilization stays near 100% even with variable request arrival rates.
