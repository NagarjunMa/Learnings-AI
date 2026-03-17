# The GPU: The "Factory Floor" of AI

Unlike the CPU (the "Manager"), the GPU is a high-throughput parallel I/O system designed for massive matrix math.

## Core Components
* **CUDA Cores:** General-purpose parallel processors for standard math operations.
* **Tensor Cores:** Specialized hardware for 4x4 matrix operations; essential for **Mixed Precision** (FP16/BF16) and high-speed inference.
* **VRAM (Video RAM):** High-speed memory (GDDR6X/HBM3) where model weights and "conversation memory" (KV Cache) must reside.
* **Memory Bandwidth:** The speed at which data moves from VRAM to Cores. This is often the primary bottleneck in LLMs (**Memory-bound** vs. **Compute-bound**).
