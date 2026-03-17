# The Framework Handshake

How the key tools connect to form a complete AI development environment.

## Core Tools

* **PyTorch / TensorFlow:** The "Translator" that converts Python logic into **CUDA Kernels** the GPU executes.
* **Docker:** The "Box" that ensures the NVIDIA Drivers, CUDA libraries, and Python environment are identical across local and RunPod environments.
* **Network Volumes:** In RunPod, mount your dataset once to a volume so multiple Pods can access it instantly without re-downloading.

## How They Connect

```
Python Code (PyTorch)
        ↓
   CUDA Kernels
        ↓
  GPU (VRAM / Cores)
        ↑
Docker Container → ensures reproducible environment
        ↑
Network Volume  → shared dataset across pods
```
