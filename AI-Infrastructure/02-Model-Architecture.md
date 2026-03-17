# Model Architecture & Loading Flow

When downloading a model (e.g., from Hugging Face), the "Blueprint" consists of:
1. **Weights:** The numerical tensors (`.safetensors` is the preferred format for speed and security).
2. **Config:** The structural manual (layer count, attention heads).
3. **Tokenizer:** The translator from human text to integer IDs.

## The Loading Sequence
* **Storage:** Model sits on a Disk or Network Volume.
* **System RAM:** Model is read into CPU memory.
* **PCIe Bus:** Data travels across the motherboard bridge (the "bottleneck").
* **VRAM:** Tensors are mapped into the GPU's local memory for active computation.
