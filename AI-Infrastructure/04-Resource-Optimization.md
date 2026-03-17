# Resource Optimization

To prevent your AI from consuming excessive resources or crashing (**OOM — Out of Memory**), use these tactics.

## VRAM Calculation Rule of Thumb

For a model with $P$ (Billion Parameters):
* **FP16 (Half-Precision):** $P \times 2$ GB
* **INT8 (Quantized):** $P \times 1$ GB
* **INT4 (Highly Quantized):** $P \times 0.5$ GB
* *Note: Always add a **20% buffer** for KV Cache and Activations.*

## Advanced Tactics
* **Quantization:** Reducing weight precision to save 4x memory with minimal accuracy loss.
* **MIG (Multi-Instance GPU):** Partitioning one A100/H100 into 7 isolated hardware slices to run multiple small services on one expensive card.
* **PagedAttention:** Managing VRAM like virtual memory to reduce fragmentation in the KV Cache (used in frameworks like **vLLM**).
