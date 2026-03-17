# Hybrid Training Workflow: Mac Mini + RunPod

Leverage the strengths of both local and cloud hardware for document digitalization (OCR) and other ML projects.

## The Pipeline

* **Local (Mac Mini M4):** Perform data labeling and preprocessing. Use the **MLX Framework** to utilize **Unified Memory** (where CPU and GPU share the same RAM pool).
* **Cloud (RunPod):** Execute the heavy **Training Loop**. Use **NVIDIA A100/L40S** for high-speed tensor throughput.
* **Speed Tactic (LoRA):** Use **Low-Rank Adaptation** to train only a fraction of the weights. This makes training significantly faster and allows large models to fit on cheaper GPUs (like the **RTX 4090**).

## When to Use Each

| Task | Where |
|------|--------|
| Data labeling / cleaning | Local (Mac Mini) |
| Preprocessing / tokenization | Local (Mac Mini) |
| Full training run | Cloud (RunPod A100/L40S) |
| LoRA fine-tuning | Cloud (RunPod RTX 4090) |
| Inference / testing | Local or Serverless |
