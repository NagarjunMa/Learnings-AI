# Engineering Decisions

## Pods vs Serverless vs Bare GPU

| Factor | Pods | Serverless | Bare/Self-Hosted |
|---|---|---|---|
| Workload pattern | Continuous or scheduled | Burst / event-driven | Constant, high utilization |
| State requirements | Persistent disk, long sessions | Stateless preferred | Full control |
| Traffic predictability | Steady or known schedule | Variable, unpredictable | Constant |
| Job duration | Any | Best for <10 min (short jobs) | Any |
| Cold start tolerance | None (always on) | Must tolerate or use min_workers | None |
| Cost model | Per second of uptime | Per second of execution | Hardware + power |
| Multi-step pipelines | Yes (stateful) | Requires external state store | Yes |
| Billing floor | While Pod is running | Zero when idle (min_workers=0) | Fixed hardware cost |

## When to Use Pods

- **Full model training** — long jobs, checkpoint to volume, can survive spot interruptions
- **Jupyter / development** — interactive, stateful, need persistent filesystem
- **Persistent inference at >50% utilization** — continuous server is cheaper than Serverless if utilization is high (Serverless billed per-second adds up vs flat rate Pod)
- **Multi-step stateful pipelines** — steps that share local files or in-memory state across requests
- **Custom network setup** — SSH tunneling, port forwarding, debug tools

## When to Use Serverless

- **Production inference APIs with variable traffic** — pay only for actual requests, scale to 0 overnight
- **Event-driven pipelines** — trigger from webhook/queue, process, done
- **Zero-idle-cost requirement** — marketing endpoints, demos, low-traffic apps
- **Bursty workloads** — handle 10x traffic spikes without pre-provisioning

## On-Demand vs Spot

| | On-Demand | Spot |
|---|---|---|
| Interruption | Never | 5s SIGTERM warning |
| Cost | Full rate | Up to 60% off |
| When to use | Production inference, time-sensitive training, customer-facing | Batch processing, training with checkpointing, offline jobs |
| Checkpoint strategy | Optional | Required for safety |

Spot is viable when:
- Training saves checkpoints to Network Volume every N steps
- Batch job is idempotent (can restart from last checkpoint)
- Job can tolerate 5s interruption warning

Spot is too risky when:
- Real-time inference serving (interruption = downtime)
- Jobs with no checkpoint mechanism
- Very long single-step jobs (no natural checkpoint boundary)

## Secure Cloud vs Community Cloud

| | Secure Cloud | Community Cloud |
|---|---|---|
| Network Volumes | Yes | No |
| SLA | Yes | No |
| Compliance suitability | Higher | Lower |
| Price | ~10-20% more | Cheapest |
| Use when | Production, need volumes, compliance matters | Dev/test, budget training without storage dependency |

Key: if you need Network Volumes (model caching, persistent checkpoints), you must use Secure Cloud.

## Hybrid Production Pattern

Best-practice production architecture:
- **Serverless endpoint**: `min_workers=1` (one active worker for fast first response), `max_workers=20` (flex burst)
- **Spot Pods**: offline batch jobs (embeddings, reranking, eval runs) at 60% cost reduction
- **On-Demand Pod**: development/fine-tuning environment with persistent volume

```
[User request] → Serverless (min=1 active, flex for burst)
                       ↓
[Batch job queue] → Spot Pods (50-60% cost savings)
                       ↓
[Dev/finetune] → On-Demand Pod + Network Volume
```

## Cost Optimization Decision Tree

1. **Can the model be quantized without quality loss?**
   - Yes → INT4/AWQ → smaller GPU → 2-3× cost reduction
   - No → stay FP16

2. **Is this LoRA fine-tuning vs full fine-tune?**
   - LoRA → RTX 4090 (24GB) often sufficient vs A100 → ~$0.74 vs ~$1.89/hr

3. **What is the utilization pattern?**
   - >50% → On-Demand Pod cheaper than Serverless
   - <50% → Serverless cheaper

4. **Does the job tolerate interruption?**
   - Yes + checkpoint mechanism → Spot Pod
   - No → On-Demand

5. **Is cold start acceptable?**
   - Yes → `min_workers=0`, idle_timeout=5s, lowest cost
   - No → `min_workers=1` or optimize cold start path

## GPU Sizing by Role

| Role | GPU Choice | Reasoning |
|---|---|---|
| Rapid prototyping | RTX 4090 | Cheapest per GB VRAM at 24GB, fast iteration |
| 7B inference (prod) | RTX 4090 or A40 | 7B INT4 fits with room; L40S for higher throughput |
| 13B inference (prod) | L40S or A100 80GB | Need 48GB+; L40S better compute/cost for prefill |
| 70B inference (prod) | H100 PCIe or H200 | Need BW for decode, 80GB+ VRAM |
| 7B fine-tune (LoRA) | RTX 4090 | QLoRA fits in 24GB |
| 13B fine-tune (LoRA) | A40 or A100 | 13B INT4 base + adapter fits in 48GB |
| Large-scale training | A100 SXM / H100 SXM | NVLink for efficient tensor parallelism |
| Multi-modal (large) | H100 NVL / H200 | Large VRAM for image+text model sizes |
