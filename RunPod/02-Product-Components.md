# RunPod Product Components

## Secure Cloud vs Community Cloud

| | Secure Cloud | Community Cloud |
|---|---|---|
| Operated by | RunPod directly | Third-party partner DCs |
| SLA | Yes | No |
| Network Volumes | Supported | Not supported |
| Uptime guarantees | Higher | Lower |
| Price | ~10-20% premium | Cheapest rates |
| Use for | Production, persistent storage | Budget training, non-critical jobs |

Trust model: Community Cloud partners sign RunPod's data security agreement but RunPod does not physically control those machines.

## Pods

Pods are persistent container VMs. Two pricing tiers:

### On-Demand Pods
- Non-interruptible — RunPod will not reclaim while you're running
- Fixed hourly rate
- Best for: production inference servers, long training that can't be checkpointed cheaply

### Spot Pods
- Up to 60% cheaper than On-Demand
- Interruptible: 5-second SIGTERM warning before reclaim
- Volume data retained after interruption
- Best for: batch jobs, training with checkpointing, offline processing

| | On-Demand | Spot |
|---|---|---|
| Interruption risk | None | Yes (5s warning) |
| Cost | Full rate | Up to 60% off |
| Volume retained | Yes | Yes |
| Best for | Production, no-tolerance jobs | Training w/ checkpoints, batch |

## Serverless

Event-driven workers, scale-to-zero when idle.

Architecture overview:
1. You push a Docker image with a RunPod handler
2. RunPod manages worker pool: 0 to max_workers
3. Request → queued → dispatched to idle/new worker → result returned
4. Worker idles for `idle_timeout` seconds then shuts down

Billing: per-second of execution only (not idle time, unless min_workers > 0)

**FlashBoot**: RunPod caches your worker at checkpoint state after first boot, subsequent cold starts ~48% complete in <200ms. Free. Requires traffic volume to warm the cache.

**Active workers**: always-on, billed even when idle — set min_workers to control floor
**Flex workers**: scale up on demand, spin down after idle_timeout

## Network Volumes

- NVMe-backed block storage
- Only available on Secure Cloud
- Price: $0.07/GB/month
- Sustained throughput: 200-400 MB/s
- Immutable downward: you can expand volume size, never shrink
- Attach at Pod creation time only — cannot hot-attach to running Pod
- Shared across Pods in the same data center (not cross-DC)
- Mount path inside container: `/runpod-volume/`
- Primary use: store model weights once, reuse across restarts without re-downloading

## Templates

Saved container launch configs. Fields:
- Container image (Docker Hub, private registry)
- Exposed ports (HTTP, TCP)
- Environment variables
- Start command / override entrypoint
- Volume mount paths
- Container disk size

Public templates: shared with RunPod community, anyone can use
Private templates: only visible in your account

Workflow:
1. Configure Pod manually
2. Save as template
3. Launch new Pods from template instantly (useful for reproducible dev envs)

## Engineering Decision Summary

| Component | Use when |
|---|---|
| Secure Cloud | Need Network Volumes or SLA |
| Community Cloud | Price-sensitive, no storage dependency |
| On-Demand Pod | Training that can't be interrupted; persistent inference |
| Spot Pod | Batch jobs with checkpointing; cost-sensitive training |
| Serverless | Variable-traffic inference API; zero-idle-cost requirement |
| Network Volume | Model weights > 5GB used repeatedly; avoid re-download |
| Template | Reproducible dev or inference container configs |
