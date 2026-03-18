# RunPod Platform Overview

## What Is RunPod

- Cloud GPU platform founded ~2022, 750k+ developers
- Per-second billing (min 1 minute on Pods, per-request on Serverless)
- Docker-based: bring any image, full root access, no proprietary runtimes
- Primary use case: AI/ML workloads — training, inference, fine-tuning

## Core Value Proposition vs Hyperscalers

| Factor | RunPod | AWS/GCP/Azure |
|---|---|---|
| Cost | 50-80% cheaper on like-for-like GPU | Premium pricing, reserved capacity costs |
| Contracts | None — pay-as-you-go | Spot = volatile, Reserved = 1-3 yr commitment |
| GPU access | Broad catalog incl. H100/H200/B200 | Often constrained by region/quota |
| Control | Full Docker, root, custom images | Managed services abstract GPU |
| Cold start | Seconds (Pods always on, Serverless ~200ms) | Lambda GPU cold starts can be minutes |
| Compliance/SLA | Limited (Secure Cloud has SLA) | Enterprise-grade SLAs |

## Two Compute Paradigms

**Pods** — Persistent virtual machines
- Container runs continuously; you pay per second of uptime
- Best for: training jobs, development, persistent inference servers

**Serverless** — Event-driven, scale-to-zero
- Workers spin up per request, idle workers terminate after N seconds
- Best for: production APIs with variable traffic, zero-idle-cost requirement

## Infrastructure Footprint

- 8+ global regions (US East, US West, EU-RO, EU-CZ, Asia, etc.)
- 30+ GPU SKUs from RTX 3090 to H200 and B200
- Community Cloud: partner data centers, cheaper, fewer guarantees
- Secure Cloud: RunPod-operated DCs, SLA, Network Volume support

## Cloud Type Quick Diff

| | Secure Cloud | Community Cloud |
|---|---|---|
| Operated by | RunPod | Third-party partners |
| SLA | Yes | No |
| Network Volumes | Yes | No |
| Price | ~10-20% higher | Cheaper |
| GPU availability | More predictable | Can be scarce |
| Use when | Production, volumes needed | Budget dev/training |

## When to Use RunPod vs Alternatives

| Scenario | Recommendation |
|---|---|
| Training on 1-4 GPUs, variable schedule | RunPod Spot Pods |
| Production inference, traffic spikes | RunPod Serverless |
| Compliance-heavy enterprise | Hyperscaler or self-hosted |
| Very long training (months) | Self-hosted or reserved cloud |
| Prototyping / exploration | RunPod On-Demand Pod |
| Offline batch at low cost | RunPod Spot Pods |
