# RunPod Infrastructure: Pods vs. Serverless

Deployment strategy is chosen based on traffic patterns and cost-efficiency.

## Comparison Table

| Feature | **Pods (Persistent)** | **Serverless (On-Demand)** |
| :--- | :--- | :--- |
| **Availability** | Always-on; 24/7 uptime. | "Scales to zero" when not in use. |
| **Latency** | **Zero** cold start; instant response. | **Cold Start** (2s–15s) to pull image/load weights. |
| **Cost** | Fixed hourly rate (pay for idle time). | Pay-per-second (only when running). |
| **Use Case** | Development, high-traffic APIs. | Bursty traffic, background tasks (OCR). |

> **FlashBoot:** RunPod's technology to cache container images, significantly reducing Serverless cold start times by keeping data "warm" across their network.
