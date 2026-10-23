# Cost: $0.052 per 1,000 records

## Summary

At 32 concurrent requests on an RTX 4090 priced at $0.74 per hour, extraction took 0.252 s per record, which is $0.052 per 1,000 records. The full 4,829-record benchmark took 1,215 s of GPU time, approximately $0.25. At 8 concurrent requests the same work costs $0.077 per 1,000 records.

Cost per 1,000 records = seconds per record x 1,000 / 3,600 x hourly price. The hourly price is the configured rate for a secure-cloud, non-interruptible RTX 4090, not a measured bill.

## Method

The timed section is the Qwen page fallback (18 pages), one request per voter box, and writing the CSVs. It excludes:

- CLIP and box detection, which run on the CPU worker on the local machine and use no GPU time.
- Pod start-up: pulling an image of approximately 10 GB and downloading approximately 17 GB of Qwen weights before vLLM can serve.
- Idle time. The GPU worker deletes its own pod after 15 minutes without a job, so the worst-case idle tail is 0.25 h x $0.74 = $0.185 per session. That is approximately the extraction cost of 3,600 records, so for small batches the idle tail and start-up exceed the extraction cost.

## Implementation

- Only the GPU work is rented. Postgres, Redis, the API, CLIP and monitoring all run locally, and the pod reaches them over an SSH tunnel.
- The pod deletes itself when idle, and also when the tunnel is lost or vLLM is unhealthy, since those count as idle time. An eval pod deletes itself after a fixed lifetime.
- The CLIP and geometry check decides approximately 90% of pages on the CPU, so approximately 1 page in 10 requires a Qwen call.
- The service records busy and idle GPU seconds and the hourly price as metrics, so the dashboard shows live cost per 1,000 records and the idle share.

## Limitations

These are single-run figures on one GPU type.
