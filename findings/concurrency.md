# Concurrency: throughput saturates at 16 requests in flight

## Summary

The same 4,829 records on the same RTX 4090 were extracted at three `MAX_CONCURRENT_REQUESTS` settings. Increasing from 8 to 16 requests in flight reduced the time per record by 31%; increasing from 16 to 32 reduced it by a further 1.7%. Accuracy did not change beyond run-to-run noise. Each setting was run once.

![Seconds per record at 8, 16 and 32 concurrent requests](img/concurrency.png)

Time per record decreases steeply from 8 to 16 and remains nearly constant from 16 to 32: above 16 requests in flight the GPU is the bottleneck.

| Concurrency | Wall clock | s per record | Records/s | $ per 1,000 records | Field errors | Field accuracy | Record accuracy |
|---|---|---|---|---|---|---|---|
| 8 | 1801.6 s | 0.373 | 2.68 | 0.077 | 127 | 99.67% | 97.87% |
| 16 | 1235.5 s | 0.256 | 3.91 | 0.053 | 129 | 99.67% | 97.80% |
| 32 | 1215.0 s | 0.252 | 3.97 | 0.052 | 130 | 99.66% | 97.78% |

Cost assumes $0.74 per GPU hour. The wall clock covers the Qwen page fallback, extraction and CSV writing; CLIP runs before the timer starts.

## Analysis: throughput

Every box is a separate request: one crop of approximately 1,066 prompt tokens (predominantly image tokens) and a JSON answer capped at 200 tokens. vLLM batches the requests in flight. At 8, the batch is too small to keep the GPU fully utilised, so additional requests increase throughput directly. At 16 the GPU is the bottleneck, and additional requests only wait in vLLM's queue. 32 is the default: it is not slower than 16 and provides headroom while the next page is decoded and cropped.

Requests are not limited per page. A semaphore caps the number in flight across page boundaries, and crops for the next page are pulled only when a slot becomes free. As a result, the GPU does not idle at the end of a page while its last box completes, and memory remains bounded to a few decoded pages.

## Analysis: accuracy

Decoding is greedy (temperature 0) with a JSON schema, so in principle the outputs are independent of concurrency. In practice the three runs range from 127 to 130 field errors. Different batch shapes change the floating-point arithmetic slightly, which can flip a near-tie between two tokens. A spread of 3 fields in 38,632 is the run-to-run noise floor for these numbers; smaller differences are not significant.
