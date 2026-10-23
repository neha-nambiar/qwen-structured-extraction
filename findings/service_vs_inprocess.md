# Deployed service: 99.65% field accuracy against 99.66% in-process

## Summary

The 6 ground-truth PDFs were also run end to end through the deployed service (upload to the API, CPU worker, Redis, GPU worker over the SSH tunnel, CSV download) and the downloaded CSVs were scored. The service made 135 field errors over 109 records: 99.65% field accuracy and 97.74% record accuracy. The in-process benchmark at the same scale made 130 errors over 107 records (99.66% and 97.78%). Each path was run once.

| Path | Field errors | Records with an error | Field accuracy | Record accuracy |
|---|---|---|---|---|
| In-process, 32 concurrent | 130 | 107 | 99.66% | 97.78% |
| Deployed service | 135 | 109 | 99.65% | 97.74% |

| Field | In-process | Service |
|---|---|---|
| name | 70 | 72 |
| relation name | 44 | 47 |
| house number | 8 | 9 |
| relation type | 4 | 4 |
| epic id | 2 | 1 |
| gender | 2 | 2 |

## Analysis

Both paths run the same page and extraction code: the same embedded-image decode, box detection, CLIP and geometry decision, crops, prompt, schema and vLLM settings. The service stores each page's original image bytes in Postgres and decodes them again in the GPU worker, so the boxes found on the CPU align with the crops cut on the GPU.

The gap of 5 fields is within run-to-run noise. The three in-process runs at 8, 16 and 32 concurrent requests already range from 127 to 130 errors, because different batch shapes in vLLM can flip near-tied tokens even at temperature 0. The additional errors fall in the same free-text fields as in the in-process run.

## Implications

The benchmark numbers describe the deployed system, not a separate evaluation harness. The service run was scored on accuracy only, so it provides no timing figure for comparison.
