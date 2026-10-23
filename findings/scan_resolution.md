# Scan resolution: 115 DPI pages have 2.2x the field error rate

## Summary

The rolls use two scan formats: 1983 x 2806 JPEG pages (240 DPI) and 949 x 1343 Flate pages (115 DPI). On the low-resolution pages, 0.62% of fields were incorrect, against 0.28% on the high-resolution pages. Record accuracy decreases from 98.15% to 95.98%. The low-resolution sample is small: 821 records from a single 31-page PDF.


| Page size | Records | Field accuracy | Record accuracy |
|---|---|---|---|
| 1983x2806 (240 DPI) | 4,008 | 99.72% | 98.15% |
| 949x1343 (115 DPI) | 821 | 99.38% | 95.98% |

| Field | Errors at 240 DPI | Errors at 115 DPI |
|---|---|---|
| name | 47 (1.17% of records) | 23 (2.80%) |
| relation name | 33 (0.82%) | 11 (1.34%) |
| house number | 1 (0.02%) | 7 (0.85%) |
| relation type | 4 | 0 |
| epic id | 2 | 0 |
| gender | 2 | 0 |

## Analysis

Both formats reach the model as the same number of tokens. vLLM is pinned to `min_pixels` 802,816, so every crop is upscaled to at least that many pixels, and the first crop at each page size measured 1,066 prompt tokens. A 115 DPI box is approximately 300 pixels wide before that upscale, against approximately 620 at 240 DPI, so the model receives the same token budget but less than half the source detail, and thin strokes and small letters merge.

The additional errors concentrate in the fields that depend most on fine detail. The name error rate more than doubles. House numbers increase from 1 error to 7, because they are short, mix digits and letters, and a thin printed dash is difficult to distinguish from an empty field at this resolution (a dash-only house number is already normalised to empty for this reason). Schema-constrained fields have no errors at the lower resolution.

## Implications

Page images are taken directly from the PDF's embedded scan, so the pipeline already uses the full stored resolution, and no higher-resolution source exists. Low-resolution rolls therefore require more review, and per-size accuracy is reported separately so that a mix of formats does not mask the difference.
