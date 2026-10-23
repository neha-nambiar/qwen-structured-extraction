# Scan resolution: 115 DPI pages have 2.2x the field error rate

## Summary

The rolls come in two scan formats: 1983 x 2806 JPEG pages (240 DPI) and 949 x 1343 Flate pages (115 DPI). On the low-resolution pages, 0.62% of fields were wrong against 0.28% on the high-resolution pages. Record accuracy drops from 98.15% to 95.98%. The low-resolution sample is small: 821 records from a single 31-page PDF.


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

## Why

Both formats reach the model as the same number of tokens. vLLM is pinned to `min_pixels` 802,816, so every crop is upscaled to at least that many pixels, and the first crop at each page size measured 1,066 prompt tokens. A 115 DPI box is only about 300 pixels wide before that upscale, against about 620 at 240 DPI, so the model gets the same token budget but much less real detail: thin strokes and small letters blur together.

The extra errors land where detail matters most. Names roughly double. House numbers go from 1 error to 7, because they are short, mix digits and letters, and a thin printed dash is hard to tell from an empty field at this resolution (a dash-only house number is already normalised to empty for this reason). Constrained fields stay clean at both resolutions.

## What it means

Page images are taken straight from the PDF's embedded scan, so the pipeline already uses every stored pixel; there is no higher-resolution source to fall back to. A low-resolution roll should be expected to need more review, and the per-size accuracy is reported separately so a mix of formats does not hide it.
