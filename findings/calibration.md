# Calibration of the CLIP margin threshold

## Summary

The CLIP margin threshold was fitted once, on 60 PDFs from 8 constituencies (1,744 pages: 1,564 voter, 180 non-voter) that share no file with the ground truth. The selected threshold is 0.15. At 0.15 no page of either class is trusted with the wrong label, and 193 of the 1,744 pages (11.07%) go to the Qwen fallback.

| Class | Pages | Wrongly trusted at 0.15 |
|---|---|---|
| voter | 1,564 | 0 |
| non-voter | 180 | 0 |

| Set | Pages | Sent to Qwen | Rate |
|---|---|---|---|
| Calibration | 1,744 | 193 | 11.07% |
| Ground truth | 184 | 18 | 9.78% |

## The rule

Every candidate from 0.00 to 1.00 in steps of 0.01 is tried, and the smallest one at which neither class has any wrongly trusted page is selected. A page counts as wrongly trusted only if the deployed rule would trust it (margin at or above the threshold and CLIP agreeing with geometry) and CLIP's label is wrong. The threshold is therefore fitted for CLIP and geometry together, not for CLIP alone.

The smallest safe threshold is the appropriate target because the threshold trades cost against risk in one direction only. A higher threshold sends more pages to Qwen, which costs GPU time but cannot make a label worse. A lower one saves fallback calls but risks trusting a wrong CLIP label, and a wrongly trusted voter page would be silently skipped.

## Guards

- The calibration set is checked by SHA-256 against the ground truth, so a renamed copy of a test PDF cannot enter the calibration set.
- The fit refuses to run on fewer than 60 PDFs or 6 constituencies, so the threshold is not tuned to one printer's scans, and so the check set is large enough for zero errors to be meaningful.
- True labels come from page position (pages 1, 2 and the last page are non-voter), so no hand labelling is needed.
- The same analysis function runs in the CPU worker, the evaluation run and the calibration, so the fitted rule is the deployed rule.

## Limits

180 non-voter pages is a small sample: zero errors there shows the threshold works on these pages, not that it can never fail. The fit was run once; changing the prompts or the CLIP model means fitting again. The ground-truth fallback rate (9.78%) is close to the calibration rate (11.07%), which suggests the threshold carries over to unseen rolls of the same format.
