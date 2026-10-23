# Page classification: 0 errors on 184 pages with a geometry check on CLIP

## Summary

On the 184 ground-truth pages (166 voter, 18 non-voter), the final page label was incorrect 0 times and no voter page was dropped. Every voter page was trusted from CLIP with a margin of at least 0.332 (median 0.973). Every non-voter page was sent to the Qwen fallback, and Qwen labelled all 18 correctly. CLIP alone assigns the voter label to 15 of the 18 non-voter pages, 6 of them with a margin above the 0.15 threshold. The box-geometry check rejects all 6.

![CLIP p(voter) against column fill for the 184 pages](img/clip_vs_fill.png)

Each point is one page. The dotted line is p(voter) = 0.575, where the margin reaches 0.15; the dashed line is the 0.92 column-fill cut-off. A CLIP-only rule trusts every page above the dotted line as voter, which includes 6 non-voter pages. All 6 lie left of the dashed line, so the geometry check rejects them. Pages with no detected box are plotted at "no boxes".

## Method

Each page is classified twice. CLIP ViT-B/32 compares the page with two voter and two non-voter text prompts and outputs p(voter), p(non-voter) and their margin. Separately, box detection measures column fill: the median detected box width over one third of the page width. The geometry label is voter at 0.92 or more. The CLIP label is trusted only if the margin is at least 0.15 and CLIP agrees with geometry. All other pages go to Qwen3-VL, which receives the page downscaled to a 1,280 px longest edge and is constrained to answer `voter` or `non_voter`.


| Ground truth | Pages | Outcome | Final label wrong |
|---|---|---|---|
| voter | 166 | all trusted from CLIP | 0 |
| non-voter | 18 | all sent to Qwen | 0 |

The 18 fallback pages fall into two groups:

- 15 pages with CLIP label voter and geometry label non-voter. Their margins range up to 0.815. The 6 above 0.15 would have been trusted as voter pages by a CLIP-only rule.
- 3 pages where CLIP agreed with geometry but the margin was below the threshold, at 0.008, 0.076 and 0.128.

## Analysis

CLIP processes the whole page downscaled to 224 x 224 pixels. At that resolution a cover page with bordered blocks, or a summary page of table cells, is visually similar to a grid of voter boxes. The prompts are zero-shot and were not tuned on these pages, so CLIP does not separate the two classes.

Column fill measures the property that separates the classes directly. Voter boxes span 0.940 to 0.945 of a column on every voter page. Table cells on non-voter pages are narrower: 10 non-voter pages had no box-sized rectangle, and the other 8 had a column fill between 0.740 and 0.908. The two checks fail on different pages, so requiring their agreement removes the CLIP errors.

## Limitations

The 18 non-voter pages are a small sample: 18 correct of 18 demonstrates correct fallback behaviour on these pages but does not establish that the fallback cannot fail. The ground truth takes true labels from position (pages 1, 2 and the last page are non-voter), which holds for this roll format. The fallback rate on this set is 9.78%, close to the 11.07% on the calibration set.
