# Box detection: classical OpenCV, exact on every voter page

## Summary

Voter boxes are found with plain OpenCV, with no learned parts. On all 166 ground-truth voter pages, the number of detected boxes equals the number of hand-checked records: 158 full pages of 30 boxes and 8 partial pages holding 3 to 19. Because every threshold is a fraction of the page size, the same settings work on both scan formats (949 and 1983 pixels wide).

## How it works

1. Convert to greyscale and binarise with a local (adaptive) threshold, so uneven scan brightness does not matter. The block size is 1.7% of the page width.
2. Keep only long strokes: a morphological opening with a horizontal kernel 2.2% of the page width and a vertical one 1.0% of the page height removes text and leaves the drawn box borders. A small closing joins broken corners.
3. Take contours and keep rectangles of voter-box size and shape: 24% to 38% of the page width, 6% to 12% of the height, aspect ratio 1.5 to 4.5, and at least 1.5% of the page area.
4. Remove duplicates by overlap (IoU 0.70), largest first, so an outer border wins over the inner one.
5. Group into rows by vertical centre and sort into reading order.

Each box is cropped with a 2% inset per edge, which drops the drawn border but keeps the serial number and EPIC id printed in the corners.

## Geometry as a page check

The detector also tells voter pages from the rest. Column fill is the median box width over one third of the page width.

| Pages | Column fill |
|---|---|
| voter (166) | 0.940 to 0.945 |
| non-voter with any box-sized rectangle (8 of 18) | 0.740 to 0.908 |
| non-voter with none (10 of 18) | no boxes |

The cut-off is 0.92. Box width is used rather than box count, so a voter page with a missed box is still a voter page. A grid check then fails such a page loudly: rows other than the last must hold exactly 3 boxes, there can be at most 10 rows and 30 boxes, and left edges must line up in at most 3 columns. A job with a broken voter page fails with the list of bad pages rather than dropping records. 15 non-voter pages had grid problems (their table cells do not form a voter grid), which is expected; grid problems only matter on voter pages.

## Limits

The gap around the cut-off is narrow: the lowest voter page sits 0.020 above it and the highest non-voter page 0.012 below. The values are measured on one roll format with 3 columns and up to 10 rows. A different layout needs the column fill re-measured and the CLIP threshold refitted.
