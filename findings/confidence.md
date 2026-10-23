# Confidence: recorded, not yet useful for flagging

## Summary

Every record stores a confidence: the mean log probability of the tokens in the model's answer. Observed values fall in a narrow band from -0.0190 to -0.0002. The confidence threshold is unset by default, so no record is flagged on confidence, and it has not been calibrated against the known mistakes.

## Why the band is so narrow

The answer is decoded against a JSON schema at temperature 0. Most of its tokens are forced by the schema: braces, quotes, colons and the eight key names, whose log probability is close to 0. The few tokens that carry a name or a house number are averaged together with all of those, so even a shaky name read barely moves the mean.

## Why it matters

Names and relation names are 88% of all field errors, and they pass every structural validator. Confidence is the only per-record signal that could route them to review. With the mean squeezed into a band about 0.02 wide, a threshold would either flag almost nothing or flag a large share of correct records.

## How it is exposed

- `CONFIDENCE_THRESHOLD` must be at most 0. When set, a record below it gets the `low_confidence` flag, and the review page shows the value only for records flagged that way.
- A Prometheus histogram of record confidence uses buckets that are dense near 0 (-0.001, -0.002, -0.005 and so on), so the distribution is visible on the dashboard despite the narrow range.

## Next step

A score over the field tokens only, or the minimum token log probability instead of the mean, would not be diluted by schema tokens. Either would need fitting against the scored mistakes before a threshold is set. No such measurement exists yet.
