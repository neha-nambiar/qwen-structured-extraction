# Confidence: mean token log probability confined to -0.0190 to -0.0002

## Summary

Every record stores a confidence: the mean log probability of the tokens in the model's answer. Observed values fall in a narrow band from -0.0190 to -0.0002. The confidence threshold is unset by default, so no record is flagged on confidence, and it has not been calibrated against the scored errors.

## Analysis

The answer is decoded against a JSON schema at temperature 0. Most of its tokens are determined by the schema: braces, quotes, colons and the eight key names, whose log probability is close to 0. The few tokens that carry a name or a house number are averaged together with all of those, so an uncertain name reading has little effect on the mean.

## Implications

Names and relation names are 88% of all field errors, and they pass every structural validator. Confidence is the only per-record signal that could route them to review. With the mean confined to a band approximately 0.02 wide, a threshold would flag either a negligible share of records or a large share of correct records.

## Implementation

- `CONFIDENCE_THRESHOLD` must be at most 0. When set, a record below it receives the `low_confidence` flag, and the review page shows the value only for records flagged on that reason.
- A Prometheus histogram of record confidence uses buckets that are dense near 0 (-0.001, -0.002, -0.005 and so on), so the distribution remains resolvable on the dashboard despite the narrow range.

## Future work

A score over the field tokens only, or the minimum token log probability instead of the mean, would not be diluted by schema tokens. Either would need to be fitted against the scored errors before a threshold is set. This measurement has not been performed.
