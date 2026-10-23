import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from eroll.pages.boxes import PAGE_TYPE_NON_VOTER, PAGE_TYPE_VOTER
from eroll.pages.classify import (
    OUTCOME_CLIP_NON_VOTER,
    OUTCOME_CLIP_VOTER,
    OUTCOME_QWEN_FALLBACK,
    decide,
    probabilities,
)

# CLIP ViT-B/32 embeds into 512 dimensions; smaller ones exercise the same arithmetic.
DIMENSIONS = st.sampled_from((2, 3, 16, 512))
# 100 is CLIP's clamped logit scale (overflow risk); small values give a flat softmax.
LOGIT_SCALES = st.one_of(st.just(100.0), st.floats(0.01, 100.0))


def unit_vector(data: st.DataObject, dimension: int) -> np.ndarray:
    vector = data.draw(arrays(np.float64, dimension, elements=st.floats(-1.0, 1.0)))
    norm = np.linalg.norm(vector)
    assume(norm > 1e-3)  # a near-zero vector has no direction to normalise to
    return vector / norm


# CLIP probabilities are a distribution and the label is the argmax.
@given(st.data(), DIMENSIONS, LOGIT_SCALES)
def test_probabilities_are_a_distribution_with_argmax_label(
    data: st.DataObject, dimension: int, logit_scale: float
) -> None:
    image = unit_vector(data, dimension)
    voter, non_voter = unit_vector(data, dimension), unit_vector(data, dimension)

    p_voter, p_non_voter = probabilities(image, np.stack([voter, non_voter]), logit_scale)

    assert 0 <= p_voter <= 1
    assert 0 <= p_non_voter <= 1
    assert abs(p_voter + p_non_voter - 1) < 1e-6

    # Cosine ties (to rounding) have no defined argmax, so only clear gaps are compared.
    voter_cosine, non_voter_cosine = float(voter @ image), float(non_voter @ image)
    assume(abs(voter_cosine - non_voter_cosine) > 1e-9)
    assert (p_voter > p_non_voter) == (voter_cosine > non_voter_cosine)


GEOMETRY_LABELS = st.sampled_from((PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER))
# 1.0 is the uncalibrated placeholder, where only a page CLIP is certain about passes;
# 0.0 is the bottom of the calibration grid, where agreement with geometry is enough.
THRESHOLDS = st.one_of(st.sampled_from((0.0, 1.0)), st.floats(0.0, 1.0))


# The cascade trusts CLIP only when confident and agreeing with geometry.
@given(st.data(), st.floats(0.0, 1.0), GEOMETRY_LABELS)
def test_decide_trusts_clip_only_when_confident_and_agreeing_with_geometry(
    data: st.DataObject, p_voter: float, geometry: str
) -> None:
    p_non_voter = 1 - p_voter
    margin = abs(p_voter - p_non_voter)
    # Random thresholds almost never land on the margin itself, so draw that case
    # explicitly: margin == threshold must trust CLIP.
    threshold = data.draw(
        st.just(margin) if margin > 0 and data.draw(st.booleans()) else THRESHOLDS
    )

    outcome, final_label = decide(p_voter, p_non_voter, geometry, threshold)

    clip_label = PAGE_TYPE_VOTER if p_voter > p_non_voter else PAGE_TYPE_NON_VOTER
    if margin >= threshold and clip_label == geometry:
        expected = OUTCOME_CLIP_VOTER if clip_label == PAGE_TYPE_VOTER else OUTCOME_CLIP_NON_VOTER
        assert (outcome, final_label) == (expected, clip_label)
    else:
        assert (outcome, final_label) == (OUTCOME_QWEN_FALLBACK, None)


def test_decide_truth_table() -> None:
    # Every probability, margin and threshold here is a dyadic fraction, so the margin
    # equals the threshold exactly in binary floating point, not just to rounding.
    voter, non_voter = PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER
    cases = [
        # p_voter, p_non_voter, geometry, threshold -> outcome, final label
        (0.875, 0.125, voter, 0.5, OUTCOME_CLIP_VOTER, voter),  # margin above, agree
        (0.875, 0.125, non_voter, 0.5, OUTCOME_QWEN_FALLBACK, None),  # above, disagree
        (0.625, 0.375, voter, 0.5, OUTCOME_QWEN_FALLBACK, None),  # below, agree
        (0.625, 0.375, non_voter, 0.5, OUTCOME_QWEN_FALLBACK, None),  # below, disagree
        (0.75, 0.25, voter, 0.5, OUTCOME_CLIP_VOTER, voter),  # equal, agree
        (0.75, 0.25, non_voter, 0.5, OUTCOME_QWEN_FALLBACK, None),  # equal, disagree
        (0.25, 0.75, non_voter, 0.5, OUTCOME_CLIP_NON_VOTER, non_voter),  # equal, agree
        (0.125, 0.875, non_voter, 0.5, OUTCOME_CLIP_NON_VOTER, non_voter),  # above, agree
        (0.125, 0.875, voter, 0.5, OUTCOME_QWEN_FALLBACK, None),  # above, disagree
    ]
    for p_voter, p_non_voter, geometry, threshold, outcome, label in cases:
        assert decide(p_voter, p_non_voter, geometry, threshold) == (outcome, label), (
            p_voter,
            geometry,
        )


def test_decide_rejects_invalid_threshold_and_geometry() -> None:
    for threshold in (1.5, -0.5):
        with pytest.raises(ValueError, match="threshold"):
            decide(0.75, 0.25, PAGE_TYPE_VOTER, threshold)
    with pytest.raises(ValueError, match="geometry"):
        decide(0.75, 0.25, "summary", 0.5)
