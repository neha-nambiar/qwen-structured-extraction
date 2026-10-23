import base64
import io
import time
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from eroll.extraction.qwen import (
    FALLBACK_LONGEST_EDGE,
    confidence,
    downscale_for_fallback,
    extract_pages,
    png_data_url,
)


def test_confidence_is_mean_logprob():
    assert confidence([-0.1, -0.3], 1, 1) == pytest.approx(-0.2)


def test_confidence_without_logprobs_names_page_and_box():
    with pytest.raises(ValueError, match=r"page 4 box 7"):
        confidence([], 4, 7)


def test_png_data_url_round_trips_pixels():
    # Noisy pixels so any lossy encoding would show up as a mismatch.
    image = Image.frombytes("RGB", (37, 23), bytes(i * 7 % 256 for i in range(37 * 23 * 3)))
    url = png_data_url(image)
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    decoded = Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :])))
    assert decoded.mode == "RGB"
    assert decoded.tobytes() == image.tobytes()


# Up to twice FALLBACK_LONGEST_EDGE so both the shrink and the pass-through paths run.
# Mode "L" keeps the largest image at ~6.5 MB so the resize stays fast.
SIDES = st.integers(1, 2 * FALLBACK_LONGEST_EDGE)


# Fallback downscale never upscales and keeps the aspect ratio.
@given(SIDES, SIDES)
def test_downscale_for_fallback_caps_longest_edge_and_keeps_aspect(width: int, height: int):
    result = downscale_for_fallback(Image.new("L", (width, height)))
    longest, shortest = max(width, height), min(width, height)
    new_longest = max(result.size)
    assert new_longest == min(longest, FALLBACK_LONGEST_EDGE)
    # The shorter edge is rounded to whole pixels, so allow one pixel either way.
    assert abs(min(result.size) - shortest * new_longest / longest) <= 1


# Stand-in client: guards against yielding a page before all its crops are submitted,
# which depends only on timing, not on the model.
class _Completions:
    def create(self, **kwargs):
        time.sleep(0.01)
        token = SimpleNamespace(logprob=-0.1)
        message = SimpleNamespace(content="{}")
        choice = SimpleNamespace(message=message, logprobs=SimpleNamespace(content=[token]))
        return SimpleNamespace(choices=[choice])


def test_extract_pages_never_yields_a_partly_submitted_page():
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    crop = Image.new("RGB", (8, 8), "white")

    def pages():
        yield 1, [crop] * 3
        # Page 1 finishes while we wait, so it is yielded right after page 2's first
        # submit; the slow caller below then lets that first request finish too.
        time.sleep(0.1)
        yield 2, [crop] * 3

    yielded = {}
    for page, records in extract_pages(client, "model", pages(), max_concurrent=8):
        yielded[page] = [record.box for record in records]
        time.sleep(0.2)  # a commit over the tunnel
    assert yielded == {1: [1, 2, 3], 2: [1, 2, 3]}
