import io

import numpy as np
import pymupdf
import pytest
from PIL import Image

from eroll.pages.pdf import image_xrefs, open_pdf, page_images


def png(width: int, height: int, seed: int) -> tuple[bytes, Image.Image]:
    # Random pixels, so a resampled or re-encoded-lossy decode would not match.
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image


def pdf_bytes(pages: list[list[bytes]]) -> bytes:
    doc = pymupdf.open()
    for images in pages:
        page = doc.new_page(width=595, height=842)
        for index, data in enumerate(images):
            # Different rects so two images on one page are two placements, not one.
            page.insert_image(pymupdf.Rect(0, 400 * index, 300, 400 * index + 400), stream=data)
    return doc.tobytes()


def test_one_png_per_page_yields_stored_pixels_and_bytes():
    first, first_image = png(40, 60, seed=1)
    second, second_image = png(50, 30, seed=2)
    doc = open_pdf(pdf_bytes([[first], [second]]), "two.pdf")

    xrefs = image_xrefs(doc, "two.pdf")
    pages = list(page_images(doc, xrefs, "two.pdf"))

    assert [page.number for page in pages] == [1, 2]
    for page, xref, original in zip(pages, xrefs, [first_image, second_image], strict=True):
        info = doc.extract_image(xref)
        assert page.ext == "png"
        assert page.data == info["image"]
        assert page.image.size == (info["width"], info["height"]) == original.size
        assert page.image.tobytes() == original.tobytes()


def test_two_images_on_page_one_names_page_and_count():
    first, _ = png(40, 60, seed=1)
    second, _ = png(40, 60, seed=2)
    doc = open_pdf(pdf_bytes([[first, second], [first]]), "bad.pdf")

    with pytest.raises(ValueError, match=r"bad\.pdf page 1: .*found 2"):
        image_xrefs(doc, "bad.pdf")


def test_non_pdf_bytes_raise():
    with pytest.raises(ValueError, match="x: cannot be opened as a PDF"):
        open_pdf(b"x", "x")
