"""Page images taken straight from each PDF page's one embedded scan, with no resampling."""

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pymupdf
from PIL import Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageImage:
    number: int  # 1-based
    data: bytes
    ext: str  # "png" (Flate, 115 DPI) or "jpeg" (240 DPI)
    image: Image.Image  # decoded RGB, same pixel size as the stored image


def open_pdf(data: bytes, name: str) -> pymupdf.Document:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError(f"{name}: cannot be opened as a PDF ({exc})") from exc
    if doc.page_count == 0:
        raise ValueError(f"{name}: contains 0 pages")
    return doc


def image_xrefs(doc: pymupdf.Document, name: str) -> list[int]:
    # Decodes nothing, so a malformed PDF fails before any page is processed.
    xrefs = []
    for page_index in range(doc.page_count):
        images = doc[page_index].get_images(full=True)
        if len(images) != 1:
            raise ValueError(
                f"{name} page {page_index + 1}: expected exactly 1 embedded image, "
                f"found {len(images)}"
            )
        xrefs.append(images[0][0])
    return xrefs


def page_images(doc: pymupdf.Document, xrefs: list[int], name: str) -> Iterator[PageImage]:
    # A generator: a decoded 1983 x 2806 page is ~16.7 MB, so hold one page at a time.
    for page_index, xref in enumerate(xrefs):
        number = page_index + 1
        info = doc.extract_image(xref)
        image = Image.open(io.BytesIO(info["image"])).convert("RGB")
        stored = (info["width"], info["height"])
        if image.size != stored:
            raise ValueError(
                f"{name} page {number}: decoded size {image.size[0]}x{image.size[1]} "
                f"does not equal stored size {stored[0]}x{stored[1]}"
            )
        log.debug("page %d: %dx%d px %s", number, image.size[0], image.size[1], info["ext"])
        yield PageImage(number=number, data=info["image"], ext=info["ext"], image=image)
