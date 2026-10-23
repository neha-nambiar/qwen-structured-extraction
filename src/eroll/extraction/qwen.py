"""Qwen3-VL requests to vLLM: field extraction, page fallback and the min_pixels probe."""

import base64
import io
import logging
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import openai
from openai import OpenAI
from PIL import Image, ImageDraw

from eroll.extraction.records import PROMPT, RECORD_SCHEMA, Record, parse_record

log = logging.getLogger(__name__)

# A record is well under 200 tokens of JSON; 1,066 prompt + 200 fits max-model-len 1,536.
MAX_NEW_TOKENS = 200

# At 32 x 32 px per vision token a 1280 x 905 page is 1,120 tokens, which with the
# prompt and output still fits max-model-len 1,536.
FALLBACK_LONGEST_EDGE = 1280

PAGE_LABELS = ("voter", "non_voter")

FALLBACK_PROMPT = (
    "This is one page of an electoral roll. Answer voter if the page is a grid of "
    "boxes each holding one voter's details, otherwise answer non_voter."
)

# Synthetic probe image (white, black border, no voter data). The processor upscales
# anything below min_pixels, so its token count shows whether min_pixels is applied.
PROBE_SIZE = (300, 100)

# A crop prompt is ~1,066 tokens with min_pixels 802,816 applied and ~352 without.
MIN_PROBE_PROMPT_TOKENS = 900


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def confidence(token_logprobs: list[float], page: int, box: int) -> float:
    """Mean per-token log probability of a completion."""
    if not token_logprobs:
        raise ValueError(f"page {page} box {box}: completion has no token log probabilities")
    return sum(token_logprobs) / len(token_logprobs)


def _failure_kind(error: openai.APIError) -> str:
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    if status is not None:
        return f"{name} {status}"
    if error.__cause__ is not None:
        return f"{name} ({type(error.__cause__).__name__})"
    return name


def _image_message(image: Image.Image, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": png_data_url(image)}},
                {"type": "text", "text": text},
            ],
        }
    ]


def extract_record(client: OpenAI, model: str, crop: Image.Image, page: int, box: int) -> Record:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_image_message(crop, PROMPT),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "record", "schema": RECORD_SCHEMA},
            },
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            logprobs=True,
        )
    except openai.APIError as error:
        raise RuntimeError(f"page {page} box {box}: {_failure_kind(error)}") from None

    choice = response.choices[0]
    content = choice.logprobs.content if choice.logprobs is not None else None
    score = confidence([token.logprob for token in content or []], page, box)
    record = parse_record(choice.message.content or "", page, box)
    return record.model_copy(update={"confidence": score})


def extract_pages(
    client: OpenAI,
    model: str,
    pages: Iterable[tuple[int, list[Image.Image]]],
    max_concurrent: int,
) -> Iterator[tuple[int, list[Record]]]:
    """Yield (page, records by box) as each page finishes, up to max_concurrent in flight."""
    # Crops are pulled only when a slot frees, so cropping overlaps requests and memory
    # stays bounded. The first failure stops new work and is re-raised.
    slots = threading.BoundedSemaphore(max_concurrent)
    pending: dict[int, list[Future[Record]]] = {}
    submitted: set[int] = set()
    executor = ThreadPoolExecutor(max_concurrent)

    def finished() -> Iterator[tuple[int, list[Record]]]:
        for page, futures in list(pending.items()):
            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            if page in submitted and all(future.done() for future in futures):
                del pending[page]
                submitted.discard(page)
                yield page, [future.result() for future in futures]

    try:
        for page, crops in pages:
            futures = []
            pending[page] = futures
            for box, crop in enumerate(crops, start=1):
                slots.acquire()
                future = executor.submit(extract_record, client, model, crop, page, box)
                future.add_done_callback(lambda _: slots.release())
                futures.append(future)
                yield from finished()
            submitted.add(page)
        while pending:
            for future in next(iter(pending.values())):
                future.exception()
            yield from finished()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


def downscale_for_fallback(image: Image.Image) -> Image.Image:
    """Shrink so the longest edge is FALLBACK_LONGEST_EDGE, keeping aspect; never upscale."""
    longest = max(image.size)
    if longest <= FALLBACK_LONGEST_EDGE:
        return image.copy()
    scale = FALLBACK_LONGEST_EDGE / longest
    size = tuple(max(1, round(side * scale)) for side in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def classify_page(client: OpenAI, model: str, image: Image.Image, page: int) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_image_message(downscale_for_fallback(image), FALLBACK_PROMPT),
            extra_body={"structured_outputs": {"choice": list(PAGE_LABELS)}},
            temperature=0,
            max_tokens=4,
        )
    except openai.APIError as error:
        raise RuntimeError(f"page {page} fallback: {_failure_kind(error)}") from None

    answer = response.choices[0].message.content
    if answer not in PAGE_LABELS:
        raise ValueError(f"page {page} fallback: answer {str(answer)[:20]!r} is not a label")
    return answer


def prompt_tokens(client: OpenAI, model: str, image: Image.Image) -> int:
    """Prompt token count for one extraction request on `image`."""
    response = client.chat.completions.create(
        model=model,
        messages=_image_message(image, PROMPT),
        temperature=0,
        max_tokens=1,
    )
    return response.usage.prompt_tokens


def probe(client: OpenAI, model: str) -> int:
    image = Image.new("RGB", PROBE_SIZE, "white")
    ImageDraw.Draw(image).rectangle(
        (0, 0, PROBE_SIZE[0] - 1, PROBE_SIZE[1] - 1), outline="black", width=3
    )
    return prompt_tokens(client, model, image)
