"""Page classification: zero-shot CLIP on the CPU, checked against box geometry."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import ImageOps
from PIL.Image import Image

from eroll.pages.boxes import (
    PAGE_TYPE_NON_VOTER,
    PAGE_TYPE_VOTER,
    Box,
    column_fill,
    detect_boxes,
    geometry_label,
    grid_problems,
)

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Not tuned on the ground truth.
VOTER_PROMPTS: tuple[str, ...] = (
    "a scanned page with a grid of thirty bordered boxes of voter details",
    "a scanned electoral roll page listing voters in three columns of boxes",
)
NON_VOTER_PROMPTS: tuple[str, ...] = (
    "a scanned cover page of an electoral roll with a map and paragraphs of text",
    "a scanned page with a summary table of elector counts",
)

# Fitted on 60 calibration PDFs from 8 constituencies, 1744 pages (1564 voter, 180
# non-voter), disjoint from the ground truth. Re-fit if the prompts or model change.
CLIP_THRESHOLD = 0.15
# Share of calibration pages sent to qwen_fallback at 0.15: 193 of 1744.
CLIP_CALIBRATION_FALLBACK_RATE = 0.1107

OUTCOME_CLIP_VOTER = "clip_voter"
OUTCOME_CLIP_NON_VOTER = "clip_non_voter"
OUTCOME_QWEN_FALLBACK = "qwen_fallback"


@dataclass(frozen=True)
class Clip:
    model: CLIPModel
    processor: CLIPProcessor
    class_embeddings: np.ndarray  # shape (2, d), row 0 voter, row 1 non_voter, unit rows
    logit_scale: float


@dataclass(frozen=True)
class PageAnalysis:
    boxes: list[Box]
    column_fill: float | None
    geometry_label: str
    grid_problems: list[str]
    p_voter: float
    p_non_voter: float
    margin: float
    outcome: str  # OUTCOME_CLIP_VOTER | OUTCOME_CLIP_NON_VOTER | OUTCOME_QWEN_FALLBACK
    final_label: str | None  # None until the Qwen fallback supplies it


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


@functools.cache
def load_clip() -> Clip:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    class_rows = []
    with torch.no_grad():
        for prompts in (VOTER_PROMPTS, NON_VOTER_PROMPTS):
            inputs = processor(text=list(prompts), return_tensors="pt", padding=True)
            features = model.get_text_features(**inputs).pooler_output.numpy()
            class_rows.append(_unit_rows(_unit_rows(features).mean(axis=0)))
        logit_scale = float(model.logit_scale.exp().item())

    return Clip(
        model=model,
        processor=processor,
        class_embeddings=np.stack(class_rows),
        logit_scale=logit_scale,
    )


def probabilities(
    image_embedding: np.ndarray, class_embeddings: np.ndarray, logit_scale: float
) -> tuple[float, float]:
    """(p_voter, p_non_voter): softmax of logit_scale x cosine over the two classes."""
    cosines = _unit_rows(class_embeddings) @ _unit_rows(image_embedding)
    logits = logit_scale * cosines
    # Subtract the max so exp cannot overflow at CLIP's logit scale of 100.
    exp = np.exp(logits - logits.max())
    p = exp / exp.sum()
    return float(p[0]), float(p[1])


def clip_probabilities(clip: Clip, image: Image) -> tuple[float, float]:
    import torch

    # CLIPProcessor centre-crops, which would cut off a portrait page's header and
    # footer; padding to a white square keeps the whole page in view.
    side = max(image.size)
    square = ImageOps.pad(image, (side, side), color="white")
    inputs = clip.processor(images=square, return_tensors="pt")
    with torch.no_grad():
        embedding = clip.model.get_image_features(**inputs).pooler_output.numpy()[0]
    return probabilities(embedding, clip.class_embeddings, clip.logit_scale)


def decide(
    p_voter: float, p_non_voter: float, geometry: str, threshold: float
) -> tuple[str, str | None]:
    """(outcome, final label); the final label is None for qwen_fallback."""
    # 0 means "trust CLIP whenever it agrees with geometry".
    if not 0 <= threshold <= 1:
        raise ValueError(f"CLIP threshold {threshold} is outside [0, 1]")
    if geometry not in (PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER):
        raise ValueError(f"geometry label {geometry!r} is not a page type")

    # Trust CLIP only when it is confident and geometry agrees; anything else goes to Qwen.
    clip_label = PAGE_TYPE_VOTER if p_voter > p_non_voter else PAGE_TYPE_NON_VOTER
    margin = abs(p_voter - p_non_voter)
    if margin >= threshold and clip_label == geometry:
        outcome = OUTCOME_CLIP_VOTER if clip_label == PAGE_TYPE_VOTER else OUTCOME_CLIP_NON_VOTER
        return outcome, clip_label
    return OUTCOME_QWEN_FALLBACK, None


def analyse_page(clip: Clip, image: Image, threshold: float) -> PageAnalysis:
    # Shared by the CPU worker, run_local and calibrate, so the calibrated rule is the deployed one.
    # Every page is analysed; page position is never assumed.
    boxes = detect_boxes(image)
    page_width = image.size[0]
    geometry = geometry_label(boxes, page_width)
    p_voter, p_non_voter = clip_probabilities(clip, image)
    outcome, final_label = decide(p_voter, p_non_voter, geometry, threshold)
    return PageAnalysis(
        boxes=boxes,
        column_fill=column_fill(boxes, page_width),
        geometry_label=geometry,
        grid_problems=grid_problems(boxes),
        p_voter=p_voter,
        p_non_voter=p_non_voter,
        margin=abs(p_voter - p_non_voter),
        outcome=outcome,
        final_label=final_label,
    )
