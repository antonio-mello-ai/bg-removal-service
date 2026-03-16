"""
Background Removal Service - Model inference logic

Two models:
  1. IS-Net (via rembg) — automatic background removal
  2. SAM ViT-L — interactive refinement with point prompts
"""

import io
import logging
from typing import Literal

import numpy as np
import torch
from PIL import Image
from rembg import new_session, remove
from segment_anything import SamPredictor, sam_model_registry

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model handles (loaded once at startup)
# ---------------------------------------------------------------------------
_rembg_session = None
_sam_predictor: SamPredictor | None = None


def load_models() -> None:
    """Load IS-Net and SAM ViT-L into GPU memory. Called once at app startup."""
    global _rembg_session, _sam_predictor

    # IS-Net runs on CPU to avoid competing with SAM/vLLM for VRAM.
    # CPU inference is fast enough for segmentation (~200ms per image).
    logger.info("Loading IS-Net model (rembg: %s) on CPU...", config.REMBG_MODEL)
    _rembg_session = new_session(
        config.REMBG_MODEL,
        providers=["CPUExecutionProvider"],
    )
    logger.info("IS-Net loaded on CPU.")

    # SAM runs on CPU to avoid VRAM contention with vLLM.
    # Refinement is an interactive operation (user clicks) so ~2-3s latency
    # is acceptable. This keeps vLLM at 85% VRAM without conflicts.
    sam_device = "cpu"
    logger.info(
        "Loading SAM %s from %s on %s...",
        config.SAM_MODEL_TYPE, config.SAM_CHECKPOINT, sam_device,
    )
    sam = sam_model_registry[config.SAM_MODEL_TYPE](
        checkpoint=config.SAM_CHECKPOINT,
    )
    sam.to(device=sam_device)
    _sam_predictor = SamPredictor(sam)
    logger.info("SAM ViT-L loaded on %s.", sam_device)


def models_status() -> dict:
    """Return loading status of each model."""
    return {
        "isnet": "loaded (cpu)" if _rembg_session is not None else "not_loaded",
        "sam_vit_l": "loaded (cpu)" if _sam_predictor is not None else "not_loaded",
    }


# ---------------------------------------------------------------------------
# Automatic removal (IS-Net)
# ---------------------------------------------------------------------------

def auto_remove(image: Image.Image) -> Image.Image:
    """Remove background automatically using IS-Net.

    Returns an RGBA image with transparent background.
    """
    result = remove(
        image,
        session=_rembg_session,
        post_process_mask=True,
    )
    return result.convert("RGBA")


# ---------------------------------------------------------------------------
# Interactive refinement (SAM ViT-L)
# ---------------------------------------------------------------------------

def refine_with_sam(
    image: Image.Image,
    points: list[dict],
) -> Image.Image:
    """Refine the IS-Net mask using SAM point prompts.

    Strategy: start from the IS-Net auto-removal mask, then use SAM to
    identify regions around each user click and add/subtract them.

    - label=0 points ("this is background"): SAM segments the region around
      the point, then SUBTRACTS it from the IS-Net mask.
    - label=1 points ("this is foreground"): SAM segments the region around
      the point, then ADDS it to the IS-Net mask.

    This preserves the good parts of the auto mask and only modifies
    the specific areas the user clicked on.
    """
    if _sam_predictor is None:
        raise RuntimeError("SAM model not loaded")

    # Step 1: Get the IS-Net base mask
    auto_result = auto_remove(image)
    base_mask = np.array(auto_result.split()[3]) > 127  # bool mask from alpha

    # Step 2: Set image for SAM (done once, reused for all points)
    img_array = np.array(image.convert("RGB"))
    _sam_predictor.set_image(img_array)

    # Step 3: For each point, segment the region and modify the base mask
    for p in points:
        coord = np.array([[p["x"], p["y"]]])
        # Always tell SAM "this point is foreground" so it segments
        # the region AROUND the point. We then decide whether to
        # subtract or add based on the user's intent.
        sam_label = np.array([1])

        masks, scores, _ = _sam_predictor.predict(
            point_coords=coord,
            point_labels=sam_label,
            multimask_output=True,
        )

        # SAM returns 3 masks at different granularities (small, medium, large).
        # Always pick the smallest — both removal and addition are fine
        # detail operations on specific areas the user clicked.
        mask_areas = [m.sum() for m in masks]
        region = masks[int(np.argmin(mask_areas))]

        if p["label"] == 0:
            base_mask = base_mask & ~region
        else:
            base_mask = base_mask | region

    # Step 4: Build RGBA from modified mask
    rgba = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = img_array
    rgba[..., 3] = (base_mask * 255).astype(np.uint8)

    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# Post-processing: apply solid background
# ---------------------------------------------------------------------------

def apply_background(
    rgba_image: Image.Image,
    background: Literal["white", "gray"] = "white",
) -> bytes:
    """Composite RGBA image onto a solid background and return JPEG bytes."""
    bg_color = config.BACKGROUNDS.get(background, config.BACKGROUNDS["white"])

    bg = Image.new("RGB", rgba_image.size, bg_color)
    bg.paste(rgba_image, mask=rgba_image.split()[3])  # paste using alpha channel

    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=config.JPEG_QUALITY)
    return buf.getvalue()
