"""
Background Removal Service - FastAPI application

Endpoints:
  GET  /health                    — Health check + model status
  POST /remove-background         — Automatic removal (IS-Net)
  POST /remove-background/refine  — Interactive refinement (SAM ViT-L)
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

import config
import models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_models()
    logger.info("Background Removal Service ready on port %s", config.PORT)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Background Removal Service",
    description="Automatic and interactive background removal for product images",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image(upload: UploadFile) -> Image.Image:
    """Read uploaded file into a PIL Image, validating size and format."""
    contents = upload.file.read()

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {size_mb:.1f}MB (max {config.MAX_IMAGE_SIZE_MB}MB)",
        )

    try:
        from io import BytesIO
        img = Image.open(BytesIO(contents))
        img.load()
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def _validate_background(background: str) -> str:
    if background not in config.BACKGROUNDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid background: '{background}'. Options: {list(config.BACKGROUNDS.keys())}",
        )
    return background


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": models.models_status(),
    }


@app.post("/remove-background")
def remove_background(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    background: str = Form(default=config.DEFAULT_BACKGROUND),
):
    """Automatic background removal using IS-Net.

    Returns JPEG with solid background applied.
    """
    background = _validate_background(background)
    img = _read_image(image)

    rgba = models.auto_remove(img)
    jpeg_bytes = models.apply_background(rgba, background=background)

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=removed_bg.jpg"},
    )


@app.post("/remove-background/refine")
def remove_background_refine(
    image: UploadFile = File(..., description="Original image file"),
    points: str = Form(
        ...,
        description='JSON array: [{"x": 100, "y": 200, "label": 0}, ...]. '
        "label=0 means background (remove), label=1 means foreground (keep).",
    ),
    background: str = Form(default=config.DEFAULT_BACKGROUND),
):
    """Interactive background removal refinement using SAM ViT-L.

    The user provides point prompts indicating areas that should be
    background (label=0) or foreground (label=1). SAM generates a
    precise segmentation mask based on these hints.

    Typical use: automatic removal missed the hole inside a tire —
    user clicks on the center, sends that point as label=0.
    """
    background = _validate_background(background)
    img = _read_image(image)

    # Parse and validate points
    try:
        point_list = json.loads(points)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'points' field")

    if not isinstance(point_list, list) or len(point_list) == 0:
        raise HTTPException(status_code=400, detail="'points' must be a non-empty array")

    if len(point_list) > config.MAX_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many points: {len(point_list)} (max {config.MAX_POINTS})",
        )

    for i, p in enumerate(point_list):
        if not isinstance(p, dict) or "x" not in p or "y" not in p or "label" not in p:
            raise HTTPException(
                status_code=400,
                detail=f"Point #{i} must have 'x', 'y', and 'label' fields",
            )
        if p["label"] not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail=f"Point #{i} label must be 0 (background) or 1 (foreground)",
            )

    rgba = models.refine_with_sam(img, point_list)
    jpeg_bytes = models.apply_background(rgba, background=background)

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=refined_bg.jpg"},
    )
