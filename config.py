"""
Background Removal Service - Configuration
"""

import os
from pathlib import Path

# --- Server ---
HOST = os.getenv("BG_REMOVAL_HOST", "0.0.0.0")
PORT = int(os.getenv("BG_REMOVAL_PORT", "8002"))

# --- Device ---
DEVICE = os.getenv("BG_REMOVAL_DEVICE", "cuda")

# --- Models ---
REMBG_MODEL = "isnet-general-use"

SAM_MODEL_TYPE = "vit_l"
SAM_CHECKPOINT = os.getenv(
    "SAM_CHECKPOINT",
    str(Path.home() / "models" / "sam_vit_l_0b3195.pth"),
)

# --- Output ---
JPEG_QUALITY = 95
DEFAULT_BACKGROUND = "white"

BACKGROUNDS = {
    "white": (255, 255, 255),
    "gray": (224, 224, 224),
}

# --- Limits ---
MAX_IMAGE_SIZE_MB = 20
MAX_POINTS = 50
