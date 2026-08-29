"""Project paths and shared constants.

The dataset itself is never committed (~4.9 GB). Point ``MVTEC_ROOT`` at a local
copy either by editing ``DEFAULT_DATA_ROOT`` or by exporting the environment
variable ``MVTEC_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
PRESENTATION_DIR = REPORTS_DIR / "presentation"

DEFAULT_DATA_ROOT = Path.home() / "Downloads" / "mvtec_anomaly_detection"


def get_data_root() -> Path:
    """Resolve the dataset root, preferring the MVTEC_ROOT env var."""
    root = Path(os.environ.get("MVTEC_ROOT", DEFAULT_DATA_ROOT)).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"MVTec dataset not found at {root!s}. Download it from "
            "https://www.mvtec.com/company/research/datasets/mvtec-ad and either "
            "place it there or set the MVTEC_ROOT environment variable."
        )
    return root


def ensure_output_dirs() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, PRESENTATION_DIR):
        d.mkdir(parents=True, exist_ok=True)


# MVTec AD splits its 15 categories into two visually distinct groups. Textures
# are near-stationary surfaces; objects are centred parts with a defined pose.
TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]

OBJECT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

ALL_CATEGORIES = sorted(TEXTURE_CATEGORIES + OBJECT_CATEGORIES)

# Size every image is resampled to before pixel statistics are computed. Full
# resolution is unnecessary for distribution-level questions and ~16x slower.
STATS_IMAGE_SIZE = 256

# A pixel-level mask is stored as 0 / 255; anything above this counts as defect.
MASK_THRESHOLD = 127
