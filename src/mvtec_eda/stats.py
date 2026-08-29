"""Pixel-level and mask-level statistics.

Two questions drive this module:

1. *How do the images themselves differ across categories?*  Brightness,
   contrast and saturation decide how much normalisation a model needs and
   whether colour carries signal at all.
2. *How large are the defects?*  The ground-truth masks answer this, and the
   answer constrains input resolution: a defect covering 0.1% of a 1024x1024
   image survives a resize to 224x224 as roughly 5 pixels.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import MASK_THRESHOLD, STATS_IMAGE_SIZE, get_data_root

# Channel spread below this (0-255 scale) means R==G==B for practical purposes:
# the image is greyscale content stored in an RGB container.
GREYSCALE_TOLERANCE = 1.0


def _load_rgb_array(path: Path, size: int = STATS_IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def image_stats(args: tuple[str, str]) -> dict:
    """Summarise one image. Takes/returns plain types so it can be pickled."""
    root_str, rel_path = args
    arr = _load_rgb_array(Path(root_str) / rel_path)
    grey = arr.mean(axis=2)

    # Mean absolute deviation between channels: 0 for a true greyscale image.
    channel_means = arr.mean(axis=(0, 1))
    channel_spread = float(channel_means.max() - channel_means.min())

    # Laplacian variance is the standard cheap proxy for focus / edge density.
    lap = (
        -4 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1]
        + grey[2:, 1:-1]
        + grey[1:-1, :-2]
        + grey[1:-1, 2:]
    )

    return {
        "image_path": rel_path,
        "mean_intensity": float(grey.mean()),
        "std_intensity": float(grey.std()),
        "min_intensity": float(grey.min()),
        "max_intensity": float(grey.max()),
        "mean_r": float(channel_means[0]),
        "mean_g": float(channel_means[1]),
        "mean_b": float(channel_means[2]),
        "channel_spread": channel_spread,
        "is_effectively_greyscale": int(channel_spread < GREYSCALE_TOLERANCE),
        "laplacian_var": float(lap.var()),
    }


def mask_stats(args: tuple[str, str]) -> dict:
    """Defect geometry for one ground-truth mask, at native resolution."""
    root_str, rel_path = args
    with Image.open(Path(root_str) / rel_path) as im:
        mask = np.asarray(im.convert("L")) > MASK_THRESHOLD

    total = mask.size
    defect_px = int(mask.sum())
    ys, xs = np.nonzero(mask)

    if defect_px:
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        centroid_y = float(ys.mean() / mask.shape[0])
        centroid_x = float(xs.mean() / mask.shape[1])
    else:
        bbox_h = bbox_w = 0
        centroid_y = centroid_x = float("nan")

    return {
        "mask_path": rel_path,
        "mask_height": int(mask.shape[0]),
        "mask_width": int(mask.shape[1]),
        "defect_pixels": defect_px,
        "defect_area_ratio": defect_px / total,
        "defect_area_pct": 100.0 * defect_px / total,
        "n_defect_regions": _count_regions(mask),
        "bbox_height": bbox_h,
        "bbox_width": bbox_w,
        "bbox_area_ratio": (bbox_h * bbox_w) / total,
        "centroid_y_norm": centroid_y,
        "centroid_x_norm": centroid_x,
    }


def _count_regions(mask: np.ndarray) -> int:
    """Number of connected defect blobs (8-connectivity)."""
    try:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=bool)
        _, count = ndimage.label(mask, structure=structure)
        return int(count)
    except ImportError:  # pragma: no cover - scipy is in requirements
        from skimage.measure import label

        return int(label(mask, connectivity=2).max())


def _run_pool(fn, items: list, workers: int | None, desc: str) -> list[dict]:
    results: list[dict] = []
    total = len(items)
    if workers == 1:
        for i, item in enumerate(items, 1):
            results.append(fn(item))
            if i % 500 == 0 or i == total:
                print(f"  {desc}: {i}/{total}")
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fn, items, chunksize=16), 1):
            results.append(res)
            if i % 500 == 0 or i == total:
                print(f"  {desc}: {i}/{total}")
    return results


def compute_image_stats(
    manifest: pd.DataFrame, data_root: Path | None = None, workers: int | None = None
) -> pd.DataFrame:
    root = str(data_root or get_data_root())
    items = [(root, p) for p in manifest.image_path.tolist()]
    return pd.DataFrame.from_records(_run_pool(image_stats, items, workers, "image stats"))


def compute_mask_stats(
    manifest: pd.DataFrame, data_root: Path | None = None, workers: int | None = None
) -> pd.DataFrame:
    root = str(data_root or get_data_root())
    paths = manifest.loc[manifest.has_mask == 1, "mask_path"].tolist()
    return pd.DataFrame.from_records(_run_pool(mask_stats, [(root, p) for p in paths], workers, "mask stats"))
