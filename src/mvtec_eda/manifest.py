"""Build a one-row-per-image manifest of the MVTec AD dataset.

Every later analysis reads this table instead of re-walking the filesystem, so
the expensive directory traversal happens exactly once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .config import OBJECT_CATEGORIES, TEXTURE_CATEGORIES, get_data_root

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _category_type(category: str) -> str:
    if category in TEXTURE_CATEGORIES:
        return "texture"
    if category in OBJECT_CATEGORIES:
        return "object"
    return "unknown"


def _mask_path_for(image_path: Path, category_dir: Path, defect_type: str) -> Path | None:
    """Locate the ground-truth mask that pairs with a defective test image.

    MVTec names masks ``<stem>_mask.png`` under ``ground_truth/<defect_type>/``.
    Good test images have no mask by design.
    """
    if defect_type == "good":
        return None
    candidate = category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    return candidate if candidate.exists() else None


def _read_header(path: Path) -> tuple[int, int, str]:
    """Read width/height/mode from the file header without decoding pixels."""
    with Image.open(path) as im:
        return im.width, im.height, im.mode


def build_manifest(data_root: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Walk the dataset and return the image-level manifest."""
    root = Path(data_root) if data_root is not None else get_data_root()
    records: list[dict] = []

    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    for category in categories:
        category_dir = root / category
        for split in ("train", "test"):
            split_dir = category_dir / split
            if not split_dir.is_dir():
                continue
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                images = sorted(
                    p for p in defect_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
                for image_path in images:
                    width, height, mode = _read_header(image_path)
                    mask_path = _mask_path_for(image_path, category_dir, defect_type)
                    records.append(
                        {
                            "category": category,
                            "category_type": _category_type(category),
                            "split": split,
                            "defect_type": defect_type,
                            "label": "normal" if defect_type == "good" else "anomalous",
                            "is_anomalous": int(defect_type != "good"),
                            "image_path": str(image_path.relative_to(root)).replace("\\", "/"),
                            "mask_path": (
                                str(mask_path.relative_to(root)).replace("\\", "/")
                                if mask_path
                                else None
                            ),
                            "has_mask": int(mask_path is not None),
                            "width": width,
                            "height": height,
                            "resolution": f"{width}x{height}",
                            "aspect_ratio": round(width / height, 4),
                            "pil_mode": mode,
                            "file_size_kb": round(image_path.stat().st_size / 1024, 2),
                        }
                    )
        if verbose:
            n = sum(r["category"] == category for r in records)
            print(f"  {category:<12} {n:>5} images")

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["category", "split", "defect_type", "image_path"]).reset_index(drop=True)


def audit_manifest(df: pd.DataFrame) -> dict:
    """Sanity checks that should hold for an intact copy of MVTec AD."""
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    anomalous_test = test[test.is_anomalous == 1]
    return {
        "n_images": len(df),
        "n_categories": df.category.nunique(),
        "n_train": len(train),
        "n_test": len(test),
        "train_defect_types": sorted(train.defect_type.unique().tolist()),
        "train_contains_anomalies": bool(train.is_anomalous.sum()),
        "n_test_good": int((test.is_anomalous == 0).sum()),
        "n_test_anomalous": len(anomalous_test),
        "n_masks_found": int(df.has_mask.sum()),
        "anomalous_test_missing_mask": int((anomalous_test.has_mask == 0).sum()),
        "duplicate_paths": int(df.image_path.duplicated().sum()),
        "n_distinct_resolutions": df.resolution.nunique(),
        "pil_modes": sorted(df.pil_mode.unique().tolist()),
    }
