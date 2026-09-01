"""Data loading for the MVTec AD baselines.

Every choice here traces back to a finding in the week-1 EDA:

* categories have six different resolutions -> resize everything to one size;
* `grid`, `screw` and `zipper` are stored single-channel -> convert to RGB so a
  pretrained backbone accepts them;
* defects are centre-biased because of how the dataset was photographed -> never
  centre-crop, which would bake that bias in;
* the training split is 100% normal -> the train loader never yields a label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import MASK_THRESHOLD, TABLES_DIR, get_data_root

# ImageNet statistics - the backbones used here were pretrained with them.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest() -> pd.DataFrame:
    path = TABLES_DIR / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/01_build_manifest.py` first."
        )
    return pd.read_csv(path)


class MVTecCategory(Dataset):
    """One split of one category.

    Yields ``(image, label, mask)``. ``mask`` is all-zero for normal images, which
    keeps the pixel-level metric simple: normal pixels are genuinely negatives.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        category: str,
        split: str,
        image_size: int,
        data_root: Path | None = None,
        normalise: bool = True,
    ) -> None:
        self.root = Path(data_root) if data_root is not None else get_data_root()
        self.image_size = image_size
        self.normalise = normalise
        rows = manifest[(manifest.category == category) & (manifest.split == split)]
        self.rows = rows.reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"no images for category={category!r} split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        with Image.open(self.root / rel_path) as im:
            # convert("RGB") also expands the single-channel categories.
            im = im.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            arr = np.asarray(im, dtype=np.float32) / 255.0
        if self.normalise:
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_mask(self, rel_path: str | float) -> torch.Tensor:
        size = self.image_size
        if not isinstance(rel_path, str):
            return torch.zeros(1, size, size, dtype=torch.float32)
        with Image.open(self.root / rel_path) as im:
            im = im.convert("L").resize((size, size), Image.NEAREST)
            arr = (np.asarray(im) > MASK_THRESHOLD).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        return (
            self._load_image(row.image_path),
            int(row.is_anomalous),
            self._load_mask(row.mask_path),
        )


def make_loaders(
    manifest: pd.DataFrame,
    category: str,
    image_size: int,
    batch_size: int = 16,
    data_root: Path | None = None,
    normalise: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (normal images only, by construction) and test loader."""
    train_ds = MVTecCategory(manifest, category, "train", image_size, data_root, normalise)
    test_ds = MVTecCategory(manifest, category, "test", image_size, data_root, normalise)
    # num_workers=0: on Windows, worker start-up costs more than it saves here.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
