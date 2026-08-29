"""Compute per-image pixel statistics and per-mask defect geometry."""

import _bootstrap  # noqa: F401
import os

import pandas as pd

from mvtec_eda.config import TABLES_DIR, ensure_output_dirs, get_data_root
from mvtec_eda.stats import compute_image_stats, compute_mask_stats


def main() -> None:
    ensure_output_dirs()
    root = get_data_root()
    manifest = pd.read_csv(TABLES_DIR / "manifest.csv")
    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Using {workers} workers on {len(manifest)} images")

    img = compute_image_stats(manifest, root, workers)
    img.to_csv(TABLES_DIR / "image_stats.csv", index=False)
    print(f"Wrote image_stats.csv ({len(img)} rows)")

    msk = compute_mask_stats(manifest, root, workers)
    msk.to_csv(TABLES_DIR / "mask_stats.csv", index=False)
    print(f"Wrote mask_stats.csv ({len(msk)} rows)")


if __name__ == "__main__":
    main()
