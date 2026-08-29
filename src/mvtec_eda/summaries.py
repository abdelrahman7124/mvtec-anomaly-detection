"""Aggregate tables shared by the notebook, the slide deck and the README."""

from __future__ import annotations

import numpy as np
import pandas as pd


def category_summary(manifest: pd.DataFrame, masks: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per category: split sizes, defect variety and typical defect size."""
    test = manifest[manifest.split == "test"]
    out = pd.DataFrame(
        {
            "type": manifest.groupby("category").category_type.first(),
            "resolution": manifest.groupby("category").resolution.first(),
            "channels": manifest.groupby("category").pil_mode.first().map({"L": 1, "RGB": 3}),
            "train": manifest[manifest.split == "train"].groupby("category").size(),
            "test_normal": test[test.is_anomalous == 0].groupby("category").size(),
            "test_anomalous": test[test.is_anomalous == 1].groupby("category").size(),
            "defect_types": test[test.is_anomalous == 1].groupby("category").defect_type.nunique(),
        }
    )
    out["total"] = out.train + out.test_normal + out.test_anomalous
    out["pct_anomalous_in_test"] = (
        100 * out.test_anomalous / (out.test_normal + out.test_anomalous)
    ).round(1)

    if masks is not None:
        lut = manifest.dropna(subset=["mask_path"]).set_index("mask_path")[["category"]]
        joined = masks.join(lut, on="mask_path")
        grp = joined.groupby("category").defect_area_pct
        out["median_defect_area_pct"] = grp.median().round(3)
        out["min_defect_area_pct"] = grp.min().round(3)
        out["max_defect_area_pct"] = grp.max().round(3)

    return out.sort_values("total", ascending=False)


def defect_type_summary(manifest: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """One row per (category, defect type) with image counts and defect size."""
    anom = manifest[(manifest.split == "test") & (manifest.is_anomalous == 1)]
    lut = anom.set_index("mask_path")[["category", "defect_type"]]
    joined = masks.join(lut, on="mask_path")
    out = joined.groupby(["category", "defect_type"]).agg(
        n_images=("mask_path", "size"),
        median_area_pct=("defect_area_pct", "median"),
        mean_regions=("n_defect_regions", "mean"),
    )
    return out.round(3).sort_values(["category", "n_images"], ascending=[True, False])


def headline_facts(manifest: pd.DataFrame, masks: pd.DataFrame, images: pd.DataFrame) -> dict:
    """The numbers quoted in the deck, computed in one place so they cannot drift."""
    test = manifest[manifest.split == "test"]
    anom_test = test[test.is_anomalous == 1]
    area = masks.defect_area_pct

    df = images.merge(manifest[["image_path", "category", "split", "is_anomalous"]], on="image_path")
    train_mean = df[df.split == "train"].groupby("category").mean_intensity.mean()
    testgood_mean = df[(df.split == "test") & (df.is_anomalous == 0)].groupby("category").mean_intensity.mean()
    drift = (testgood_mean - train_mean).abs()

    grey_cats = sorted(manifest.loc[manifest.pil_mode == "L", "category"].unique().tolist())
    per_defect = anom_test.groupby(["category", "defect_type"]).size()

    return {
        "n_images": len(manifest),
        "n_categories": manifest.category.nunique(),
        "n_textures": int((manifest.groupby("category").category_type.first() == "texture").sum()),
        "n_objects": int((manifest.groupby("category").category_type.first() == "object").sum()),
        "n_train": int((manifest.split == "train").sum()),
        "n_test": len(test),
        "n_test_normal": int((test.is_anomalous == 0).sum()),
        "n_test_anomalous": len(anom_test),
        "pct_test_anomalous": round(100 * len(anom_test) / len(test), 1),
        "n_train_anomalies": int(manifest[manifest.split == "train"].is_anomalous.sum()),
        "n_defect_types": int(anom_test.groupby(["category", "defect_type"]).ngroups),
        "median_images_per_defect_type": int(np.median(per_defect.values)),
        "min_images_per_defect_type": int(per_defect.min()),
        "n_masks": int(manifest.has_mask.sum()),
        "median_defect_area_pct": round(float(area.median()), 2),
        "pct_defects_under_1pct": round(float(100 * (area < 1).mean()), 1),
        "pct_defects_under_5pct": round(float(100 * (area < 5).mean()), 1),
        "smallest_defect_pct": round(float(area.min()), 3),
        "largest_defect_pct": round(float(area.max()), 1),
        "pct_single_region_masks": round(float(100 * (masks.n_defect_regions == 1).mean()), 1),
        "max_defect_regions": int(masks.n_defect_regions.max()),
        "n_resolutions": manifest.resolution.nunique(),
        "resolutions": sorted(manifest.resolution.unique().tolist()),
        "greyscale_categories": grey_cats,
        "n_greyscale_images": int((manifest.pil_mode == "L").sum()),
        "n_categories_within_5_levels": int((drift <= 5).sum()),
        "max_intensity_drift": round(float(drift.max()), 1),
        "smallest_category": manifest.category.value_counts().idxmin(),
        "smallest_category_train": int(
            manifest[(manifest.split == "train")].category.value_counts().min()
        ),
    }
