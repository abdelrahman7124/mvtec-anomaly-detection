"""Render every figure and write the summary tables used by the deck."""

import _bootstrap  # noqa: F401
import json

import pandas as pd

from mvtec_eda import plots
from mvtec_eda.config import TABLES_DIR, ensure_output_dirs
from mvtec_eda.summaries import category_summary, defect_type_summary, headline_facts


def main() -> None:
    ensure_output_dirs()
    manifest = pd.read_csv(TABLES_DIR / "manifest.csv")
    masks = pd.read_csv(TABLES_DIR / "mask_stats.csv")
    images = pd.read_csv(TABLES_DIR / "image_stats.csv")

    print("Writing summary tables...")
    category_summary(manifest, masks).to_csv(TABLES_DIR / "category_summary.csv")
    defect_type_summary(manifest, masks).to_csv(TABLES_DIR / "defect_type_summary.csv")
    facts = headline_facts(manifest, masks, images)
    (TABLES_DIR / "headline_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    print("Rendering figures...")
    plots.apply_style()
    plots.fig_split_overview(manifest)
    plots.fig_composition(manifest)
    plots.fig_defect_types(manifest)
    plots.fig_image_properties(manifest)
    plots.fig_normal_samples(manifest)
    plots.fig_defect_examples(manifest)
    plots.fig_defect_examples_wide(manifest)
    plots.fig_defect_area(masks)
    plots.fig_defect_area_by_category(masks, manifest)
    plots.fig_appearance(images, manifest)
    plots.fig_defect_location(masks, manifest)
    plots.fig_train_test_shift(images, manifest)
    print("Done.")


if __name__ == "__main__":
    main()
