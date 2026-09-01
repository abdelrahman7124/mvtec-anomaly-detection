"""Render the baseline-result figures used in the model half of the deck."""

import _bootstrap  # noqa: F401

import pandas as pd

from mvtec_eda import plots
from mvtec_eda.config import TABLES_DIR, ensure_output_dirs
from mvtec_eda.data import load_manifest

RESULTS_CSV = TABLES_DIR / "baseline_results.csv"
QUALITATIVE_NPZ = TABLES_DIR / "qualitative_maps.npz"


def main() -> None:
    ensure_output_dirs()
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found - run `python scripts/05_run_baselines.py` first."
        )

    results = pd.read_csv(RESULTS_CSV).dropna(
        subset=["ae_image_auroc", "padim_image_auroc", "ae_pixel_auroc", "padim_pixel_auroc"]
    )
    print(f"{len(results)} categories with complete results")

    plots.apply_style()
    plots.fig_baseline_comparison(results, metric="image_auroc")
    plots.fig_baseline_comparison(results, metric="pixel_auroc")
    plots.fig_baseline_summary(results)

    if QUALITATIVE_NPZ.exists():
        plots.fig_qualitative_maps(QUALITATIVE_NPZ, load_manifest())
    else:
        print("  (no qualitative_maps.npz - skipping the heatmap figure)")

    print("\nMean AUROC across categories:")
    print(
        results[["ae_image_auroc", "padim_image_auroc",
                 "ae_pixel_auroc", "padim_pixel_auroc"]].mean().round(3).to_string()
    )


if __name__ == "__main__":
    main()
