"""Train and evaluate both one-class baselines on every category.

Baseline 1 - convolutional autoencoder trained from scratch on normal images.
Baseline 2 - PaDiM on frozen ImageNet features, no backpropagation.

Results are appended to reports/tables/baseline_results.csv after each category,
so a long run can be interrupted and resumed without losing work.
"""

import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np
import pandas as pd
import torch

from mvtec_eda.config import ALL_CATEGORIES, TABLES_DIR, ensure_output_dirs
from mvtec_eda.data import load_manifest, make_loaders
from mvtec_eda.evaluate import collect_scores, image_auroc, pixel_auroc
from mvtec_eda.models import ConvAutoencoder, PaDiM, train_autoencoder

RESULTS_CSV = TABLES_DIR / "baseline_results.csv"
QUALITATIVE_NPZ = TABLES_DIR / "qualitative_maps.npz"

# Categories kept for the qualitative side-by-side figure in the deck.
QUALITATIVE = ["screw", "carpet", "bottle"]


def run_autoencoder(manifest, category, args, device):
    # Unnormalised [0,1] images: the decoder ends in a sigmoid.
    train_loader, test_loader = make_loaders(
        manifest, category, args.ae_size, args.batch_size, normalise=False
    )
    model = ConvAutoencoder(latent_dim=args.latent_dim)
    train_autoencoder(model, train_loader, epochs=args.epochs, device=device)
    model.eval()

    labels, scores, maps, masks = collect_scores(
        lambda images: model.anomaly_map(images.to(device)).cpu(),
        test_loader,
        out_size=args.ae_size,
    )
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks, maps),
    }, (labels, maps, masks)


def run_padim(manifest, category, args, device):
    train_loader, test_loader = make_loaders(
        manifest, category, args.padim_size, args.batch_size, normalise=True
    )
    model = PaDiM(n_features=args.n_features, device=device)
    model.fit(train_loader)

    labels, scores, maps, masks = collect_scores(
        lambda images: model.anomaly_map(images, out_size=args.padim_size),
        test_loader,
        out_size=args.padim_size,
    )
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks, maps),
    }, (labels, maps, masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ae-size", type=int, default=128)
    parser.add_argument("--padim-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--skip-padim", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = load_manifest()
    categories = args.categories or ALL_CATEGORIES
    print(f"device={device}  categories={len(categories)}  epochs={args.epochs}")

    rows = []
    if RESULTS_CSV.exists():
        rows = pd.read_csv(RESULTS_CSV).to_dict("records")
        done = {r["category"] for r in rows}
        categories = [c for c in categories if c not in done]
        print(f"resuming - {len(done)} categories already done")

    qualitative: dict[str, np.ndarray] = {}
    if QUALITATIVE_NPZ.exists():
        qualitative = dict(np.load(QUALITATIVE_NPZ))

    for category in categories:
        print(f"\n=== {category} ===")
        row = {"category": category}

        if not args.skip_ae:
            t0 = time.time()
            print("  autoencoder")
            metrics, (labels, maps, masks) = run_autoencoder(manifest, category, args, device)
            row.update({f"ae_{k}": v for k, v in metrics.items()})
            row["ae_seconds"] = round(time.time() - t0, 1)
            print(f"    image AUROC {metrics['image_auroc']:.3f}  "
                  f"pixel AUROC {metrics['pixel_auroc']:.3f}  "
                  f"({row['ae_seconds']}s)")
            if category in QUALITATIVE:
                qualitative[f"{category}__ae_maps"] = maps
                qualitative[f"{category}__labels"] = labels
                qualitative[f"{category}__masks"] = masks

        if not args.skip_padim:
            t0 = time.time()
            print("  padim")
            metrics, (labels, maps, masks) = run_padim(manifest, category, args, device)
            row.update({f"padim_{k}": v for k, v in metrics.items()})
            row["padim_seconds"] = round(time.time() - t0, 1)
            print(f"    image AUROC {metrics['image_auroc']:.3f}  "
                  f"pixel AUROC {metrics['pixel_auroc']:.3f}  "
                  f"({row['padim_seconds']}s)")
            if category in QUALITATIVE:
                qualitative[f"{category}__padim_maps"] = maps

        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        if qualitative:
            np.savez_compressed(QUALITATIVE_NPZ, **qualitative)

    df = pd.DataFrame(rows)
    print(f"\nWrote {RESULTS_CSV}")
    cols = [c for c in ("ae_image_auroc", "ae_pixel_auroc",
                        "padim_image_auroc", "padim_pixel_auroc") if c in df]
    if cols:
        print("\nMean across categories:")
        print(df[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
