"""Figures for the MVTec AD project.

Design rules, applied to every figure here:

* **one panel, one message.** No figure carries two charts side by side - if two
  things need saying, they are two figures.
* **large type.** These are read from the back of a room, not on a laptop.
* **at most one annotation.** A median line or a reference band, never a cloud of
  callouts; the supporting numbers belong in the speaker's takeaway line.
* **colour by role, not by rank.** Blue is always "normal", orange always
  "anomalous", teal "texture", violet "object" - across the whole deck.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

from .config import FIGURES_DIR, MASK_THRESHOLD, get_data_root

# --- Design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

C_NORMAL = BLUE
C_ANOMALOUS = ORANGE
C_TEXTURE = AQUA
C_OBJECT = VIOLET

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 190,
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 14,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 16,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.frameon": False,
            "legend.fontsize": 13,
        }
    )


def _despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")
    return path


def _type_colours(index, ctype) -> list[str]:
    return [C_TEXTURE if ctype[c] == "texture" else C_OBJECT for c in index]


def _type_legend(ax, loc="lower right") -> None:
    ax.legend(
        handles=[Patch(facecolor=C_TEXTURE, label="Texture"),
                 Patch(facecolor=C_OBJECT, label="Object")],
        loc=loc,
    )


# --- 1 · Where the images live --------------------------------------------
def fig_composition(manifest: pd.DataFrame) -> Path:
    """Three bars. The whole point is that the first one has no orange in it."""
    test = manifest[manifest.split == "test"]
    parts = [
        ("Train\n(normal)", int((manifest.split == "train").sum()), C_NORMAL),
        ("Test\n(normal)", int((test.is_anomalous == 0).sum()), AQUA),
        ("Test\n(anomalous)", int((test.is_anomalous == 1).sum()), C_ANOMALOUS),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    xs = np.arange(3)
    values = [p[1] for p in parts]
    ax.bar(xs, values, color=[p[2] for p in parts], width=0.58)
    for x, v in zip(xs, values):
        ax.text(x, v + 70, f"{v:,}", ha="center", fontsize=17, fontweight="bold", color=INK)

    ax.set_xticks(xs, [p[0] for p in parts])
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Images")
    ax.set_title("No anomalies exist in the training set")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig02_composition.png")


# --- 2 · Per-category split ------------------------------------------------
def fig_split_overview(manifest: pd.DataFrame) -> Path:
    test = manifest[manifest.split == "test"]
    tab = pd.DataFrame(
        {
            "train": manifest[manifest.split == "train"].groupby("category").size(),
            "test_normal": test[test.is_anomalous == 0].groupby("category").size(),
            "test_anomalous": test[test.is_anomalous == 1].groupby("category").size(),
        }
    ).fillna(0)
    tab = tab.sort_values(tab.columns.tolist(), ascending=True)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(tab))
    left = np.zeros(len(tab))
    for col, colour, label in (
        ("train", C_NORMAL, "Train (normal)"),
        ("test_normal", AQUA, "Test (normal)"),
        ("test_anomalous", C_ANOMALOUS, "Test (anomalous)"),
    ):
        ax.barh(y, tab[col], left=left, color=colour, label=label, height=0.7,
                edgecolor=SURFACE, linewidth=2)
        left += tab[col].to_numpy()

    ax.set_yticks(y, tab.index)
    ax.set_xlabel("Images")
    ax.set_title("Every category is split the same way")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig01_split_overview.png")


# --- 3 · Defect types are tiny --------------------------------------------
def fig_defect_types(manifest: pd.DataFrame) -> Path:
    anom = manifest[(manifest.split == "test") & (manifest.is_anomalous == 1)]
    per_defect = anom.groupby(["category", "defect_type"]).size()

    fig, ax = plt.subplots(figsize=(10, 6))
    counts, _, _ = ax.hist(per_defect.values, bins=np.arange(5, 34, 2),
                           color=C_ANOMALOUS, edgecolor=SURFACE, linewidth=1.6)
    med = float(np.median(per_defect.values))
    ax.axvline(med, color=INK, lw=2.2, ls="--")
    ax.set_ylim(0, counts.max() * 1.26)
    ax.text(med + 0.8, counts.max() * 1.13, f"median {med:.0f} images",
            fontsize=15, fontweight="bold", color=INK)

    ax.set_xlabel("Images available for one defect type")
    ax.set_ylabel("Number of defect types")
    ax.set_title(f"All {len(per_defect)} defect types are small")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig03_defect_types.png")


# --- 4 · Resolution and channels ------------------------------------------
def fig_image_properties(manifest: pd.DataFrame) -> Path:
    per_cat = manifest.groupby("category").agg(
        width=("width", "first"), mode=("pil_mode", "first")
    ).sort_values("width")

    fig, ax = plt.subplots(figsize=(10.5, 7))
    grey = per_cat["mode"] == "L"
    ax.barh(np.arange(len(per_cat)), per_cat.width,
            color=[INK_MUTED if g else C_NORMAL for g in grey], height=0.7)
    for i, (w, g) in enumerate(zip(per_cat.width, grey)):
        note = f"{w} px" + ("  ·  greyscale" if g else "")
        ax.text(w + 14, i, note, va="center", fontsize=13,
                color=INK_MUTED if g else INK_SECONDARY,
                fontweight="bold" if g else "normal")

    ax.set_yticks(np.arange(len(per_cat)), per_cat.index)
    ax.set_xlabel("Image side length (pixels) — every image is square")
    ax.set_xlim(0, per_cat.width.max() * 1.34)
    ax.set_title("Six resolutions, and three greyscale categories")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig04_image_properties.png")


# --- 5 · How big is a defect? ---------------------------------------------
def fig_defect_area(masks: pd.DataFrame) -> Path:
    area = masks.defect_area_pct.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 40)
    counts, _, _ = ax.hist(area, bins=bins, color=C_ANOMALOUS,
                           edgecolor=SURFACE, linewidth=0.7)
    med = float(np.median(area))
    ax.axvline(med, color=INK, lw=2.2, ls="--")
    ax.set_ylim(0, counts.max() * 1.24)
    ax.text(med * 1.18, counts.max() * 1.11, f"median {med:.1f}%",
            fontsize=15, fontweight="bold", color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("Share of the image covered by the defect (%)")
    ax.set_ylabel("Defective images")
    ax.set_title("Most defects are tiny")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig07_defect_area.png")


# --- 6 · Defect size per category -----------------------------------------
def fig_defect_area_by_category(masks: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    lut = manifest.dropna(subset=["mask_path"]).set_index("mask_path")[["category", "category_type"]]
    df = masks.join(lut, on="mask_path")
    med = df.groupby("category").defect_area_pct.median().sort_values()
    ctype = df.groupby("category").category_type.first()

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(np.arange(len(med)), med.values, color=_type_colours(med.index, ctype), height=0.7)
    for i, v in enumerate(med.values):
        ax.text(v + med.max() * 0.015, i, f"{v:.1f}%", va="center", fontsize=13,
                color=INK_SECONDARY)

    # Linear, not log: the values are labelled directly, and log tick labels
    # ("10^0") cost the audience more than the compression saves.
    ax.set_yticks(np.arange(len(med)), med.index)
    ax.set_xlabel("Median defect size (% of image)")
    ax.set_xlim(0, med.max() * 1.22)
    ratio = med.max() / med.min()
    ax.set_title(f"Defect size varies {ratio:.0f}x between categories")
    ax.grid(axis="y", visible=False)
    _type_legend(ax)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig08_defect_area_by_category.png")


# --- 7 · Brightness --------------------------------------------------------
def fig_appearance(images: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    df = images.merge(manifest[["image_path", "category", "category_type"]], on="image_path")
    med = df.groupby("category").mean_intensity.median().sort_values()
    ctype = df.groupby("category").category_type.first()

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(np.arange(len(med)), med.values, color=_type_colours(med.index, ctype), height=0.7)
    for i, v in enumerate(med.values):
        ax.text(v + 3, i, f"{v:.0f}", va="center", fontsize=13, color=INK_SECONDARY)

    ax.set_yticks(np.arange(len(med)), med.index)
    ax.set_xlabel("Median pixel brightness (0–255)")
    ax.set_xlim(0, 255)
    ax.set_title("Categories look nothing like each other")
    ax.grid(axis="y", visible=False)
    _type_legend(ax, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig09_appearance.png")


# --- 8 · Where defects sit -------------------------------------------------
def fig_defect_location(masks: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    df = masks.dropna(subset=["centroid_x_norm", "centroid_y_norm"])

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    h = ax.hist2d(df.centroid_x_norm, df.centroid_y_norm, bins=20,
                  range=[[0, 1], [0, 1]], cmap=cmap)
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Vertical position")
    ax.set_title("Defects cluster in the middle")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label="defects")
    fig.tight_layout()
    return _save(fig, "fig10_defect_location.png")


# --- 9 · Train vs test drift ----------------------------------------------
def fig_train_test_shift(images: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    df = images.merge(manifest[["image_path", "category", "split", "is_anomalous"]],
                      on="image_path")
    train = df[df.split == "train"].groupby("category").mean_intensity.mean()
    test_good = (df[(df.split == "test") & (df.is_anomalous == 0)]
                 .groupby("category").mean_intensity.mean())
    delta = (test_good - train).sort_values()
    tol = 5.0

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.axvspan(-tol, tol, color=GRID, zorder=0)
    over = delta.abs() > tol
    ax.barh(np.arange(len(delta)), delta.values, height=0.7, zorder=2,
            color=[RED if o else C_NORMAL for o in over])
    ax.axvline(0, color=INK, lw=1.6, zorder=3)
    for i, (v, o) in enumerate(zip(delta.values, over)):
        if o:
            ax.text(v + (1.2 if v > 0 else -1.2), i, f"{v:+.0f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=13,
                    color=RED, fontweight="bold")

    span = float(delta.abs().max())
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_yticks(np.arange(len(delta)), delta.index)
    ax.set_xlabel("Test-normal brightness minus train brightness")
    ax.set_title(f"{int((~over).sum())} of 15 categories match their training set")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig11_train_test_shift.png")


# --- Image-based figures ---------------------------------------------------
def _open_display(rel_path: str, root: Path, size: int = 320) -> np.ndarray:
    with Image.open(root / rel_path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


def _open_mask(rel_path: str, root: Path, size: int = 320) -> np.ndarray:
    with Image.open(root / rel_path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.NEAREST)) > MASK_THRESHOLD


def _bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)


def fig_normal_samples(manifest: pd.DataFrame, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else get_data_root()
    train = manifest[manifest.split == "train"]
    rng = np.random.default_rng(7)

    cats = sorted(train.category.unique())
    fig, axes = plt.subplots(3, 5, figsize=(14, 8.8))
    for ax, cat in zip(axes.ravel(), cats):
        rows = train[train.category == cat]
        pick = rows.iloc[int(rng.integers(len(rows)))]
        ax.imshow(_open_display(pick.image_path, root))
        colour = C_TEXTURE if pick.category_type == "texture" else C_OBJECT
        ax.set_title(cat, fontsize=15, color=colour, pad=7, fontweight="bold")
        _bare(ax)

    fig.suptitle("Five textures and ten objects", y=0.995, fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig05_normal_samples.png")


DEFAULT_DEFECT_PICKS = [
    ("screw", "scratch_neck"),
    ("carpet", "hole"),
    ("bottle", "broken_large"),
    ("transistor", "misplaced"),
    ("leather", "color"),
]


def _defect_triplet(manifest: pd.DataFrame, root: Path, cat: str, defect: str):
    good = manifest[(manifest.category == cat) & (manifest.split == "train")].iloc[0]
    bad_rows = manifest[(manifest.category == cat) & (manifest.defect_type == defect)]
    bad = bad_rows.iloc[len(bad_rows) // 2]

    img = _open_display(bad.image_path, root)
    mask = _open_mask(bad.mask_path, root)
    overlay = img.copy().astype(np.float32)
    overlay[mask] = 0.4 * overlay[mask] + 0.6 * np.array([227, 73, 72], dtype=np.float32)
    return (_open_display(good.image_path, root), img, overlay.astype(np.uint8),
            100.0 * mask.mean())


def fig_defect_examples(manifest: pd.DataFrame, data_root: Path | None = None,
                        picks: list[tuple[str, str]] | None = None) -> Path:
    """Portrait layout - one category per row. Used in the notebook."""
    root = Path(data_root) if data_root is not None else get_data_root()
    picks = picks or DEFAULT_DEFECT_PICKS

    fig, axes = plt.subplots(len(picks), 3, figsize=(9.6, 3.15 * len(picks)))
    for r, (cat, defect) in enumerate(picks):
        good, bad, overlay, area = _defect_triplet(manifest, root, cat, defect)
        for c, (arr, label, colour) in enumerate([
            (good, "normal", INK_SECONDARY),
            (bad, defect.replace("_", " "), C_ANOMALOUS),
            (overlay, f"{area:.1f}% of pixels", RED),
        ]):
            ax = axes[r, c]
            ax.imshow(arr)
            ax.set_title(label, fontsize=13, color=colour, pad=6)
            _bare(ax)
            if c == 0:
                ax.set_ylabel(cat, fontsize=15, fontweight="bold", color=INK, labelpad=10)

    fig.suptitle("What the model has to catch", y=0.995, fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    return _save(fig, "fig06_defect_examples.png")


def fig_defect_examples_wide(manifest: pd.DataFrame, data_root: Path | None = None,
                             picks: list[tuple[str, str]] | None = None) -> Path:
    """Landscape twin of :func:`fig_defect_examples`, for 16:9 slides."""
    root = Path(data_root) if data_root is not None else get_data_root()
    picks = picks or DEFAULT_DEFECT_PICKS
    row_labels = ["normal", "defective", "ground truth"]

    fig, axes = plt.subplots(3, len(picks), figsize=(3.05 * len(picks), 9.9))
    for c, (cat, defect) in enumerate(picks):
        good, bad, overlay, area = _defect_triplet(manifest, root, cat, defect)
        for r, (arr, note, colour) in enumerate([
            (good, None, None),
            (bad, defect.replace("_", " "), C_ANOMALOUS),
            (overlay, f"{area:.1f}% of pixels", RED),
        ]):
            ax = axes[r, c]
            ax.imshow(arr)
            _bare(ax)
            if r == 0:
                ax.set_title(cat, fontsize=17, fontweight="bold", color=INK, pad=9)
            if note:
                ax.set_xlabel(note, fontsize=13, color=colour, labelpad=6)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=15, fontweight="bold",
                              color=INK_SECONDARY, labelpad=13)

    fig.suptitle("What the model has to catch", y=0.99, fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    return _save(fig, "fig06b_defect_examples_wide.png")


# --- Baseline results ------------------------------------------------------
def fig_baseline_comparison(results: pd.DataFrame, metric: str = "image_auroc") -> Path:
    """Autoencoder vs PaDiM, one grouped bar per category."""
    ae_col, pd_col = f"ae_{metric}", f"padim_{metric}"
    df = results.dropna(subset=[ae_col, pd_col]).sort_values(pd_col)
    label = "image-level" if metric == "image_auroc" else "pixel-level"

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df[ae_col], height=0.38, color=C_ANOMALOUS, label="Autoencoder")
    ax.barh(y + 0.2, df[pd_col], height=0.38, color=C_NORMAL, label="PaDiM")
    ax.axvline(0.5, color=INK_MUTED, lw=1.6, ls="--")
    ax.text(0.505, len(df) - 0.45, "random guessing", fontsize=12.5,
            color=INK_MUTED, style="italic")

    ax.set_yticks(y, df.category)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel(f"{label} AUROC")
    ax.set_title(f"PaDiM beats the autoencoder in every category")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    _despine(ax)
    fig.tight_layout()
    return _save(fig, f"fig12_baseline_{metric}.png")


def fig_baseline_summary(results: pd.DataFrame) -> Path:
    """Four bars: the mean of each metric for each model."""
    pairs = [
        ("Autoencoder\nimage", results.ae_image_auroc.mean(), C_ANOMALOUS),
        ("PaDiM\nimage", results.padim_image_auroc.mean(), C_NORMAL),
        ("Autoencoder\npixel", results.ae_pixel_auroc.mean(), C_ANOMALOUS),
        ("PaDiM\npixel", results.padim_pixel_auroc.mean(), C_NORMAL),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    xs = np.arange(4)
    ax.bar(xs, [p[1] for p in pairs], color=[p[2] for p in pairs], width=0.6)
    for x, (_, v, _c) in zip(xs, pairs):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=17,
                fontweight="bold", color=INK)
    ax.axhline(0.5, color=INK_MUTED, lw=1.6, ls="--")
    ax.text(3.42, 0.515, "random", fontsize=12.5, color=INK_MUTED,
            style="italic", ha="right")

    ax.set_xticks(xs, [p[0] for p in pairs])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean AUROC across 15 categories")
    ax.set_title("Pretrained features beat a trained autoencoder")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig13_baseline_summary.png")


def fig_qualitative_maps(npz_path: Path, manifest: pd.DataFrame,
                         data_root: Path | None = None,
                         categories: list[str] | None = None) -> Path:
    """Image, ground truth, autoencoder map and PaDiM map, side by side."""
    root = Path(data_root) if data_root is not None else get_data_root()
    store = np.load(npz_path)
    categories = categories or ["screw", "carpet", "bottle"]
    categories = [c for c in categories if f"{c}__padim_maps" in store]

    fig, axes = plt.subplots(len(categories), 4, figsize=(13, 3.3 * len(categories)))
    axes = np.atleast_2d(axes)
    col_titles = ["input", "ground truth", "autoencoder", "PaDiM"]

    for r, cat in enumerate(categories):
        labels = store[f"{cat}__labels"]
        masks = store[f"{cat}__masks"]
        ae_maps = store[f"{cat}__ae_maps"]
        pd_maps = store[f"{cat}__padim_maps"]

        # Pick the defective test image with the largest ground-truth region.
        anom = np.flatnonzero(labels == 1)
        pick = anom[np.argmax(masks[anom].sum(axis=(1, 2)))]

        test_rows = manifest[(manifest.category == cat) & (manifest.split == "test")]
        test_rows = test_rows.sort_values(["defect_type", "image_path"]).reset_index(drop=True)
        img = _open_display(test_rows.iloc[pick].image_path, root, size=320)

        panels = [
            (img, None, False),
            (masks[pick], "gray", False),
            (ae_maps[pick], "inferno", True),
            (pd_maps[pick], "inferno", True),
        ]
        for c, (arr, cmap, stretch) in enumerate(panels):
            ax = axes[r, c]
            if stretch:
                # A few extreme pixels otherwise crush the whole map to black,
                # which flatters neither model. Clip to the 1st-99th percentile.
                lo, hi = np.percentile(arr, [1, 99])
                ax.imshow(arr, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-6))
            elif cmap:
                ax.imshow(arr, cmap=cmap)
            else:
                ax.imshow(arr)
            _bare(ax)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=16, fontweight="bold", pad=9)
            if c == 0:
                ax.set_ylabel(cat, fontsize=15, fontweight="bold", labelpad=12)

    fig.suptitle("Where each model thinks the defect is", y=0.995,
                 fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, "fig14_qualitative_maps.png")
