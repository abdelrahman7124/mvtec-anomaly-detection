"""Presentation-quality figures for the MVTec AD exploratory analysis.

Colour follows a validated categorical palette: hues are assigned in fixed slot
order and never cycled, magnitude uses a single-hue blue ramp, and every figure
with two or more series carries a legend so identity is never colour-alone.
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

# Categorical slots, fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

# Semantic roles used consistently across every figure in the deck.
C_NORMAL = BLUE
C_ANOMALOUS = ORANGE
C_TEXTURE = AQUA
C_OBJECT = VIOLET

# Single-hue sequential ramp (blue, light -> dark).
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
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


def _subtitle(fig, text: str, y: float = 0.945) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=10.5, color=INK_SECONDARY)


# --- Figure 1: how every category is split ---------------------------------
def fig_split_overview(manifest: pd.DataFrame) -> Path:
    test = manifest[manifest.split == "test"]
    tab = pd.DataFrame(
        {
            "train_good": manifest[manifest.split == "train"].groupby("category").size(),
            "test_good": test[test.is_anomalous == 0].groupby("category").size(),
            "test_anomalous": test[test.is_anomalous == 1].groupby("category").size(),
        }
    ).fillna(0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total")

    fig, ax = plt.subplots(figsize=(11, 7.2))
    y = np.arange(len(tab))
    segments = [
        ("train_good", C_NORMAL, "Train — normal only"),
        ("test_good", AQUA, "Test — normal"),
        ("test_anomalous", C_ANOMALOUS, "Test — anomalous"),
    ]
    left = np.zeros(len(tab))
    for col, colour, label in segments:
        ax.barh(
            y,
            tab[col],
            left=left,
            color=colour,
            label=label,
            height=0.68,
            edgecolor=SURFACE,
            linewidth=2,
        )
        left += tab[col].to_numpy()

    for i, total in enumerate(tab.total):
        ax.text(total + 8, i, f"{int(total)}", va="center", fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(y, tab.index)
    ax.set_xlabel("Number of images")
    ax.set_xlim(0, tab.total.max() * 1.1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=1)
    _despine(ax)
    fig.suptitle("Every category follows the same one-class split", y=0.99)
    _subtitle(fig, "Training data contains no anomalies at all — the defining constraint of MVTec AD", 0.945)
    fig.subplots_adjust(top=0.90)
    return _save(fig, "fig01_split_overview.png")


# --- Figure 2: the headline composition ------------------------------------
def fig_composition(manifest: pd.DataFrame) -> Path:
    n_train = int((manifest.split == "train").sum())
    test = manifest[manifest.split == "test"]
    n_test_good = int((test.is_anomalous == 0).sum())
    n_test_anom = int((test.is_anomalous == 1).sum())
    total = len(manifest)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    parts = [("Train\nnormal", n_train, C_NORMAL), ("Test\nnormal", n_test_good, AQUA),
             ("Test\nanomalous", n_test_anom, C_ANOMALOUS)]
    xs = np.arange(3)
    ax.bar(xs, [p[1] for p in parts], color=[p[2] for p in parts], width=0.62)
    for x, (_, v, _c) in zip(xs, parts):
        ax.text(x, v + 60, f"{v:,}\n{100 * v / total:.1f}%", ha="center", fontsize=10.5,
                color=INK, fontweight="bold", linespacing=1.35)
    ax.set_xticks(xs, [p[0] for p in parts])
    ax.set_ylim(0, max(p[1] for p in parts) * 1.28)
    ax.set_ylabel("Images")
    ax.set_title("Where the 5,354 images live")
    ax.grid(axis="x", visible=False)
    _despine(ax)

    ax = axes[1]
    labels = ["Normal", "Anomalous"]
    train_vals = [n_train, 0]
    test_vals = [n_test_good, n_test_anom]
    xs = np.arange(2)
    w = 0.36
    ax.bar(xs - w / 2, train_vals, w, color=C_NORMAL, label="Train")
    ax.bar(xs + w / 2, test_vals, w, color=C_ANOMALOUS, label="Test")
    for x, v in zip(xs - w / 2, train_vals):
        ax.text(x, v + 50, f"{v:,}", ha="center", fontsize=10, color=INK_SECONDARY)
    for x, v in zip(xs + w / 2, test_vals):
        ax.text(x, v + 50, f"{v:,}", ha="center", fontsize=10, color=INK_SECONDARY)
    ax.annotate(
        "zero anomalies\nto train on",
        xy=(1 - w / 2, 30),
        xytext=(0.52, 2250),
        fontsize=10,
        color=RED,
        fontweight="bold",
        ha="center",
        linespacing=1.35,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, connectionstyle="arc3,rad=0.25"),
    )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Images")
    ax.set_ylim(0, n_train * 1.15)
    ax.set_title("Labels available per split")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    _despine(ax)

    fig.suptitle("The dataset is built for one-class learning, not classification", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig02_composition.png")


# --- Figure 3: defect taxonomy ---------------------------------------------
def fig_defect_types(manifest: pd.DataFrame) -> Path:
    anom = manifest[(manifest.split == "test") & (manifest.is_anomalous == 1)]
    per_cat = anom.groupby("category").defect_type.nunique().sort_values()
    ctype = manifest.groupby("category").category_type.first()
    per_defect = anom.groupby(["category", "defect_type"]).size()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    colours = [C_TEXTURE if ctype[c] == "texture" else C_OBJECT for c in per_cat.index]
    ax.barh(np.arange(len(per_cat)), per_cat.values, color=colours, height=0.68)
    for i, v in enumerate(per_cat.values):
        ax.text(v + 0.12, i, str(v), va="center", fontsize=10, color=INK_SECONDARY)
    ax.set_yticks(np.arange(len(per_cat)), per_cat.index)
    ax.set_xlabel("Distinct defect types")
    ax.set_xlim(0, per_cat.max() + 1.1)
    ax.set_title("Defect types defined per category")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[Patch(facecolor=C_TEXTURE, label="Texture"), Patch(facecolor=C_OBJECT, label="Object")],
        loc="lower right",
    )
    _despine(ax)

    ax = axes[1]
    counts, _, _ = ax.hist(per_defect.values, bins=np.arange(5, 34, 2), color=C_ANOMALOUS,
                           edgecolor=SURFACE, linewidth=1.5)
    ax.set_ylim(0, counts.max() * 1.24)
    med = float(np.median(per_defect.values))
    ax.axvline(med, color=INK, lw=1.8, ls="--")
    ax.text(med + 0.7, counts.max() * 1.12, f"median {med:.0f} images", fontsize=10,
            color=INK, fontweight="bold")
    ax.set_xlabel("Images available for one defect type")
    ax.set_ylabel("Number of defect types")
    ax.set_title(f"Each of the {len(per_defect)} defect types is tiny")
    ax.grid(axis="x", visible=False)
    _despine(ax)

    fig.suptitle("73 category-specific defect types, none with enough images to train on", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig03_defect_types.png")


# --- Figure 4: acquisition properties --------------------------------------
def fig_image_properties(manifest: pd.DataFrame) -> Path:
    per_cat = manifest.groupby("category").agg(
        width=("width", "first"), mode=("pil_mode", "first"), ctype=("category_type", "first")
    ).sort_values("width")
    res_counts = manifest.resolution.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.barh(np.arange(len(per_cat)), per_cat.width, color=C_NORMAL, height=0.68)
    for i, (w, m) in enumerate(zip(per_cat.width, per_cat["mode"])):
        tag = "1-channel" if m == "L" else "3-channel"
        ax.text(w + 12, i, f"{w}px · {tag}", va="center", fontsize=9.5, color=INK_SECONDARY)
    ax.set_yticks(np.arange(len(per_cat)), per_cat.index)
    ax.set_xlabel("Image side length (pixels) — all images are square")
    ax.set_xlim(0, per_cat.width.max() * 1.32)
    ax.set_title("One fixed resolution per category")
    ax.grid(axis="y", visible=False)
    _despine(ax)

    ax = axes[1]
    ramp = SEQ_BLUE[1 : 1 + len(res_counts)]
    ax.bar(np.arange(len(res_counts)), res_counts.values, color=ramp, width=0.62)
    for i, v in enumerate(res_counts.values):
        ax.text(i, v + 45, f"{v:,}", ha="center", fontsize=10, color=INK_SECONDARY)
    ax.set_xticks(np.arange(len(res_counts)), [r.split("x")[0] for r in res_counts.index])
    ax.set_xlabel("Side length (pixels)")
    ax.set_ylabel("Images")
    ax.set_ylim(0, res_counts.max() * 1.16)
    ax.set_title("Six distinct resolutions across the dataset")
    ax.grid(axis="x", visible=False)
    _despine(ax)

    fig.suptitle("Images are high-resolution and square, but not uniform across categories", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig04_image_properties.png")


# --- Figure 7: how big is a defect? ----------------------------------------
def fig_defect_area(masks: pd.DataFrame) -> Path:
    area = masks.defect_area_pct.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    ax = axes[0]
    bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 45)
    ax.hist(area, bins=bins, color=C_ANOMALOUS, edgecolor=SURFACE, linewidth=0.8)
    med = float(np.median(area))
    ax.axvline(med, color=INK, lw=1.8, ls="--")
    ax.text(med * 1.15, ax.get_ylim()[1] * 0.9, f"median\n{med:.2f}%", fontsize=10.5,
            color=INK, fontweight="bold", linespacing=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("Share of the image covered by the defect (%, log scale)")
    ax.set_ylabel("Number of defective images")
    ax.set_title("Most defects are very small")
    ax.grid(axis="x", visible=False)
    _despine(ax)

    ax = axes[1]
    xs = np.sort(area)
    ys = 100.0 * np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=C_NORMAL, lw=2.2)
    for thr, colour in ((1.0, RED), (5.0, VIOLET)):
        pct = 100.0 * (area < thr).mean()
        ax.plot([thr, thr], [0, pct], color=colour, lw=1.6, ls=":")
        ax.plot([xs.min(), thr], [pct, pct], color=colour, lw=1.6, ls=":")
        ax.scatter([thr], [pct], s=44, color=colour, zorder=5, edgecolor=SURFACE, linewidth=2)
        ax.text(thr * 1.2, pct - 6, f"{pct:.0f}% of defects\ncover < {thr:.0f}% of the image",
                fontsize=9.5, color=colour, fontweight="bold", linespacing=1.3)
    ax.set_xscale("log")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Defect area (%, log scale)")
    ax.set_ylabel("Cumulative share of defects (%)")
    ax.set_title("Cumulative distribution")
    _despine(ax)

    fig.suptitle("Defect size is the hardest constraint in this dataset", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig07_defect_area.png")


# --- Figure 8: defect size varies by category ------------------------------
def fig_defect_area_by_category(masks: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    lut = manifest.dropna(subset=["mask_path"]).set_index("mask_path")[["category", "category_type"]]
    df = masks.join(lut, on="mask_path")
    order = df.groupby("category").defect_area_pct.median().sort_values().index.tolist()
    ctype = df.groupby("category").category_type.first()

    fig, ax = plt.subplots(figsize=(12, 6.4))
    data = [df.loc[df.category == c, "defect_area_pct"].to_numpy() for c in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color=INK, lw=2))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(C_TEXTURE if ctype[c] == "texture" else C_OBJECT)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(1.5)
        patch.set_alpha(0.92)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_MUTED)

    for i, c in enumerate(order, start=1):
        vals = df.loc[df.category == c, "defect_area_pct"]
        ax.scatter(vals, np.random.default_rng(0).normal(i, 0.075, len(vals)),
                   s=5, color=INK_MUTED, alpha=0.3, zorder=1)

    ax.set_xscale("log")
    ax.set_yticks(np.arange(1, len(order) + 1), order)
    ax.set_xlabel("Defect area as share of image (%, log scale)")
    ax.set_title("Defect size spans three orders of magnitude between categories")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[Patch(facecolor=C_TEXTURE, label="Texture"), Patch(facecolor=C_OBJECT, label="Object")],
        loc="lower right",
    )
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig08_defect_area_by_category.png")


# --- Image-based figures ---------------------------------------------------
def _open_display(rel_path: str, root: Path, size: int = 320) -> np.ndarray:
    with Image.open(root / rel_path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS))


def _open_mask(rel_path: str, root: Path, size: int = 320) -> np.ndarray:
    with Image.open(root / rel_path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.NEAREST)) > MASK_THRESHOLD


def fig_normal_samples(manifest: pd.DataFrame, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else get_data_root()
    train = manifest[manifest.split == "train"]
    rng = np.random.default_rng(7)

    cats = sorted(train.category.unique())
    fig, axes = plt.subplots(3, 5, figsize=(14, 9))
    for ax, cat in zip(axes.ravel(), cats):
        rows = train[train.category == cat]
        pick = rows.iloc[int(rng.integers(len(rows)))]
        ax.imshow(_open_display(pick.image_path, root))
        is_texture = pick.category_type == "texture"
        ax.set_title(cat, fontsize=12, color=C_TEXTURE if is_texture else C_OBJECT, pad=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)

    fig.suptitle("Fifteen categories: five textures and ten objects", y=0.985)
    _subtitle(fig, "One defect-free training image per category — teal titles are textures, violet are objects", 0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    return _save(fig, "fig05_normal_samples.png")


def fig_defect_examples(manifest: pd.DataFrame, data_root: Path | None = None,
                        picks: list[tuple[str, str]] | None = None) -> Path:
    """Normal / defective / ground-truth overlay, one row per chosen category."""
    root = Path(data_root) if data_root is not None else get_data_root()
    picks = picks or [
        ("screw", "scratch_neck"),
        ("carpet", "hole"),
        ("bottle", "broken_large"),
        ("transistor", "misplaced"),
        ("leather", "color"),
    ]

    fig, axes = plt.subplots(len(picks), 3, figsize=(9.6, 3.15 * len(picks)))
    for r, (cat, defect) in enumerate(picks):
        good = manifest[(manifest.category == cat) & (manifest.split == "train")].iloc[0]
        bad_rows = manifest[(manifest.category == cat) & (manifest.defect_type == defect)]
        bad = bad_rows.iloc[len(bad_rows) // 2]

        img = _open_display(bad.image_path, root)
        mask = _open_mask(bad.mask_path, root)
        overlay = img.copy().astype(np.float32)
        tint = np.array([227, 73, 72], dtype=np.float32)  # RED
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * tint

        area_pct = 100.0 * mask.mean()
        panels = [
            (_open_display(good.image_path, root), "normal", INK_SECONDARY),
            (img, f"defect: {defect.replace('_', ' ')}", C_ANOMALOUS),
            (overlay.astype(np.uint8), f"ground truth — {area_pct:.2f}% of pixels", RED),
        ]
        for c, (arr, label, colour) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr)
            ax.set_title(label, fontsize=10.5, color=colour, pad=5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if c == 0:
                ax.set_ylabel(cat, fontsize=13, fontweight="bold", color=INK, labelpad=10)

    fig.suptitle("What the model has to catch", y=0.995)
    _subtitle(fig, "Pixel-accurate masks mark every defect — many cover well under 1% of the frame", 0.972)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    return _save(fig, "fig06_defect_examples.png")


# --- Figure 9: appearance statistics ---------------------------------------
def fig_appearance(images: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    df = images.merge(manifest[["image_path", "category", "category_type"]], on="image_path")
    order = df.groupby("category").mean_intensity.median().sort_values().index.tolist()
    ctype = df.groupby("category").category_type.first()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

    ax = axes[0]
    data = [df.loc[df.category == c, "mean_intensity"].to_numpy() for c in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color=INK, lw=2))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(C_TEXTURE if ctype[c] == "texture" else C_OBJECT)
        patch.set_edgecolor(SURFACE); patch.set_linewidth(1.5); patch.set_alpha(0.92)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_MUTED)
    ax.set_yticks(np.arange(1, len(order) + 1), order)
    ax.set_xlabel("Mean pixel intensity (0–255)")
    ax.set_title("Brightness differs sharply between categories")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=C_TEXTURE, label="Texture"),
                       Patch(facecolor=C_OBJECT, label="Object")], loc="lower right")
    _despine(ax)

    ax = axes[1]
    spread = df.groupby("category").channel_spread.mean().reindex(order)
    is_grey = spread < 1.0
    ax.barh(np.arange(len(spread)), spread.values,
            color=[INK_MUTED if g else C_NORMAL for g in is_grey], height=0.68)
    for i, (v, g) in enumerate(zip(spread.values, is_grey)):
        ax.text(v + 0.6, i, "single-channel" if g else f"{v:.1f}", va="center", fontsize=9.5,
                color=INK_MUTED if g else INK_SECONDARY,
                fontweight="bold" if g else "normal")
    ax.set_yticks(np.arange(len(spread)), spread.index)
    ax.set_xlabel("Mean spread between R, G and B channels (0–255)")
    ax.set_xlim(0, max(spread.max() * 1.42, 8))
    ax.set_title("Three categories carry no colour at all")
    ax.grid(axis="y", visible=False)
    _despine(ax)

    fig.suptitle("Categories are visually heterogeneous — one global normalisation will not fit all", y=1.0)
    fig.tight_layout()
    return _save(fig, "fig09_appearance.png")


# --- Figure 10: where do defects appear? -----------------------------------
def fig_defect_location(masks: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    """Defect centroids, and whether textures and objects actually differ.

    They do not: the radial profiles overlap almost exactly. The real finding is
    that both are centre-biased, which is a property of how MVTec framed its
    shots rather than of defects in general.
    """
    lut = manifest.dropna(subset=["mask_path"]).set_index("mask_path")[["category_type"]]
    df = masks.join(lut, on="mask_path").dropna(subset=["centroid_x_norm", "centroid_y_norm"])
    df = df.assign(radius=np.hypot(df.centroid_x_norm - 0.5, df.centroid_y_norm - 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    ax = axes[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    h = ax.hist2d(df.centroid_x_norm, df.centroid_y_norm, bins=20, range=[[0, 1], [0, 1]], cmap=cmap)
    ax.set_title(f"All {len(df)} defect centroids")
    ax.set_xlabel("Horizontal position in image")
    ax.set_ylabel("Vertical position in image")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(False)
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label="defects per cell")

    ax = axes[1]
    bins = np.linspace(0, 0.72, 25)
    for group, colour in (("texture", C_TEXTURE), ("object", C_OBJECT)):
        sub = df[df.category_type == group]
        ax.hist(sub.radius, bins=bins, density=True, histtype="step", lw=2.4, color=colour,
                label=f"{group.capitalize()} (n={len(sub)})")
    ax.axvline(0.2, color=INK, lw=1.6, ls="--")
    within = 100.0 * (df.radius < 0.2).mean()
    ax.text(0.19, ax.get_ylim()[1] * 0.55,
            f"{within:.0f}% of defects sit" + chr(10) + "within 0.2 of centre",
            fontsize=10, color=INK, fontweight="bold", linespacing=1.3, ha="right")
    ax.set_xlabel("Distance of defect centre from image centre (normalised)")
    ax.set_ylabel("Density")
    ax.set_title("Textures and objects behave the same")
    ax.legend(loc="upper right")
    _despine(ax)

    fig.suptitle("Defects are centre-biased in every category type", y=1.0)
    _subtitle(fig, "The two radial profiles overlap — a real production line would not centre "
                   "defects this way, so this bias must not be learned", 0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return _save(fig, "fig10_defect_location.png")


# --- Figure 11: is the test set drawn from the same distribution? ----------
def fig_train_test_shift(images: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    df = images.merge(manifest[["image_path", "category", "split", "is_anomalous"]], on="image_path")
    train = df[df.split == "train"].groupby("category").mean_intensity.mean()
    test_good = df[(df.split == "test") & (df.is_anomalous == 0)].groupby("category").mean_intensity.mean()
    test_anom = df[(df.split == "test") & (df.is_anomalous == 1)].groupby("category").mean_intensity.mean()

    delta_good = (test_good - train).sort_values()
    delta_anom = (test_anom - train).reindex(delta_good.index)
    tol = 5.0
    n_within = int((delta_good.abs() <= tol).sum())

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    y = np.arange(len(delta_good))
    ax.axvspan(-tol, tol, color=GRID, alpha=0.75, zorder=0)
    ax.barh(y - 0.19, delta_good.values, height=0.36, color=C_NORMAL,
            label="Test normal − train", zorder=2)
    ax.barh(y + 0.19, delta_anom.values, height=0.36, color=C_ANOMALOUS,
            label="Test anomalous − train", zorder=2)
    ax.axvline(0, color=INK, lw=1.4, zorder=3)

    for i, (cat, v) in enumerate(delta_good.items()):
        if abs(v) > tol:
            ax.text(v + (1.0 if v > 0 else -1.0), i - 0.19, f"{v:+.1f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=9.5, color=RED, fontweight="bold")

    span = max(delta_good.abs().max(), delta_anom.abs().max())
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.text(0, len(delta_good) - 0.3, f"±{tol:.0f} levels", ha="center", fontsize=9.5,
            color=INK_MUTED, style="italic")
    ax.set_yticks(y, delta_good.index)
    ax.set_xlabel("Difference in mean pixel intensity vs. the training set (0–255 scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _despine(ax)

    fig.suptitle(f"Test normals match training for {n_within} of 15 categories", y=0.99)
    _subtitle(fig, "screw, grid and leather drift beyond ±5 intensity levels — "
                   "normalise per category, not globally", 0.945)
    fig.subplots_adjust(top=0.885)
    return _save(fig, "fig11_train_test_shift.png")


DEFAULT_DEFECT_PICKS = [
    ("screw", "scratch_neck"),
    ("carpet", "hole"),
    ("bottle", "broken_large"),
    ("transistor", "misplaced"),
    ("leather", "color"),
]


def fig_defect_examples_wide(manifest: pd.DataFrame, data_root: Path | None = None,
                             picks: list[tuple[str, str]] | None = None) -> Path:
    """Landscape twin of :func:`fig_defect_examples`, for 16:9 slides.

    Categories run across the columns and the three views down the rows, which
    turns the portrait figure into a wide one that fills a slide.
    """
    root = Path(data_root) if data_root is not None else get_data_root()
    picks = picks or DEFAULT_DEFECT_PICKS
    row_labels = ["normal", "defective", "ground truth"]

    fig, axes = plt.subplots(3, len(picks), figsize=(3.05 * len(picks), 9.9))
    for c, (cat, defect) in enumerate(picks):
        good = manifest[(manifest.category == cat) & (manifest.split == "train")].iloc[0]
        bad_rows = manifest[(manifest.category == cat) & (manifest.defect_type == defect)]
        bad = bad_rows.iloc[len(bad_rows) // 2]

        img = _open_display(bad.image_path, root)
        mask = _open_mask(bad.mask_path, root)
        overlay = img.copy().astype(np.float32)
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([227, 73, 72], dtype=np.float32)
        area_pct = 100.0 * mask.mean()

        panels = [
            (_open_display(good.image_path, root), None),
            (img, defect.replace("_", " ")),
            (overlay.astype(np.uint8), f"{area_pct:.2f}% of pixels"),
        ]
        for r, (arr, note) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if r == 0:
                ax.set_title(cat, fontsize=15, fontweight="bold", color=INK, pad=8)
            if note:
                ax.set_xlabel(note, fontsize=11,
                              color=C_ANOMALOUS if r == 1 else RED, labelpad=5)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=13, fontweight="bold",
                              color=INK_SECONDARY, labelpad=12)

    fig.suptitle("What the model has to catch", y=0.985, fontsize=19)
    _subtitle(fig, "Pixel-accurate masks mark every defect — many cover well under 1% of the frame",
              0.958)
    fig.tight_layout(rect=(0, 0, 1, 0.938))
    return _save(fig, "fig06b_defect_examples_wide.png")
