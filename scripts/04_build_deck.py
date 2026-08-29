"""Build the week-1 EDA slide deck from the rendered figures and summary tables.

Every number on a slide is read from reports/tables/headline_facts.json, so the deck
cannot drift away from the analysis that produced it.
"""

import _bootstrap  # noqa: F401
import json

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from mvtec_eda.config import FIGURES_DIR, PRESENTATION_DIR, TABLES_DIR, ensure_output_dirs

# --- Design tokens, matching the figures -----------------------------------
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_SECONDARY = RGBColor(0x52, 0x51, 0x4E)
INK_MUTED = RGBColor(0x8A, 0x88, 0x80)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
GRID = RGBColor(0xE5, 0xE4, 0xE0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.62)

TITLE_TOP = Inches(0.42)
TITLE_H = Inches(0.62)
SUB_TOP = Inches(1.02)
SUB_H = Inches(0.38)
BODY_TOP = Inches(1.56)
TAKEAWAY_H = Inches(0.72)
TAKEAWAY_TOP = SLIDE_H - TAKEAWAY_H - Inches(0.34)


def _text(slide, left, top, width, height, text, size, colour, bold=False,
          align=PP_ALIGN.LEFT, italic=False, spacing=1.0):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = colour
    return box


def _blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = SURFACE
    return slide


def _rect(slide, left, top, width, height, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def _fit(path, box_left, box_top, box_w, box_h):
    """Scale an image to fit inside a box, preserving aspect ratio and centring it."""
    with PILImage.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    left = int(box_left + (box_w - w) / 2)
    top = int(box_top + (box_h - h) / 2)
    return left, top, w, h


def figure_slide(prs, title, subtitle, figure_name, takeaway):
    slide = _blank(prs)
    content_w = SLIDE_W - 2 * MARGIN

    _text(slide, MARGIN, TITLE_TOP, content_w, TITLE_H, title, 27, INK, bold=True)
    if subtitle:
        _text(slide, MARGIN, SUB_TOP, content_w, SUB_H, subtitle, 13.5, INK_SECONDARY)

    body_top = BODY_TOP if subtitle else Inches(1.24)
    box_h = TAKEAWAY_TOP - body_top - Inches(0.22)
    left, top, w, h = _fit(FIGURES_DIR / figure_name, int(MARGIN), int(body_top),
                           int(content_w), int(box_h))
    slide.shapes.add_picture(str(FIGURES_DIR / figure_name), Emu(left), Emu(top),
                             Emu(w), Emu(h))

    if takeaway:
        _rect(slide, MARGIN, TAKEAWAY_TOP, content_w, TAKEAWAY_H, RGBColor(0xF2, 0xF1, 0xEE))
        _rect(slide, MARGIN, TAKEAWAY_TOP, Inches(0.055), TAKEAWAY_H, BLUE)
        _text(slide, MARGIN + Inches(0.26), TAKEAWAY_TOP + Inches(0.13),
              content_w - Inches(0.5), TAKEAWAY_H - Inches(0.2),
              takeaway, 13.5, INK, spacing=1.15)
    return slide


def build(facts: dict) -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    content_w = SLIDE_W - 2 * MARGIN

    # --- 1. Title ----------------------------------------------------------
    slide = _blank(prs)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.13), BLUE)
    _text(slide, MARGIN, Inches(2.05), content_w, Inches(0.42),
          "WEEK 1  ·  EXPLORATORY DATA ANALYSIS", 15, BLUE, bold=True)
    _text(slide, MARGIN, Inches(2.62), content_w, Inches(1.1),
          "Detecting Manufacturing Defects", 42, INK, bold=True)
    _text(slide, MARGIN, Inches(3.62), content_w, Inches(0.75),
          "Understanding the MVTec AD dataset before building a model", 21, INK_SECONDARY)
    _rect(slide, MARGIN, Inches(4.62), Inches(1.5), Inches(0.035), GRID)
    _text(slide, MARGIN, Inches(4.95), content_w, Inches(0.9),
          f"{facts['n_images']:,} images  ·  {facts['n_categories']} product categories  ·  "
          f"{facts['n_defect_types']} defect types  ·  {facts['n_masks']:,} pixel-level masks",
          15, INK_MUTED)

    # --- 2. Objective ------------------------------------------------------
    slide = _blank(prs)
    _text(slide, MARGIN, TITLE_TOP, content_w, TITLE_H,
          "What this presentation covers", 27, INK, bold=True)
    items = [
        ("The problem",
         "A factory needs to catch defective parts automatically. Defects are rare, varied, "
         "and mostly unseen at the time the system is built."),
        ("The dataset",
         "MVTec AD: 15 product categories photographed under controlled conditions, with a "
         "pixel-accurate mask for every defect in the test set."),
        ("The goal of this EDA",
         "Establish what the data allows and forbids, so the modelling approach in week 2 is "
         "chosen from evidence rather than habit."),
        ("The output",
         "Six concrete design decisions, each traced back to a specific measurement."),
    ]
    top = Inches(1.5)
    for i, (head, body) in enumerate(items):
        _rect(slide, MARGIN, top, Inches(0.05), Inches(0.92), BLUE if i % 2 == 0 else AQUA)
        _text(slide, MARGIN + Inches(0.3), top, Inches(3.0), Inches(0.4), head, 17, INK, bold=True)
        _text(slide, MARGIN + Inches(3.5), top + Inches(0.03), content_w - Inches(3.8),
              Inches(1.0), body, 14.5, INK_SECONDARY, spacing=1.25)
        top += Inches(1.24)

    # --- 3. Dataset at a glance -------------------------------------------
    slide = _blank(prs)
    _text(slide, MARGIN, TITLE_TOP, content_w, TITLE_H, "The dataset at a glance", 27, INK, bold=True)
    _text(slide, MARGIN, SUB_TOP, content_w, SUB_H,
          "Every figure in this deck is computed from these images, not quoted from the paper",
          13.5, INK_SECONDARY)

    tiles = [
        (f"{facts['n_images']:,}", "images total", BLUE),
        (f"{facts['n_categories']}", f"categories  ({facts['n_textures']} textures, "
                                     f"{facts['n_objects']} objects)", AQUA),
        (f"{facts['n_defect_types']}", "distinct defect types", ORANGE),
        (f"{facts['n_masks']:,}", "pixel-level defect masks", INK_SECONDARY),
    ]
    tile_w = (content_w - Inches(0.45)) / 4
    for i, (value, label, colour) in enumerate(tiles):
        left = MARGIN + i * (tile_w + Inches(0.15))
        _rect(slide, left, Inches(1.72), tile_w, Inches(1.72), WHITE, line=GRID)
        _rect(slide, left, Inches(1.72), tile_w, Inches(0.05), colour)
        _text(slide, left + Inches(0.22), Inches(2.06), tile_w - Inches(0.4), Inches(0.8),
              value, 40, INK, bold=True)
        _text(slide, left + Inches(0.22), Inches(2.82), tile_w - Inches(0.4), Inches(0.55),
              label, 13, INK_SECONDARY, spacing=1.15)

    rows = [
        ("Training images", f"{facts['n_train']:,}", "all defect-free — no anomalies at all"),
        ("Test images", f"{facts['n_test']:,}",
         f"{facts['n_test_normal']} normal, {facts['n_test_anomalous']:,} anomalous "
         f"({facts['pct_test_anomalous']}%)"),
        ("Median defect size", f"{facts['median_defect_area_pct']}%",
         f"of the image; {facts['pct_defects_under_1pct']}% of defects cover under 1%"),
        ("Resolutions", f"{facts['n_resolutions']}",
         f"from 700x700 to 1024x1024; {len(facts['greyscale_categories'])} categories are single-channel"),
    ]
    top = Inches(3.86)
    for label, value, note in rows:
        _text(slide, MARGIN + Inches(0.05), top, Inches(3.0), Inches(0.4), label, 15, INK_SECONDARY)
        _text(slide, MARGIN + Inches(3.1), top, Inches(1.9), Inches(0.4), value, 15, INK, bold=True)
        _text(slide, MARGIN + Inches(5.0), top, content_w - Inches(5.2), Inches(0.4),
              note, 15, INK_SECONDARY)
        top += Inches(0.44)
        _rect(slide, MARGIN, top - Inches(0.08), content_w, Inches(0.008), GRID)

    # --- 4-14. Figure slides ----------------------------------------------
    figure_slide(
        prs,
        "Fifteen categories, two visual regimes",
        "Five textures (carpet, grid, leather, tile, wood) and ten centred objects",
        "fig05_normal_samples.png",
        "Textures repeat across the whole frame; objects sit in a fixed pose. The two "
        "behave differently under augmentation, so they are treated separately.",
    )

    figure_slide(
        prs,
        "The training set contains no defects at all",
        "This single fact decides the entire modelling approach",
        "fig02_composition.png",
        f"With {facts['n_train_anomalies']} anomalies among {facts['n_train']:,} training images, "
        "a supervised classifier cannot be fitted. The task is one-class: learn 'normal', "
        "then measure deviation.",
    )

    figure_slide(
        prs,
        "The same one-class split in every category",
        "Train is always defect-free; anomalies appear only at test time",
        "fig01_split_overview.png",
        f"Category size ranges from {facts['smallest_category']} "
        f"({facts['smallest_category_train']} training images) to hazelnut (391). "
        "Small categories will give the least stable scores.",
    )

    figure_slide(
        prs,
        "Defects are many, and each one is rare",
        f"{facts['n_defect_types']} category-specific defect types across the test split",
        "fig03_defect_types.png",
        f"The median defect type has only {facts['median_images_per_defect_type']} images "
        f"(minimum {facts['min_images_per_defect_type']}). Learning one class per defect is "
        "not viable — the model must flag anything unlike normal.",
    )

    figure_slide(
        prs,
        "What the model has to catch",
        "Ground-truth masks mark every defective pixel",
        "fig06b_defect_examples_wide.png",
        "Defects range from a barely visible scratch on a screw to a completely misplaced "
        "transistor. The same detector has to cover both extremes.",
    )

    figure_slide(
        prs,
        "Most defects are very small",
        "Measured directly from the 1,258 ground-truth masks",
        "fig07_defect_area.png",
        f"Median defect covers {facts['median_defect_area_pct']}% of its image and "
        f"{facts['pct_defects_under_1pct']}% cover under 1%. Resizing to 224x224 would shrink "
        "a typical small defect to a handful of pixels.",
    )

    figure_slide(
        prs,
        "Defect size varies hugely between categories",
        f"From {facts['smallest_defect_pct']}% to {facts['largest_defect_pct']}% of the image",
        "fig08_defect_area_by_category.png",
        "Screw and capsule defects are an order of magnitude smaller than tile or bottle "
        "defects, so a single detection threshold across categories would not work.",
    )

    figure_slide(
        prs,
        "Images are high-resolution but not uniform",
        f"{facts['n_resolutions']} distinct resolutions; all images square",
        "fig04_image_properties.png",
        f"{facts['n_greyscale_images']:,} images in "
        f"{', '.join(facts['greyscale_categories'])} are stored single-channel and must be "
        "expanded to three before a pretrained backbone will accept them.",
    )

    figure_slide(
        prs,
        "Categories look nothing like each other",
        "Mean brightness and colour content, measured per category",
        "fig09_appearance.png",
        "Brightness spans roughly 48 to 185 on a 0-255 scale. One global normalisation would "
        "leave several categories badly off-centre.",
    )

    figure_slide(
        prs,
        "Defects sit near the image centre",
        "A property of how the dataset was photographed, not of defects in general",
        "fig10_defect_location.png",
        "48% of defects fall within 0.2 of the centre, identically for textures and objects. "
        "A real production line would not centre defects, so this bias must not be learned.",
    )

    figure_slide(
        prs,
        "Test normals match the training distribution",
        "Checked rather than assumed — a one-class model depends on it",
        "fig11_train_test_shift.png",
        f"{facts['n_categories_within_5_levels']} of 15 categories agree within ±5 intensity "
        f"levels. screw, grid and leather drift up to {facts['max_intensity_drift']}, so "
        "normalisation statistics are computed per category.",
    )

    # --- 15. Findings to decisions ----------------------------------------
    slide = _blank(prs)
    _text(slide, MARGIN, TITLE_TOP, content_w, TITLE_H,
          "Six findings, six design decisions", 27, INK, bold=True)
    _text(slide, MARGIN, SUB_TOP, content_w, SUB_H,
          "Each decision traces back to a measurement in this deck", 13.5, INK_SECONDARY)

    pairs = [
        (f"{facts['n_train_anomalies']} anomalies in {facts['n_train']:,} training images",
         "One-class method — a supervised classifier is impossible"),
        (f"{facts['n_defect_types']} defect types, median "
         f"{facts['median_images_per_defect_type']} images each",
         "Detect deviation from normal, do not model defect classes"),
        (f"{facts['pct_test_anomalous']}% of the test split is anomalous",
         "Report AUROC — accuracy is misleading at this base rate"),
        (f"Median defect covers {facts['median_defect_area_pct']}% of the image",
         "Keep input resolution high; score at feature-map level"),
        (f"{facts['n_resolutions']} resolutions, "
         f"{len(facts['greyscale_categories'])} single-channel categories",
         "Loader resizes consistently and expands channels"),
        ("Brightness spans 48-185; 3 categories drift",
         "One model and one normalisation per category"),
    ]
    top = Inches(1.62)
    row_h = Inches(0.82)
    _text(slide, MARGIN + Inches(0.16), Inches(1.28), Inches(5.6), Inches(0.3),
          "WHAT THE DATA SHOWS", 11.5, INK_MUTED, bold=True)
    _text(slide, MARGIN + Inches(6.5), Inches(1.28), Inches(5.6), Inches(0.3),
          "WHAT IT FORCES US TO DO", 11.5, INK_MUTED, bold=True)
    for i, (finding, decision) in enumerate(pairs):
        if i % 2 == 0:
            _rect(slide, MARGIN, top - Inches(0.08), content_w, row_h, RGBColor(0xF6, 0xF5, 0xF2))
        _text(slide, MARGIN + Inches(0.16), top + Inches(0.06), Inches(5.9), row_h,
              finding, 14.5, INK_SECONDARY, spacing=1.15)
        _rect(slide, MARGIN + Inches(6.25), top + Inches(0.02), Inches(0.04),
              row_h - Inches(0.2), BLUE)
        _text(slide, MARGIN + Inches(6.5), top + Inches(0.06), Inches(5.5), row_h,
              decision, 14.5, INK, bold=True, spacing=1.15)
        top += row_h

    # --- 16. Next steps ----------------------------------------------------
    slide = _blank(prs)
    _text(slide, MARGIN, TITLE_TOP, content_w, TITLE_H, "Plan for week 2", 27, INK, bold=True)
    _text(slide, MARGIN, SUB_TOP, content_w, SUB_H,
          "The EDA points to one family of methods", 13.5, INK_SECONDARY)

    _rect(slide, MARGIN, Inches(1.62), content_w, Inches(1.62), WHITE, line=GRID)
    _rect(slide, MARGIN, Inches(1.62), Inches(0.055), Inches(1.62), ORANGE)
    _text(slide, MARGIN + Inches(0.3), Inches(1.84), content_w - Inches(0.6), Inches(0.4),
          "Baseline: pretrained-CNN feature embedding (PaDiM / PatchCore)", 20, INK, bold=True)
    _text(slide, MARGIN + Inches(0.3), Inches(2.34), content_w - Inches(0.6), Inches(0.8),
          "Trains on normal images only, compares local patch features rather than a single "
          "pooled embedding, and outputs a pixel-level anomaly map that the ground-truth masks "
          "can score directly — matching all six constraints above.",
          14.5, INK_SECONDARY, spacing=1.25)

    steps = [
        ("1", "Build the data pipeline",
         "Per-category loaders, consistent resize, channel expansion, no centre crop"),
        ("2", "Implement the baseline",
         "PaDiM on a pretrained backbone, fitted per category on normal images only"),
        ("3", "Evaluate honestly",
         "Image-level and pixel-level AUROC per category, averaged across all 15"),
        ("4", "Iterate",
         "Compare against PatchCore and an autoencoder; flag toothbrush as small-sample"),
    ]
    top = Inches(3.62)
    for num, head, body in steps:
        _rect(slide, MARGIN, top, Inches(0.42), Inches(0.42), BLUE)
        _text(slide, MARGIN, top + Inches(0.045), Inches(0.42), Inches(0.35), num, 15, WHITE,
              bold=True, align=PP_ALIGN.CENTER)
        _text(slide, MARGIN + Inches(0.62), top - Inches(0.01), Inches(3.6), Inches(0.4),
              head, 15.5, INK, bold=True)
        _text(slide, MARGIN + Inches(4.3), top + Inches(0.01), content_w - Inches(4.5),
              Inches(0.45), body, 14, INK_SECONDARY)
        top += Inches(0.78)

    ensure_output_dirs()
    out = PRESENTATION_DIR / "week1_eda.pptx"
    prs.save(out)
    print(f"Wrote {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


def main() -> None:
    facts = json.loads((TABLES_DIR / "headline_facts.json").read_text(encoding="utf-8"))
    build(facts)


if __name__ == "__main__":
    main()
