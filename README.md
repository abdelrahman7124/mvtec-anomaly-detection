# Detecting Manufacturing Defects — MVTec AD

Visual anomaly detection for factory quality inspection, built on the
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark.

**Status: week 1 of 4 — exploratory data analysis complete.**

The goal of this phase is not to build a model but to establish what the data allows and
forbids, so the modelling approach is chosen from evidence. Everything below is measured
from the images themselves, not quoted from the dataset paper.

---

## Deliverables

| What | Where |
|---|---|
| Full EDA with narrative and outputs | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| Week-1 presentation (16 slides) | [`reports/presentation/week1_eda.pptx`](reports/presentation/week1_eda.pptx) |
| Figures | [`reports/figures/`](reports/figures/) |
| Summary tables | [`reports/tables/`](reports/tables/) |

---

## What the EDA found

**The dataset.** 5,354 images across 15 product categories — 5 textures (carpet, grid,
leather, tile, wood) and 10 objects (bottle, cable, capsule, hazelnut, metal_nut, pill,
screw, toothbrush, transistor, zipper). 3,629 training images, 1,725 test images, and
1,258 pixel-level ground-truth masks.

**The defining constraint.** The training split contains **zero anomalies**. All 3,629
training images are defect-free; every defect appears only at test time. This rules out a
supervised classifier outright and makes the task one-class: model what "normal" looks
like, then measure deviation from it.

**Six findings, six design decisions:**

| Finding | Decision it forces |
|---|---|
| 0 anomalies among 3,629 training images | One-class / unsupervised method |
| 73 defect types, median 17 images each | Detect deviation from normal, don't model defect classes |
| 72.9% of the test split is anomalous | Report AUROC — accuracy is misleading at this base rate |
| Median defect covers 1.54% of its image; 39.7% cover under 1% | Keep input resolution high; score at feature-map level |
| 6 resolutions (700–1024 px); 3 categories single-channel | Loader must resize consistently and expand channels |
| Brightness spans 48–185; 3 categories drift from train | One model and one normalisation per category |

Two further observations worth carrying forward:

- **Centre bias.** 48% of defects sit within 0.2 (normalised) of the image centre, and
  textures and objects behave identically in this respect. This is an artefact of how
  MVTec framed its shots — a real production line would not centre defects — so
  centre-cropping should be avoided.
- **`toothbrush` is the weak category.** 60 training images and a single defect type. It
  is the most likely source of an unstable score and is worth reporting separately.

**Planned baseline for week 2:** a pretrained-CNN feature-embedding method (PaDiM or
PatchCore). Both train on normal images only, compare local patch features rather than a
single pooled embedding, and produce a pixel-level anomaly map the ground-truth masks can
score directly — matching every constraint above.

---

## Getting the data

The dataset is ~4.9 GB and is **not** committed. Download it from
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) (free for
non-commercial use) and extract it so the layout looks like:

```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/
│   ├── test/{good,broken_large,broken_small,contamination}/
│   └── ground_truth/{broken_large,broken_small,contamination}/
├── cable/
└── ... 13 more categories
```

The code looks in `~/Downloads/mvtec_anomaly_detection` by default. To use another
location, set the environment variable:

```bash
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
```

## Running the analysis

```bash
pip install -r requirements.txt
```

Then run the four steps in order:

```bash
python scripts/01_build_manifest.py && python scripts/02_compute_stats.py && python scripts/03_make_figures.py && python scripts/04_build_deck.py
```

| Step | What it does | Runtime |
|---|---|---|
| `01_build_manifest.py` | Walks the dataset, writes one row per image, runs integrity checks | ~10 s |
| `02_compute_stats.py` | Per-image pixel statistics and per-mask defect geometry (multiprocess) | ~3 min |
| `03_make_figures.py` | Renders all 12 figures and the summary tables | ~1 min |
| `04_build_deck.py` | Assembles the 16-slide presentation | ~5 s |

The notebook reads the tables produced by steps 1–2, so run those before opening it.

## Repository layout

```
├── notebooks/01_eda.ipynb        # the analysis, with narrative and outputs
├── scripts/                      # numbered pipeline, run in order
├── src/mvtec_eda/
│   ├── config.py                 # paths, category groups, constants
│   ├── manifest.py               # dataset walk + integrity audit
│   ├── stats.py                  # pixel and mask statistics
│   ├── summaries.py              # aggregate tables and headline facts
│   └── plots.py                  # every figure
└── reports/
    ├── figures/                  # generated PNGs
    ├── tables/                   # generated CSVs and headline_facts.json
    └── presentation/week1_eda.pptx
```

Every number quoted in the deck is read from `reports/tables/headline_facts.json`, which
is generated by step 3 — so the slides cannot drift away from the analysis behind them.

## Dataset citation

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
> *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

The dataset is licensed CC BY-NC-SA 4.0 and is used here for non-commercial research.
