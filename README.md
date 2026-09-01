# Detecting Manufacturing Defects — MVTec AD

Visual anomaly detection for factory quality inspection, built on the
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark.

The dataset gives you 3,629 defect-free training images and **zero** defective ones. That
single constraint decides everything else in this repository.

---

## Deliverables

| What | Where |
|---|---|
| Full EDA with narrative and outputs | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| Presentation | [`reports/presentation/week1_eda.pptx`](reports/presentation/week1_eda.pptx) |
| Figures | [`reports/figures/`](reports/figures/) |
| Summary tables | [`reports/tables/`](reports/tables/) |

---

## What the EDA found

**The dataset.** 5,354 images across 15 product categories — 5 textures (carpet, grid,
leather, tile, wood) and 10 objects (bottle, cable, capsule, hazelnut, metal_nut, pill,
screw, toothbrush, transistor, zipper). 3,629 training images, 1,725 test images, and
1,258 pixel-level ground-truth masks.

**Six findings, six design decisions:**

| Finding | Decision it forces |
|---|---|
| 0 anomalies among 3,629 training images | One-class / unsupervised method |
| 73 defect types, median 17 images each | Detect deviation from normal, don't model defect classes |
| 72.9% of the test split is anomalous | Report AUROC — accuracy is misleading at this base rate |
| Median defect covers 1.54% of its image; 39.7% cover under 1% | Keep resolution high; score per patch, not per image |
| 6 resolutions (700–1024 px); 3 categories single-channel | Loader must resize consistently and expand channels |
| Brightness spans 48–185; 3 categories drift from train | One model and one normalisation per category |

Two further observations worth carrying forward:

- **Centre bias.** 48% of defects sit within 0.2 (normalised) of the image centre, and
  textures and objects behave identically in this respect. This is an artefact of how
  MVTec framed its shots — a real production line would not centre defects — so
  centre-cropping is avoided throughout.
- **`toothbrush` is the weak category.** 60 training images and a single defect type. It
  is the most likely source of an unstable score and is worth reporting separately.

## The two baselines

Both train on defect-free images only, because the data allows nothing else. The
difference between them is *where they look*.

**Convolutional autoencoder** (`ConvAutoencoder` in `src/mvtec_eda/models.py`) — compress
each image to a small spatial bottleneck and rebuild it. Trained only on normal images, so
it never learns to reconstruct a defect; anomaly score is per-pixel reconstruction error.
This is the obvious deep-learning answer, and the EDA predicts it will struggle: one
bottleneck describes the whole image, so a defect covering 1.5% of the frame barely moves
the loss.

**PaDiM** (`PaDiM` in the same module) — push images through a frozen ImageNet-pretrained
ResNet-18, with no backpropagation at all. At each position in the feature grid, fit a
multivariate Gaussian over the normal training images, then score a test patch by its
Mahalanobis distance to that position's Gaussian. Every patch position keeps its own
model, so a small local deviation cannot be averaged away.

Image scores come from the mean of the top 1% of anomaly-map pixels rather than the global
mean — again because the typical defect is small.

### Reference point

On `bottle`, PaDiM reaches **0.997 image AUROC / 0.985 pixel AUROC**, matching the
published figure for that category. Note that bottle is one of the easiest categories;
expect the 15-category mean to land lower (the PaDiM paper reports ~0.95 mean image AUROC
on a ResNet-18 backbone, with `screw` the usual weak point).

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
location, set `MVTEC_ROOT`:

```bash
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
```

## Running it

```bash
pip install -r requirements.txt
```

**Step 1 — the EDA** (fast, run these first; everything else depends on the manifest):

```bash
python scripts/01_build_manifest.py && python scripts/02_compute_stats.py && python scripts/03_make_figures.py
```

**Step 2 — the baselines** (slow — see the timing note below):

```bash
python scripts/05_run_baselines.py --epochs 15 --batch-size 32
```

**Step 3 — the result figures and the deck:**

```bash
python scripts/06_model_figures.py && python scripts/04_build_deck.py
```

Then open `notebooks/01_eda.ipynb` and Run All to refresh it with the model results. The
notebook's model section is guarded, so it also runs fine before step 2 exists.

| Script | What it does | Runtime |
|---|---|---|
| `01_build_manifest.py` | Walks the dataset, one row per image, integrity checks | ~10 s |
| `02_compute_stats.py` | Pixel statistics and defect geometry (multiprocess) | ~3 min |
| `03_make_figures.py` | The 12 EDA figures and the summary tables | ~1 min |
| `05_run_baselines.py` | Trains and evaluates both baselines, all 15 categories | see below |
| `06_model_figures.py` | Comparison charts and qualitative heatmaps | ~30 s |
| `04_build_deck.py` | Assembles the presentation | ~5 s |

### Timing and useful flags

On **CPU**, expect roughly **6–7 minutes per category** (~90 minutes total): the
autoencoder dominates, PaDiM takes about a minute. On a **CUDA GPU** it is far faster and
is picked up automatically — no flags needed.

`05_run_baselines.py` appends to `reports/tables/baseline_results.csv` after each category
and **skips categories already in that file**, so you can stop it and resume freely. To
start clean, delete that CSV first — otherwise a resumed run can mix results from
different settings.

```bash
# just PaDiM, all categories - about 15 minutes on CPU
python scripts/05_run_baselines.py --skip-ae

# one category, to check it works
python scripts/05_run_baselines.py --categories bottle --epochs 3
```

| Flag | Default | Notes |
|---|---|---|
| `--categories` | all 15 | Space-separated subset |
| `--epochs` | 30 | Autoencoder only. 15 is a reasonable CPU compromise |
| `--ae-size` | 128 | Autoencoder input resolution |
| `--padim-size` | 256 | PaDiM input resolution |
| `--batch-size` | 16 | 32 is faster if you have the RAM |
| `--n-features` | 100 | PaDiM random channel subsample |
| `--skip-ae` / `--skip-padim` | off | Run one baseline only |

**A caveat to state if you present these numbers:** at 15 epochs and 128 px the
autoencoder is genuinely undertrained. More epochs would lift it somewhat — they would not
close the gap to PaDiM, but "we gave it 15 epochs" is the honest framing, not
"autoencoders can't do this."

## Repository layout

```
├── notebooks/01_eda.ipynb        # the analysis, with narrative and outputs
├── scripts/                      # numbered pipeline, run in order
├── src/mvtec_eda/
│   ├── config.py                 # paths, category groups, constants
│   ├── manifest.py               # dataset walk + integrity audit
│   ├── stats.py                  # pixel and mask statistics
│   ├── summaries.py              # aggregate tables and headline facts
│   ├── data.py                   # per-category torch loaders
│   ├── models.py                 # autoencoder + PaDiM
│   ├── evaluate.py               # image- and pixel-level AUROC
│   └── plots.py                  # every figure
└── reports/
    ├── figures/                  # generated PNGs
    ├── tables/                   # generated CSVs and headline_facts.json
    └── presentation/week1_eda.pptx
```

Every number quoted in the deck is read from `reports/tables/` — `headline_facts.json` for
the EDA, `baseline_results.csv` for the models — so the slides cannot drift away from the
analysis behind them. The deck builder degrades gracefully: without baseline results it
produces the EDA-only deck and tells you what is missing.

## Where this goes next

AUROC on MVTec is close to saturated, so chasing it further is not the interesting work.
The gap that matters is that the test split is 72.9% anomalous while a real line is
perhaps 1%:

1. **Operating-point analysis** — pick a threshold per category and report
   precision/recall/FPR there, re-weighted to realistic defect rates. A 0.99-AUROC model
   at a 5% false-positive rate rejects hundreds of good parts per shift.
2. **Test the centre-bias claim** — re-evaluate under random crops, translations and
   rotations. PaDiM compares patch positions across images, which only holds because MVTec
   poses every object identically.
3. **Error analysis against the EDA** — join per-defect areas to per-image scores and
   confirm whether the misses are the small defects.
4. **PatchCore** as a stronger reference point.

## Dataset citation

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
> *A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

The dataset is licensed CC BY-NC-SA 4.0 and is used here for non-commercial research.
