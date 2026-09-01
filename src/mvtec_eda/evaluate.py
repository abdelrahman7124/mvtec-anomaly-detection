"""Metrics for one-class anomaly detection.

The EDA ruled out accuracy: 72.9% of the test split is anomalous, so predicting
"anomalous" for everything already scores 72.9%. AUROC is threshold-free and
unaffected by that base rate, which is why both metrics below use it.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Does the model rank defective images above clean ones?"""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def pixel_auroc(
    masks: np.ndarray,
    maps: np.ndarray,
    max_pixels: int = 4_000_000,
    seed: int = 0,
) -> float:
    """Does the model put high scores on the pixels the mask marks as defective?

    Subsamples when there are more pixels than ``max_pixels`` - the estimate is
    stable well below the full 60M+ pixels a category would otherwise contribute.
    """
    y = np.asarray(masks).ravel().astype(np.uint8)
    s = np.asarray(maps).ravel().astype(np.float32)
    if y.min() == y.max():
        return float("nan")

    if y.size > max_pixels:
        rng = np.random.default_rng(seed)
        # Keep every defective pixel; subsample the normal ones, which dominate.
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        budget = max(max_pixels - pos.size, 1)
        if neg.size > budget:
            neg = rng.choice(neg, size=budget, replace=False)
        keep = np.concatenate([pos, neg])
        y, s = y[keep], s[keep]

    return float(roc_auc_score(y, s))


@torch.no_grad()
def collect_scores(score_fn, loader, out_size: int):
    """Run ``score_fn`` over a loader and gather labels, image scores and maps.

    ``score_fn`` takes a batch of images and returns an anomaly map shaped
    ``(B, 1, out_size, out_size)``.
    """
    from .models import image_score_from_map

    labels, scores, all_maps, all_masks = [], [], [], []
    for images, batch_labels, masks in loader:
        amap = score_fn(images)
        if amap.shape[-1] != out_size:
            amap = torch.nn.functional.interpolate(
                amap, size=(out_size, out_size), mode="bilinear", align_corners=False
            )
        labels.append(np.asarray(batch_labels))
        scores.append(image_score_from_map(amap))
        all_maps.append(amap.squeeze(1).numpy().astype(np.float32))
        all_masks.append(masks.squeeze(1).numpy().astype(np.uint8))

    return (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(all_maps),
        np.concatenate(all_masks),
    )
