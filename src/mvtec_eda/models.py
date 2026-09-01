"""Two one-class baselines for MVTec AD.

Both train on defect-free images only, because the EDA established there is no
other option: the training split contains zero anomalies.

* :class:`ConvAutoencoder` - the naive deep-learning approach. Learn to
  reconstruct normal images; whatever reconstructs badly is called anomalous.
* :class:`PaDiM` - fits a Gaussian to the pretrained-CNN features at each patch
  position and scores by Mahalanobis distance. No backpropagation at all.

The contrast between them is the point: the autoencoder compresses the whole
image into one bottleneck, which is exactly what destroys a defect covering 1.5%
of the frame. PaDiM keeps every patch position separate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


# --------------------------------------------------------------------------
# Baseline 1: convolutional autoencoder
# --------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    """Symmetric conv autoencoder with a hard spatial bottleneck.

    Input and output are in [0, 1]; train with MSE against the input.
    """

    def __init__(self, latent_dim: int = 128, base: int = 32) -> None:
        super().__init__()

        def down(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def up(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            down(3, base), down(base, base * 2), down(base * 2, base * 4),
            down(base * 4, base * 8),
            nn.Conv2d(base * 8, latent_dim, 3, padding=1),
        )
        # 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            up(base * 8, base * 4), up(base * 4, base * 2), up(base * 2, base),
            nn.ConvTranspose2d(base, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel squared reconstruction error, summed over colour channels."""
        recon = self(x)
        return ((recon - x) ** 2).mean(dim=1, keepdim=True)


def train_autoencoder(
    model: ConvAutoencoder,
    loader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 10,
) -> list[float]:
    """Fit the autoencoder on normal images. Returns the per-epoch mean loss."""
    model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        total, n_batches = 0.0, 0
        for images, _labels, _masks in loader:
            images = images.to(device)
            optimiser.zero_grad()
            loss = F.mse_loss(model(images), images)
            loss.backward()
            optimiser.step()
            total += loss.item()
            n_batches += 1
        scheduler.step()
        history.append(total / max(n_batches, 1))
        if epoch % log_every == 0 or epoch == epochs:
            print(f"      epoch {epoch:>3}/{epochs}  loss {history[-1]:.5f}")
    return history


# --------------------------------------------------------------------------
# Baseline 2: PaDiM
# --------------------------------------------------------------------------
class PaDiM:
    """Patch Distribution Modeling on frozen ResNet-18 features.

    For each position in the feature grid, fit a multivariate Gaussian over the
    training images, then score a test patch by its Mahalanobis distance to that
    position's Gaussian.
    """

    def __init__(
        self,
        n_features: int = 100,
        device: str = "cpu",
        seed: int = 0,
        eps: float = 0.01,
    ) -> None:
        self.device = device
        self.n_features = n_features
        self.eps = eps

        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        backbone.eval().to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        # layer1/2/3 give 64 + 128 + 256 = 448 channels once aligned.
        total_channels = 64 + 128 + 256
        generator = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(total_channels, generator=generator)[:n_features]

        self.mean: torch.Tensor | None = None
        self.inv_cov: torch.Tensor | None = None

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate layer1-3 features, aligned to the layer1 grid."""
        m = self.backbone
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f1 = m.layer1(x)
        f2 = m.layer2(f1)
        f3 = m.layer3(f2)
        size = f1.shape[-2:]
        feats = torch.cat(
            [
                f1,
                F.interpolate(f2, size=size, mode="nearest"),
                F.interpolate(f3, size=size, mode="nearest"),
            ],
            dim=1,
        )
        return feats[:, self.idx]  # subsample channels immediately to save memory

    @torch.no_grad()
    def fit(self, loader) -> "PaDiM":
        chunks = []
        for images, _labels, _masks in loader:
            chunks.append(self._embed(images.to(self.device)).cpu())
        embeddings = torch.cat(chunks)  # (N, C, H, W)

        n, c, h, w = embeddings.shape
        flat = embeddings.permute(0, 2, 3, 1).reshape(n, h * w, c)  # (N, P, C)

        self.mean = flat.mean(dim=0)  # (P, C)
        centred = flat - self.mean
        # (P, C, C) covariance, one per patch position.
        cov = torch.einsum("npi,npj->pij", centred, centred) / max(n - 1, 1)
        cov += self.eps * torch.eye(c).unsqueeze(0)
        self.inv_cov = torch.linalg.inv(cov)
        self.grid = (h, w)
        return self

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor, out_size: int) -> torch.Tensor:
        """Mahalanobis distance per patch, upsampled and smoothed to image size."""
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("call fit() before scoring")

        emb = self._embed(x.to(self.device)).cpu()
        b, c, h, w = emb.shape
        flat = emb.permute(0, 2, 3, 1).reshape(b, h * w, c)

        delta = flat - self.mean.unsqueeze(0)  # (B, P, C)
        # sqrt(d^T S^-1 d), batched over positions.
        left = torch.einsum("bpi,pij->bpj", delta, self.inv_cov)
        dist = torch.sqrt(torch.clamp((left * delta).sum(-1), min=0.0))

        amap = dist.reshape(b, 1, h, w)
        amap = F.interpolate(amap, size=(out_size, out_size), mode="bilinear",
                             align_corners=False)
        return _gaussian_blur(amap, sigma=4.0)


def _gaussian_blur(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur - PaDiM smooths its map before scoring."""
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).to(x.dtype)
    k1 = kernel.view(1, 1, 1, -1)
    k2 = kernel.view(1, 1, -1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k1)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k2)
    return x


def image_score_from_map(amap: torch.Tensor, top_percent: float = 1.0) -> np.ndarray:
    """Collapse an anomaly map to one score per image.

    The mean over the top ``top_percent`` of pixels, not the global mean: the EDA
    showed a typical defect covers ~1.5% of the frame, so averaging everything
    would drown it in normal background.
    """
    b = amap.shape[0]
    flat = amap.reshape(b, -1)
    k = max(1, int(flat.shape[1] * top_percent / 100.0))
    top = torch.topk(flat, k, dim=1).values
    return top.mean(dim=1).numpy()
