"""Train the mono height head on pseudolabels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from rocksnitch.logging_utils import get_logger
from rocksnitch.training.mono_head import MonoHeadConfig, MonoHeightHead, build_head
from rocksnitch.training.pseudolabel import PseudolabelRecord, read_pseudolabels


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 40
    grad_clip: float = 1.0
    device: str = "cuda"
    val_frac: float = 0.1
    seed: int = 0


def _prepare_arrays(records: Iterable[PseudolabelRecord]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for r in records:
        if r.feature is None:
            continue
        xs.append(np.asarray(r.feature, dtype=np.float32))
        ys.append(r.height_m)
    if not xs:
        raise ValueError("No pseudolabels with feature vectors found.")
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def train_head(
    labels_path: Path,
    out_dir: Path,
    *,
    config: TrainConfig | None = None,
) -> Path:
    """Train and save the mono head. Returns the checkpoint path."""
    cfg = config or TrainConfig()
    log = get_logger(__name__)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_pseudolabels(Path(labels_path))
    X, y = _prepare_arrays(records)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    nval = max(1, int(len(X) * cfg.val_frac))
    Xv, yv = X[:nval], y[:nval]
    Xt, yt = X[nval:], y[nval:]

    head_cfg = MonoHeadConfig(feature_dim=X.shape[-1])
    head = build_head(head_cfg)

    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise RuntimeError("Training requires torch. `pip install -e '.[gpu]'`.") from e

    head._lazy_build_torch()
    assert head._torch_model is not None
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model: nn.Module = head._torch_model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(np.log(np.clip(yt, 1e-3, None))).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(np.log(np.clip(yv, 1e-3, None))).to(device)

    best_val = float("inf")
    ckpt_path = out_dir / "last.ckpt"
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(Xt_t), device=device)
        losses = []
        for i in range(0, len(Xt_t), cfg.batch_size):
            batch = perm[i : i + cfg.batch_size]
            pred = model(Xt_t[batch]).squeeze(-1)
            loss = loss_fn(pred, yt_t[batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv_t).squeeze(-1)
            val_loss = float(loss_fn(pred_v, yv_t).item())
        log.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, float(np.mean(losses)), val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
    log.info("Best val_loss=%.4f saved to %s", best_val, ckpt_path)
    return ckpt_path


__all__ = ["TrainConfig", "train_head"]
