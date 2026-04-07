"""Training loop for the homoglyph visual similarity model.

Reads all hyperparameters from configs/default.yaml (or a config supplied via
--config).  Each run is saved to outputs/runs/<run_name>/ and contains:
  - best.pt          : checkpoint of the model with the highest val AUC
  - config.yaml      : copy of the config used for this run
  - log.csv          : epoch-level metrics (train_loss, val_loss, val_auc)

Usage:
    python training/train.py
    python training/train.py --config configs/my_experiment.yaml
"""
import argparse
import csv
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.encoder import VisualEncoder
from models.similarity import SimilarityHead
from training.dataset import NamePairDataset, collate_fn


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_run_name(cfg: dict) -> str:
    name = cfg.get("run_name")
    if not name:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(name)


def save_config(cfg: dict, dest: Path) -> None:
    with dest.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device):
    r = cfg["rendering"]
    s = cfg["slicing"]
    m = cfg["model"]

    slice_dim = r["height"] * s["slice_width"]
    encoder = VisualEncoder(
        slice_dim=slice_dim,
        embed_dim=m["embed_dim"],
        pooling=m["pooling"],
    ).to(device)
    head = SimilarityHead(embed_dim=m["embed_dim"]).to(device)
    return encoder, head


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    r = cfg["rendering"]
    s = cfg["slicing"]
    t = cfg["training"]

    dataset_kwargs = dict(
        height=r["height"],
        background=r["background"],
        slice_width=s["slice_width"],
        stride=s["stride"],
        remove_padding=s["remove_padding"],
        pad_to_width=s.get("pad_to_width"),
    )
    loader_kwargs = dict(
        batch_size=t["batch_size"],
        collate_fn=collate_fn,
        num_workers=t["num_workers"],
        pin_memory=True,
    )

    train_ds = NamePairDataset(ROOT / cfg["data"]["train_pkl"], **dataset_kwargs)
    val_ds   = NamePairDataset(ROOT / cfg["data"]["val_pkl"],   **dataset_kwargs)

    max_samples = cfg["data"].get("max_samples")
    if max_samples is not None:
        from collections import defaultdict
        from torch.utils.data import Subset
        import random

        def stratified_indices(ds, n: int) -> list[int]:
            rng = random.Random(42)  # fixed seed → same subset for every model
            groups: dict[int, list[int]] = defaultdict(list)
            for i, (_, _, label) in enumerate(ds.rows):
                groups[int(label)].append(i)
            per_class = n // len(groups)
            indices: list[int] = []
            for label_indices in groups.values():
                indices.extend(rng.sample(label_indices, min(per_class, len(label_indices))))
            rng.shuffle(indices)
            return indices

        train_ds = Subset(train_ds, stratified_indices(train_ds, max_samples))
        val_ds   = Subset(val_ds,   stratified_indices(val_ds,   max_samples))

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(
    encoder: VisualEncoder,
    head: SimilarityHead,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, list[float], list[int]]:
    """Run one train or val epoch.

    Returns:
        mean_loss, all_scores (sigmoid probabilities), all_labels
    """
    training = optimizer is not None
    encoder.train(training)
    head.train(training)

    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[int]   = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (slices_a, slices_b, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            slices_a = slices_a.to(device)
            slices_b = slices_b.to(device)
            labels_f = labels.float().to(device)

            emb_a  = encoder(slices_a)
            emb_b  = encoder(slices_b)
            logits = head(emb_a, emb_b)
            loss   = criterion(logits, labels_f)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_scores.extend(probs if isinstance(probs, list) else [probs])
            all_labels.extend(labels.tolist())

    mean_loss = total_loss / len(all_labels)
    return mean_loss, all_scores, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    run_name = resolve_run_name(cfg)
    cfg["run_name"] = run_name          # bake the resolved name back in

    run_dir = ROOT / "outputs" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.yaml")
    shutil.copy(ROOT / args.config, run_dir / "source_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name}")
    print(f"Dir: {run_dir}")
    print(f"Device: {device}\n")

    train_loader, val_loader = build_loaders(cfg)
    encoder, head = build_model(cfg, device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc"])

    best_auc = -1.0
    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(
            encoder, head, train_loader, criterion, optimizer, device
        )
        val_loss, val_scores, val_labels = run_epoch(
            encoder, head, val_loader, criterion, None, device
        )

        val_auc = roc_auc_score(val_labels, val_scores)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc])

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                },
                run_dir / "best.pt",
            )
            print(f"  ✓ new best val_auc={best_auc:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
