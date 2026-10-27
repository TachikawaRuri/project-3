"""Project 3 - Fundus multi-label classification (20 labels).

Single-model training script. CPU-friendly.

Usage:
    python train.py --model resnet18 --epochs 20 --out ../runs/r18_base
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = [
    "DR", "NORMAL", "MH", "ODC", "TSLN", "ARMD", "DN", "MYA", "BRVO", "ODP",
    "CRVO", "CNV", "RS", "ODE", "LS", "CSR", "HTR", "ASR", "CRS", "OTHER",
]
DATA_ROOT = Path(os.environ.get(
    "PROJECT3_DATA", Path(__file__).resolve().parent.parent / "student_dataset"))


def worker_init(_):
    """Workers must stay single-threaded, otherwise they fight the main process
    for CPU cores and training slows to a crawl."""
    torch.set_num_threads(1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FundusDataset(Dataset):
    def __init__(self, csv_path, transform, img_dir=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir) if img_dir else DATA_ROOT / "images"
        self.transform = transform
        self.has_labels = LABELS[0] in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["path"]).convert("RGB")
        img = self.transform(img)
        if self.has_labels:
            y = torch.tensor(row[LABELS].values.astype("float32"))
        else:
            y = torch.zeros(len(LABELS), dtype=torch.float32)
        return img, y, idx


def build_transforms(size, strong=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if strong:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            norm,
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def build_model(name, num_classes=20, pretrained=True):
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model {name}")
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y, _ in loader:
        x = x.to(device)
        probs.append(torch.sigmoid(model(x)).float().cpu().numpy())
        targets.append(y.numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    per_class = {}
    for j, name in enumerate(LABELS):
        if targets[:, j].min() != targets[:, j].max():
            per_class[name] = float(roc_auc_score(targets[:, j], probs[:, j]))
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    micro = float(roc_auc_score(targets.ravel(), probs.ravel()))
    return {"macro_auc": macro, "micro_auc": micro, "per_class": per_class,
            "probs": probs, "targets": targets}


def compute_pos_weight(csv_path):
    df = pd.read_csv(csv_path)
    pos = df[LABELS].sum().values.astype(np.float64)
    neg = len(df) - pos
    return (neg / np.maximum(pos, 1)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--pos-weight", action="store_true",
                   help="use inverse-frequency pos_weight in BCE")
    p.add_argument("--pos-weight-power", type=float, default=0.5,
                   help="pos_weight = (neg/pos)**power, cap at 20")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--weak-aug", action="store_true")
    args = p.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    print("device:", device, flush=True)

    train_tf, eval_tf = build_transforms(args.size, strong=not args.weak_aug)
    train_ds = FundusDataset(DATA_ROOT / "train.csv", train_tf)
    val_ds = FundusDataset(DATA_ROOT / "val.csv", eval_tf)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=False,
                          drop_last=False, persistent_workers=args.workers > 0,
                          worker_init_fn=worker_init)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=False,
                        persistent_workers=args.workers > 0,
                        worker_init_fn=worker_init)

    model = build_model(args.model, len(LABELS), not args.no_pretrained).to(device)
    if args.pos_weight:
        pw = np.clip(compute_pos_weight(DATA_ROOT / "train.csv") ** args.pos_weight_power, 1.0, 20.0)
        print("pos_weight:", dict(zip(LABELS, np.round(pw, 2))), flush=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "macro_auc", "micro_auc",
                                "epoch_sec", "best_macro_auc"])

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y, _ in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total / max(n, 1)
        scheduler.step()

        metrics = evaluate(model, val_ld, device)
        elapsed = time.time() - t0
        tag = ""
        if metrics["macro_auc"] > best:
            best = metrics["macro_auc"]
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "macro_auc": best, "size": args.size}, out / "best.pt")
            tag = "  <-- best"
        np.savez(out / "val_last.npz", probs=metrics["probs"], targets=metrics["targets"])
        print(f"ep {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | "
              f"macroAUC {metrics['macro_auc']:.4f} | microAUC {metrics['micro_auc']:.4f} | "
              f"{elapsed:.1f}s{tag}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{scheduler.get_last_lr()[0]:.6f}",
                                    f"{train_loss:.6f}", f"{metrics['macro_auc']:.6f}",
                                    f"{metrics['micro_auc']:.6f}", f"{elapsed:.1f}",
                                    f"{best:.6f}"])
        with open(out / "val_metrics.json", "w") as f:
            json.dump({"epoch": epoch, "macro_auc": metrics["macro_auc"],
                       "micro_auc": metrics["micro_auc"],
                       "per_class": metrics["per_class"]}, f, indent=2)

    print(f"BEST macro AUC = {best:.4f}  ->  {out/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
