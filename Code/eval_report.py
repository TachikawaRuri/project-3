"""Validation metrics + Grad-CAM figures for the report.

Usage:
    python eval_report.py --ckpt ../runs/r18_base/best.pt --outdir ../runs/r18_base/report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from train import (DATA_ROOT, LABELS, FundusDataset, build_model, build_transforms,
                   worker_init)


def get_target_layer(model, name):
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(name)


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):
        self.acts = out.detach()

    def _bwd(self, _, __, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy()


def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--n-cam", type=int, default=8, help="how many val images to render")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name, size = ckpt["model"], ckpt.get("size", 224)
    torch.set_num_threads(10)
    model = build_model(name, len(LABELS), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, eval_tf = build_transforms(size, strong=False)
    ds = FundusDataset(DATA_ROOT / "val.csv", eval_tf)
    ld = DataLoader(ds, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, worker_init_fn=worker_init)

    probs, targets = [], []
    with torch.no_grad():
        for x, y, _ in ld:
            probs.append(torch.sigmoid(model(x)).numpy())
            targets.append(y.numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    per_class = {}
    rows = []
    for j, lab in enumerate(LABELS):
        col = {"label": lab, "n_pos": int(targets[:, j].sum()),
               "pos_rate": round(float(targets[:, j].mean()), 4)}
        if targets[:, j].min() != targets[:, j].max():
            col["auc"] = round(float(roc_auc_score(targets[:, j], probs[:, j])), 4)
        prec, rec, f1, _ = precision_recall_fscore_support(
            targets[:, j], (probs[:, j] >= 0.5).astype(int), average="binary", zero_division=0)
        col.update({"precision@0.5": round(float(prec), 4),
                    "recall@0.5": round(float(rec), 4),
                    "f1@0.5": round(float(f1), 4)})
        per_class[lab] = col
        rows.append(col)

    aucs = [v["auc"] for v in per_class.values() if "auc" in v]
    summary = {
        "model": name, "epoch": ckpt["epoch"], "img_size": size,
        "macro_auc": round(float(np.mean(aucs)), 4),
        "micro_auc": round(float(roc_auc_score(targets.ravel(), probs.ravel())), 4),
        "subset_accuracy(acc@0.5)": round(float(accuracy_score(targets, (probs >= 0.5).astype(int))), 4),
        "f1_micro@0.5": round(float(f1_score(targets, (probs >= 0.5).astype(int), average="micro", zero_division=0)), 4),
        "f1_macro@0.5": round(float(f1_score(targets, (probs >= 0.5).astype(int), average="macro", zero_division=0)), 4),
        "n_val": int(len(targets)),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- Grad-CAM: pick the most confident correct example per label ----
    if args.n_cam > 0:
        try:
            from matplotlib import cm
            from PIL import ImageDraw
        except ImportError:
            print("matplotlib missing, skipping Grad-CAM")
            return
        cam = GradCAM(model, get_target_layer(model, name))
        picked = []
        for j, lab in enumerate(LABELS):
            cand = np.where(targets[:, j] == 1)[0]
            if len(cand) == 0:
                continue
            best = cand[np.argmax(probs[cand, j])]
            picked.append((int(best), j, lab))
        picked = picked[: args.n_cam]

        for rank, (idx, j, lab) in enumerate(picked):
            img, _, _ = ds[idx]
            heat = cam(img.unsqueeze(0), j)
            base = denormalize(img).permute(1, 2, 0).numpy()
            base = (base * 255).astype(np.uint8)
            hm = (cm.jet(heat)[..., :3] * 255).astype(np.uint8)
            overlay = (0.55 * base + 0.45 * hm).astype(np.uint8)
            canvas = np.concatenate([base, overlay], axis=1)
            im = Image.fromarray(canvas)
            ImageDraw.Draw(im).text((6, 6), f"{lab} p={probs[idx, j]:.2f}",
                                    fill=(255, 255, 255))
            im.save(out / f"cam_{rank:02d}_{lab}.png")
        print(f"saved {len(picked)} Grad-CAM figures to {out}")


if __name__ == "__main__":
    main()
