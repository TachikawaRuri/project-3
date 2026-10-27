"""Run inference on test.csv and write submission.csv (probabilities).

Usage:
    python predict.py --ckpt ../runs/r18_base/best.pt --out ../submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import DATA_ROOT, LABELS, FundusDataset, build_model, build_transforms, worker_init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "submission.csv"))
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--tta", default="hflip", choices=["none", "hflip"],
                   help="test-time augmentation")
    p.add_argument("--save-probs", action="store_true",
                   help="also save raw probs+paths npz next to the csv")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name, size = ckpt["model"], ckpt.get("size", 224)
    print(f"checkpoint: {model_name}, val macro AUC {ckpt['macro_auc']:.4f}, epoch {ckpt['epoch']}")

    torch.set_num_threads(args.threads)
    model = build_model(model_name, len(LABELS), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, eval_tf = build_transforms(size, strong=False)
    ds = FundusDataset(DATA_ROOT / "test.csv", eval_tf)
    ld = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    worker_init_fn=worker_init)

    all_probs = []
    with torch.no_grad():
        for x, _, _ in ld:
            p1 = torch.sigmoid(model(x)).float()
            if args.tta == "hflip":
                p2 = torch.sigmoid(model(torch.flip(x, dims=[3]))).float()
                p1 = (p1 + p2) / 2
            all_probs.append(p1.numpy())
    probs = np.concatenate(all_probs)

    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    pred_df = pd.DataFrame({"path": ds.df["path"].values})
    for j, name in enumerate(LABELS):
        pred_df[name] = probs[:, j]
    # keep the official row order
    sub = sample[["path"]].merge(pred_df, on="path", how="left")
    assert sub[LABELS].notna().all().all(), "some test images missing predictions"
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  shape={sub.shape}  cols={list(sub.columns)}")
    print(sub.head(3).to_string())

    if args.save_probs:
        np.savez(Path(args.out).with_suffix(".npz"),
                 paths=sub["path"].values, probs=sub[LABELS].values)


if __name__ == "__main__":
    main()
