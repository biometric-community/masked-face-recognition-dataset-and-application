"""LFW + RMFRD verification (Table IV-style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from maskfr.data.pairs import build_lfw, build_rmfrd_pairs
from maskfr.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer, best_t = 0.0, 1.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer, best_t


@torch.no_grad()
def eval_pairs(model, ds, device, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer, thr = best_acc_eer(s, y)
    # TPR @ FPR=1e-3 proxy (paper reports TPR at fixed FAR)
    tpr_at = {}
    for far in (1e-2, 1e-3):
        # threshold from negatives
        neg = s[y == 0]
        if len(neg) == 0:
            tpr_at[f"tpr@far{far}"] = None
            continue
        t = float(np.quantile(neg, 1.0 - far))
        pos = s[y == 1]
        tpr_at[f"tpr@far{far}"] = float((pos >= t).mean()) if len(pos) else None
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} {tpr_at}")
    return {
        "accuracy": acc,
        "eer": eer,
        "threshold": thr,
        "n_pairs": int(len(y)),
        **tpr_at,
    }


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    image_size = int(cfg["model"].get("image_size", 112))
    emb_dim = int(ckpt.get("emb_dim", cfg["model"].get("emb_dim", 512)))
    model = FaceEmbedder(
        int(ckpt["num_classes"]),
        emb_dim=emb_dim,
        pretrained=False,
        loss=str(ckpt.get("loss", "cosface")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_pairs = cfg["eval"].get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    bs = int(cfg["eval"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 4))

    results = {"verification": {}}
    results["verification"]["lfw"] = eval_pairs(
        model, build_lfw(cfg, image_size, max_pairs=max_pairs), device, bs, nw, "lfw"
    )
    results["verification"]["rmfrd"] = eval_pairs(
        model,
        build_rmfrd_pairs(cfg, image_size, max_pairs=max_pairs, seed=int(cfg["train"].get("seed", 42))),
        device,
        bs,
        nw,
        "rmfrd",
    )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
