"""Figures + REPORT.md for masked-face recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from maskfr.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        [h["epoch"] for h in hist],
        [h["loss"] for h in hist],
        color=MULTI_SERIES_COLORS[0],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CosFace CE loss")
    ax.set_title("CosFace ResNet-18 (CelebA + mask aug)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(ver: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = list(ver.keys())
    accs = [ver[k]["accuracy"] * 100 for k in names]
    eers = [ver[k]["eer"] * 100 for k in names]
    x = range(len(names))
    w = 0.35
    ax.bar([i - w / 2 for i in x], accs, w, label="Accuracy", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.bar([i + w / 2 for i in x], eers, w, label="EER", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("%")
    ax.set_title("Verification: LFW vs RMFRD (masked↔unmasked)")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Masked Face Recognition Dataset and Application",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin: CosFace ResNet-18 trained on CelebA with lower-face mask augmentation;",
        "evaluated on LFW and RMFRD masked↔unmasked verification pairs.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Mask aug prob: {train_s.get('mask_aug_prob')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification",
        "",
        "| Dataset | Accuracy | EER | TPR@FAR1e-2 | TPR@FAR1e-3 | Pairs |",
        "|---------|----------|-----|-------------|-------------|-------|",
    ]
    for k, v in sorted(eval_res.get("verification", {}).items()):
        t2 = v.get("tpr@far0.01")
        t3 = v.get("tpr@far0.001")
        t2s = f"{t2*100:.2f}%" if t2 is not None else "—"
        t3s = f"{t3*100:.2f}%" if t3 is not None else "—"
        lines.append(
            f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {t2s} | {t3s} | {v.get('n_pairs', '—')} |"
        )
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verify_metrics.svg)",
        "",
        "## Notes",
        "",
        "- Paper trains on WebFace-MASK / MS1MV3 / Glint360k; we use CelebA + synthetic mask aug under the 5 GiB gate.",
        "- RMFRD pairs are built from common AFDB face/masked identity folders (see `DEVIATIONS.md`).",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_verify(eval_res["verification"], fig_dir / "verify_metrics.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
