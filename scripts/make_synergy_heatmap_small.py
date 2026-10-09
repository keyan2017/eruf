"""单独画出「AUROC across all 16 modality combinations」绝对 AUROC 热力图（小图版，供 PPT 作子图）。

只取 make_synergy_figure.py 的面板 (a)（6 模型 × 16 组合绝对 AUROC），画成更小的紧凑单图。
数据源：experiments/mimic/ph_experiment/synergy_matrix_16.json（已核实）。

诚实口径（与 make_synergy_figure.py 一致）：每个格子是「实际具备该组合所需模态的子人群」上的
AUROC，非同一固定人群；缺失其它模态内容置零、mask 强制；"complete" 列为完整模态子队列
（n=3,043），与 Table 1 全队列（n=12,971）不可直接比较。

输出：figures/fig_synergy_heatmap_ppt.png（300 dpi）+ .pdf。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "figures"
SRC = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment" / "synergy_matrix_16.json"
MODEL_ORDER = ["ERUF", "QA-MoE", "DrFuse", "ModDrop", "MUSE", "MedFuse"]
ERUF_COLOR = "#d62728"


def _label(name: str) -> str:
    if name == "structured":
        return "∅"
    if name == "ecg+echo+vitals+cxr":
        return "complete"
    return name.replace("vitals", "vit")


def main() -> None:
    data = json.load(open(SRC, encoding="utf-8"))
    combos = data["combos"]
    labels = [_label(c) for c in combos]
    models = data["models"]

    M = np.array([[models[m][c] for c in combos] for m in MODEL_ORDER], dtype=float)
    n, k = M.shape

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.64, vmax=0.82)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(MODEL_ORDER, fontsize=7.0)
    for i in range(n):
        for j in range(k):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                    color="white" if (not np.isnan(v) and v > 0.73) else "black")
    # ERUF 行高亮
    for i in range(n):
        if MODEL_ORDER[i] == "ERUF":
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), k, 1, fill=False,
                                       edgecolor=ERUF_COLOR, lw=2.0))
    ax.set_xlabel("Available modality combination (bits = [ecg, echo, vitals, cxr])", fontsize=7.0)
    ax.set_ylabel("Model", fontsize=7.0)
    ax.tick_params(axis="both", which="both", length=2)

    cbar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
    cbar.ax.tick_params(labelsize=6.0)
    cbar.set_label("AUROC", fontsize=7.0)

    fig.tight_layout()
    dst = OUT / "fig_synergy_heatmap_ppt.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", dst, "(300 dpi, small for PPT)")
    print("saved", dst.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
