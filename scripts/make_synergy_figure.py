"""生成 Figure 4：Full Modality Synergy Matrix（16 组合 × 6 模型热力图）。

读 scripts/eval_synergy_matrix_16.py 产出的 synergy_matrix_16.json，画两张面板：
  (a) 绝对 AUROC 热力图：6 模型 × 16 组合，ERUF 行高亮。
  (b) ΔAUROC 热力图：ERUF − 各基线（5 行 × 16 列），发散色阶，红=基线更强、绿=ERUF 更强，
      直接回答「任意稀疏组合下 ERUF 是否压制基线」。

诚实口径（写入图注）：每个格子是「实际具备该组合所需模态的子人群」上的 AUROC，非同一固定
人群；缺失其它模态内容置零、mask 强制。

输出：figures/fig5_synergy_matrix.png（300 dpi 位图）+ .pdf（矢量，供论文排版）。
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

DPI = 300
F_ANNO = 9      # 单元格数值
F_XTICK = 9     # x 轴组合名
F_YTICK = 11    # y 轴模型名
F_TITLE = 12    # 子图标题


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
    n = len(MODEL_ORDER)
    k = len(combos)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8.5), gridspec_kw={"height_ratios": [1.0, 0.85]},
    )

    # ---- (a) 绝对 AUROC ----
    im1 = ax1.imshow(M, aspect="auto", cmap="viridis", vmin=0.64, vmax=0.82)
    ax1.set_xticks(range(k)); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=F_XTICK)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(MODEL_ORDER, fontsize=F_YTICK)
    for i in range(n):
        for j in range(k):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=F_ANNO,
                     color="white" if (not np.isnan(v) and v > 0.73) else "black")
    ax1.set_title("(a) AUROC across all 16 modality combinations", fontsize=F_TITLE)
    ax1.set_ylabel("Model")
    ax1.set_xlabel("Available modality combination (bits = [ecg, echo, vitals, cxr])")
    for i in range(n):
        if MODEL_ORDER[i] == "ERUF":
            ax1.add_patch(plt.Rectangle((-0.5, i - 0.5), k, 1, fill=False,
                                        edgecolor=ERUF_COLOR, lw=2.5))
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.01)
    cbar1.set_label("AUROC")

    # ---- (b) ΔAUROC（ERUF − 基线）----
    D = np.array([[models["ERUF"][c] - models[m][c] for c in combos]
                  for m in MODEL_ORDER[1:]], dtype=float)
    vmax = float(np.nanmax(np.abs(D)))
    vmax = max(vmax, 1e-3)
    im2 = ax2.imshow(D, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(k)); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=F_XTICK)
    ax2.set_yticks(range(n - 1)); ax2.set_yticklabels(MODEL_ORDER[1:], fontsize=F_YTICK)
    for i in range(n - 1):
        for j in range(k):
            v = D[i, j]
            txt = "—" if np.isnan(v) else f"{v:+.3f}"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=F_ANNO, color="black")
    ax2.set_title("(b) ΔAUROC = ERUF − baseline (green = ERUF wins, red = baseline wins)", fontsize=F_TITLE)
    ax2.set_ylabel("Baseline")
    ax2.set_xlabel("Available modality combination (bits = [ecg, echo, vitals, cxr])")
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.01)
    cbar2.set_label("ΔAUROC")

    fig.tight_layout()
    dst = OUT / "fig5_synergy_matrix.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=DPI, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", dst, f"({DPI} dpi)")
    print("saved", dst.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
