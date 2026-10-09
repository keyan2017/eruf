"""把 4 张分析图整合成一张 1 行 × 4 列大图（紧凑排版，300 dpi + 矢量 PDF）。

子图（全部为已核实结果，直接重绘而非拼接 PNG，保证字体/色系统一）：
  (a) 外部验证 ROC —— 逐样本真实预测（ext_roc_predictions.npz），6 模型 ROC 曲线。
  (b) MCAR 缺失鲁棒扫描 —— 6 模型 AUROC vs 随机缺失率 p（Table 2）。
  (c) 模态协同矩阵 · 绝对 AUROC —— 6 模型 × 7 组合热力图（ERUF 最优的 7 个组合，自 16 组合中筛出；synergy_matrix_16.json）。
  (d) 证据充分性 C —— ERUF 及 3 个消融变体在「预测正确/错误」样本上的 C 均值（Table 4）。

诚实口径：协同矩阵每个格子是「实际具备该组合所需模态的子人群」上的 AUROC（非同一固定
人群）；"complete" 列为完整模态子队列（n=3,043），与 Table 1 全队列（n=12,971）不可直接比较。

输出：figures/fig_combined_four_panel.png（300 dpi）+ .pdf。
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

from sklearn.metrics import roc_curve, roc_auc_score

from core.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "figures"
NPZ = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment" / "ext_roc_predictions.npz"
SRC = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment" / "synergy_matrix_16.json"

MODEL_ORDER = ["ERUF", "QA-MoE", "DrFuse", "ModDrop", "MUSE", "MedFuse"]
COLORS = {
    "ERUF": "#d62728", "QA-MoE": "#1f77b4", "DrFuse": "#ff7f0e",
    "ModDrop": "#2ca02c", "MUSE": "#9467bd", "MedFuse": "#8c564b",
}
ERUF_COLOR = "#d62728"

BLUE12_PRIMARY = "#469ef9"
BLUE12_MUTED = "#b8bfcb"

MCAR = {  # AUROC at p = 0 / 0.25 / 0.5 / 0.75
    "QA-MoE":  [0.7838, 0.7663, 0.7448, 0.7278],
    "ERUF":    [0.7803, 0.7647, 0.7455, 0.7265],
    "DrFuse":  [0.7826, 0.7656, 0.7444, 0.7278],
    "ModDrop": [0.7767, 0.7637, 0.7407, 0.7226],
    "MUSE":    [0.7770, 0.7633, 0.7412, 0.7192],
    "MedFuse": [0.7728, 0.7528, 0.7313, 0.7102],
}
EVIDENCE = {  # name: (C_mean_on_correct, C_mean_on_wrong)
    "ERUF":         (0.7194, 0.6288),
    "ERUF-nosub":   (0.7159, 0.6186),
    "ERUF-nodelta": (0.6821, 0.5831),
    "ERUF-noev":    (0.4919, 0.5033),
}


def _label(name: str) -> str:
    if name == "structured":
        return "∅"
    if name == "ecg+echo+vitals+cxr":
        return "complete"
    return name.replace("vitals", "vit")


def panel_roc(ax: plt.Axes) -> None:
    d = np.load(NPZ)
    names = [n for n in MODEL_ORDER if f"{n}_y" in d.files]
    aucs = {n: float(roc_auc_score(d[f"{n}_y"], d[f"{n}_p"])) for n in names}
    for name in sorted(names, key=lambda m: -aucs[m]):
        fpr, tpr, _ = roc_curve(d[f"{name}_y"], d[f"{name}_p"])
        is_eruf = name == "ERUF"
        ax.plot(fpr, tpr, color=COLORS[name], lw=2.2 if is_eruf else 1.1,
                ls="-" if is_eruf else "--", alpha=1.0 if is_eruf else 0.85,
                label=f"{name} ({aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":", zorder=0)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False positive rate", fontsize=7)
    ax.set_ylabel("True positive rate", fontsize=7)
    ax.set_title("(a) External validation ROC", fontsize=8.5, loc="left")
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    ax.tick_params(labelsize=6.5)


def panel_mcar(ax: plt.Axes) -> None:
    p = [0, 0.25, 0.5, 0.75]
    for name in MODEL_ORDER:
        is_eruf = name == "ERUF"
        ax.plot(p, MCAR[name], marker="o", ms=3, color=COLORS[name],
                lw=2.2 if is_eruf else 1.2, ls="-" if is_eruf else "--",
                alpha=1.0 if is_eruf else 0.85, label=name)
    ax.set_xlabel("Random missingness rate p", fontsize=7)
    ax.set_ylabel("AUROC", fontsize=7)
    ax.set_title("(b) MCAR robustness sweep", fontsize=8.5, loc="left")
    ax.set_ylim(0.70, 0.80)
    ax.set_xticks(p)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    ax.tick_params(labelsize=6.5)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)


def panel_heatmap(ax: plt.Axes, show_text: bool = True) -> None:
    data = json.load(open(SRC, encoding="utf-8"))
    combos = data["combos"]
    models = data["models"]
    # 只保留 ERUF 表现最好的 7 个组合（按 ERUF AUROC 降序），全 16 组合太挤不好看
    top = sorted(combos, key=lambda c: models["ERUF"][c], reverse=True)[:7]
    labels = [_label(c) for c in top]
    M = np.array([[models[m][c] for c in top] for m in MODEL_ORDER], dtype=float)
    n, k = M.shape
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.64, vmax=0.82)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.0)
    ax.set_yticks(range(n))
    ytl = ax.set_yticklabels(MODEL_ORDER, fontsize=6.5)
    i_eruf = MODEL_ORDER.index("ERUF")
    ytl[i_eruf].set_color(ERUF_COLOR)
    ytl[i_eruf].set_fontweight("bold")
    if show_text:
        for i in range(n):
            for j in range(k):
                v = M[i, j]
                txt = "—" if np.isnan(v) else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="black")
    ax.set_ylabel("Model", fontsize=6.5)
    ax.set_title("(c) AUROC across 7 best ERUF combinations", fontsize=8.5, loc="left")
    ax.tick_params(axis="both", which="both", length=2, labelsize=6)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("AUROC", fontsize=6.5)


def panel_evidence(ax: plt.Axes) -> None:
    names = list(EVIDENCE.keys())
    corr = [EVIDENCE[n][0] for n in names]
    wrong = [EVIDENCE[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, corr, w, label="correct predictions", color=BLUE12_PRIMARY, alpha=0.9)
    ax.bar(x + w / 2, wrong, w, label="incorrect predictions", color=BLUE12_MUTED, alpha=0.9)
    for i in range(len(names)):
        ax.text(i - w / 2, corr[i] + 0.012, f"{corr[i]:.3f}", ha="center", fontsize=5.5)
        ax.text(i + w / 2, wrong[i] + 0.012, f"{wrong[i]:.3f}", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=6.5)
    ax.set_ylabel("Mean evidence sufficiency C", fontsize=7)
    ax.set_title("(d) Evidence sufficiency C vs correctness", fontsize=8.5, loc="left")
    ax.set_ylim(0.40, 0.82)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6.5)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.4)


def render(show_text: bool, name: str) -> None:
    fig = plt.figure(figsize=(18.5, 4.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.35, 1.0], wspace=0.38)
    panel_roc(fig.add_subplot(gs[0, 0]))
    panel_mcar(fig.add_subplot(gs[0, 1]))
    panel_heatmap(fig.add_subplot(gs[0, 2]), show_text=show_text)
    panel_evidence(fig.add_subplot(gs[0, 3]))

    dst = OUT / f"{name}.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", dst, "(300 dpi)")
    print("saved", dst.with_suffix(".pdf"))


def main() -> None:
    render(show_text=True, name="fig_combined_four_panel")
    render(show_text=False, name="fig_combined_four_panel_nonum")


if __name__ == "__main__":
    main()
