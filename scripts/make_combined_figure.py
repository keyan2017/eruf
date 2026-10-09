"""把三个分析合并成一张多子图大图（紧凑排版，300 dpi + 矢量 PDF）。

子图（按数据来源，全部为已核实结果）：
  (a) MCAR 缺失鲁棒扫描 —— 6 模型 AUROC vs 随机缺失率 p（数据 = Table 2）。
  (b) 证据充分性 C —— 4 个 ERUF 变体在「预测正确 / 错误」样本上的 C 均值（数据 = Table 4）。
  (c) 模态协同矩阵 · 绝对 AUROC —— 6 模型 × 16 组合热力图（synergy_matrix_16.json）。
  (d) 模态协同矩阵 · ΔAUROC —— ERUF − 5 基线 × 16 组合发散热力图（同 JSON）。

诚实口径与 make_synergy_figure.py 一致：协同矩阵每个格子是「实际具备该组合所需模态的
子人群」上的 AUROC（非同一固定人群），缺失其它模态置零、mask 强制；"complete" 列为
完整模态子队列（n=3,043），与 Table 1 全队列（n=12,971）不可直接比较。

输出：figures/fig_combined_robustness.png（300 dpi）+ .pdf。
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

# ---- 已核实数据（Table 2 / Table 4）----
MCAR = {  # AUROC at p = 0 / 0.25 / 0.5 / 0.75
    "QA-MoE":  [0.7838, 0.7663, 0.7448, 0.7278],
    "ERUF":    [0.7803, 0.7647, 0.7455, 0.7265],
    "DrFuse":  [0.7826, 0.7656, 0.7444, 0.7278],
    "ModDrop": [0.7767, 0.7637, 0.7407, 0.7226],
    "MUSE":    [0.7770, 0.7633, 0.7412, 0.7192],
    "MedFuse": [0.7728, 0.7528, 0.7313, 0.7102],
}
EVIDENCE = {  # (C_mean_on_correct, C_mean_on_wrong)
    "ERUF (full)":       (0.7194, 0.6288),
    "- subset discr.":   (0.7159, 0.6186),
    "- evidence C":      (0.6821, 0.5831),
    "- δ/rectifier":     (0.4919, 0.5033),
}

MODEL_ORDER = ["ERUF", "QA-MoE", "DrFuse", "ModDrop", "MUSE", "MedFuse"]
COLORS = {
    "ERUF": "#d62728", "QA-MoE": "#1f77b4", "DrFuse": "#ff7f0e",
    "ModDrop": "#2ca02c", "MUSE": "#9467bd", "MedFuse": "#8c564b",
}
ERUF_COLOR = "#d62728"


def _label(name: str) -> str:
    if name == "structured":
        return "∅"
    if name == "ecg+echo+vitals+cxr":
        return "complete"
    return name.replace("vitals", "vit")


def panel_mcar(ax: plt.Axes) -> None:
    p = [0, 0.25, 0.5, 0.75]
    for name in MODEL_ORDER:
        is_eruf = name == "ERUF"
        ax.plot(p, MCAR[name], marker="o", ms=3.5, label=name,
                color=ERUF_COLOR if is_eruf else COLORS[name],
                lw=2.4 if is_eruf else 1.3, ls="-" if is_eruf else "--",
                alpha=1.0 if is_eruf else 0.85)
    ax.set_xlabel("Random missingness rate p", fontsize=8)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_title("(a) MCAR robustness sweep", fontsize=9, loc="left")
    ax.set_ylim(0.70, 0.80)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    ax.tick_params(labelsize=7)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)


def panel_evidence(ax: plt.Axes) -> None:
    names = list(EVIDENCE.keys())
    corr = [EVIDENCE[n][0] for n in names]
    wrong = [EVIDENCE[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, corr, w, label="correct predictions", color="#2ca02c", alpha=0.9)
    ax.bar(x + w / 2, wrong, w, label="incorrect predictions", color="#9e9e9e", alpha=0.9)
    for i in range(len(names)):
        ax.text(i - w / 2, corr[i] + 0.012, f"{corr[i]:.3f}", ha="center", fontsize=6)
        ax.text(i + w / 2, wrong[i] + 0.012, f"{wrong[i]:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=6.5)
    ax.set_ylabel("Mean evidence sufficiency C", fontsize=8)
    ax.set_title("(b) Evidence sufficiency C vs correctness", fontsize=9, loc="left")
    ax.set_ylim(0.40, 0.82)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.4)


def panel_synergy_abs(ax: plt.Axes, combos: list[str], labels: list[str],
                      models: dict[str, dict[str, float]]) -> None:
    k = len(combos)
    M = np.array([[models[m][c] for c in combos] for m in MODEL_ORDER], dtype=float)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.64, vmax=0.82)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(MODEL_ORDER, fontsize=7)
    for i in range(len(MODEL_ORDER)):
        for j in range(k):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if (not np.isnan(v) and v > 0.74) else "black")
    ax.set_title("(c) Synergy — AUROC across 16 combinations", fontsize=9, loc="left")
    for i in range(len(MODEL_ORDER)):
        if MODEL_ORDER[i] == "ERUF":
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), k, 1, fill=False,
                                       edgecolor=ERUF_COLOR, lw=2.0))
    fig = ax.figure
    fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9).set_label("AUROC", fontsize=7)


def panel_synergy_delta(ax: plt.Axes, combos: list[str], labels: list[str],
                        models: dict[str, dict[str, float]]) -> None:
    k = len(combos)
    D = np.array([[models["ERUF"][c] - models[m][c] for c in combos]
                  for m in MODEL_ORDER[1:]], dtype=float)
    vmax = float(np.nanmax(np.abs(D)))
    vmax = max(vmax, 1e-3)
    im = ax.imshow(D, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(MODEL_ORDER) - 1)); ax.set_yticklabels(MODEL_ORDER[1:], fontsize=7)
    for i in range(len(MODEL_ORDER) - 1):
        for j in range(k):
            v = D[i, j]
            txt = "—" if np.isnan(v) else f"{v:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="black")
    ax.set_title("(d) Synergy — ΔAUROC (ERUF − baseline, green = ERUF wins)",
                 fontsize=9, loc="left")
    ax.figure.colorbar(im, ax=ax, pad=0.01, shrink=0.9).set_label("ΔAUROC", fontsize=7)


def main() -> None:
    data = json.load(open(SRC, encoding="utf-8"))
    combos = data["combos"]
    labels = [_label(c) for c in combos]
    models = data["models"]

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.5], width_ratios=[1, 1],
                          hspace=0.42, wspace=0.14)
    panel_mcar(fig.add_subplot(gs[0, 0]))
    panel_evidence(fig.add_subplot(gs[0, 1]))
    panel_synergy_abs(fig.add_subplot(gs[1, 0]), combos, labels, models)
    panel_synergy_delta(fig.add_subplot(gs[1, 1]), combos, labels, models)

    dst = OUT / "fig_combined_robustness.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", dst, "(300 dpi)")
    print("saved", dst.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
