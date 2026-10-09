"""单独画出证据充分性 C 面板（对应 fig_combined_four_panel.png 的 (d) 子图）。

ERUF 及 3 个消融变体在「预测正确 / 预测错误」样本上的 C 均值（数据 = Table 4，
与 fig5_evidence.png 的 C-vs-correctness 面板口径一致）。

输出：figures/fig_evidence_panel.png（300 dpi）+ .pdf。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "figures"

EVIDENCE = {  # name: (C_mean_on_correct, C_mean_on_wrong)
    "ERUF":         (0.7194, 0.6288),
    "ERUF-nosub":   (0.7159, 0.6186),
    "ERUF-nodelta": (0.6821, 0.5831),
    "ERUF-noev":    (0.4919, 0.5033),
}


def main() -> None:
    names = list(EVIDENCE.keys())
    corr = [EVIDENCE[n][0] for n in names]
    wrong = [EVIDENCE[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars_c = ax.bar(x - w / 2, corr, w, label="correct predictions",
                    color="#2ca02c", alpha=0.9)
    bars_w = ax.bar(x + w / 2, wrong, w, label="incorrect predictions",
                    color="#9e9e9e", alpha=0.9)
    for i in range(len(names)):
        ax.text(i - w / 2, corr[i] + 0.012, f"{corr[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, wrong[i] + 0.012, f"{wrong[i]:.3f}", ha="center", fontsize=8)

    # ERUF 柱高亮描边
    bars_c[0].set_edgecolor("#d62728"); bars_c[0].set_linewidth(1.5)
    bars_w[0].set_edgecolor("#d62728"); bars_w[0].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean evidence sufficiency C", fontsize=9)
    ax.set_ylim(0.40, 0.82)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.4)

    fig.tight_layout()
    dst = OUT / "fig_evidence_panel.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", dst, "(300 dpi)")
    print("saved", dst.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
