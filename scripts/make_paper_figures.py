"""Generate paper figures for the ERUF paper from verified experimental results.

Data sources (all verified against the summary JSONs under experiments/mimic/ph_experiment/):
  - qamoe_summary.json               -> QA-MoE
  - baselines_summary.json            -> DrFuse, ModDrop, MUSE, MedFuse
  - eruf_full_summary.json            -> ERUF
  - eruf_{noev,nodelta,nosub}_summary.json -> ERUF ablations
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
os.makedirs(OUT, exist_ok=True)

# ---- verified data -------------------------------------------------------
# (full AUROC, AUPRC, ECE, ext AUROC, ext AUPRC, Delta)
MAIN = {
    "QA-MoE":  (0.7564, 0.8221, 0.1386, 0.7194, 0.2874, 0.0560),
    "ERUF":    (0.7534, 0.8199, 0.0591, 0.7438, 0.3047, 0.0538),
    "DrFuse":  (0.7526, 0.8187, 0.0938, 0.7283, 0.2740, 0.0548),
    "ModDrop": (0.7478, 0.8170, 0.0534, 0.7157, 0.2882, 0.0541),
    "MUSE":    (0.7468, 0.8122, 0.0718, 0.7085, 0.2767, 0.0578),
    "MedFuse": (0.7401, 0.8113, 0.0888, 0.6546, 0.2287, 0.0626),
}
MCAR = {  # AUROC at p = 0 / 0.25 / 0.5 / 0.75
    "QA-MoE":  [0.7838, 0.7663, 0.7448, 0.7278],
    "ERUF":    [0.7803, 0.7647, 0.7455, 0.7265],
    "DrFuse":  [0.7826, 0.7656, 0.7444, 0.7278],
    "ModDrop": [0.7767, 0.7637, 0.7407, 0.7226],
    "MUSE":    [0.7770, 0.7633, 0.7412, 0.7192],
    "MedFuse": [0.7728, 0.7528, 0.7313, 0.7102],
}
ABLATION = {  # (ext AUROC, ECE, Delta)
    "ERUF (full)":                (0.7438, 0.0591, 0.0538),
    "- subset discrimination":     (0.6954, 0.0655, 0.0502),
    "- evidence head C":           (0.7136, 0.0797, 0.0472),
    "- delta / rectifier":         (0.7403, 0.1037, 0.0485),
}
EVIDENCE = {  # (C_mean, ece_C, err_det_auroc, C_correct, C_wrong)
    "ERUF":           (0.6923, 0.0182, 0.6813, 0.7194, 0.6288),
    "ERUF-nosub":     (0.6858, 0.0070, 0.6930, 0.7159, 0.6186),
    "ERUF-nodelta":   (0.6511, 0.0358, 0.7036, 0.6821, 0.5831),
    "ERUF-noev":      (0.4955, 0.1892, 0.3854, 0.4919, 0.5033),
}
MATRIX = {
    "structured": 0.7037, "ecg": 0.7376, "echo": 0.7277, "vitals": 0.7306,
    "cxr": 0.7191, "ecg+echo": 0.7485, "ecg+echo+vitals": 0.7712, "all": 0.7803,
}

ERUF_COLOR = "#d62728"
BASELINE_COLOR = "#1f77b4"

def _order(methods):
    return [m for m in MAIN if m in methods]

# ---- Fig 1: main results (full vs ext AUROC) ----------------------------
def fig1_main():
    methods = list(MAIN.keys())
    full = [MAIN[m][0] for m in methods]
    ext = [MAIN[m][3] for m in methods]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    colors = [ERUF_COLOR if m == "ERUF" else BASELINE_COLOR for m in methods]
    ax.bar(x - w/2, full, w, label="Full-cohort AUROC", color=colors, alpha=0.9)
    ax.bar(x + w/2, ext, w, label="External-validation AUROC", color=colors, alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0.60, 0.82); ax.set_ylabel("AUROC")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_main_auroc.png", dpi=200); plt.close(fig)

# ---- Fig 2: calibration (ECE) and robustness (Delta) ---------------------
def fig2_calib_robust():
    methods = list(MAIN.keys())
    ece = [MAIN[m][2] for m in methods]
    delta = [MAIN[m][5] for m in methods]
    x = np.arange(len(methods))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = [ERUF_COLOR if m == "ERUF" else BASELINE_COLOR for m in methods]
    a1.bar(x, ece, color=colors, alpha=0.9)
    a1.set_xticks(x); a1.set_xticklabels(methods, rotation=30, ha="right")
    a1.set_ylabel("ECE (lower is better)"); a1.set_title("Calibration error")
    a2.bar(x, delta, color=colors, alpha=0.9)
    a2.set_xticks(x); a2.set_xticklabels(methods, rotation=30, ha="right")
    a2.set_ylabel("ΔAUROC (lower is more robust)"); a2.set_title("MCAR robustness decay")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_calibration_robustness.png", dpi=200); plt.close(fig)

# ---- Fig 3: MCAR robustness sweep ----------------------------------------
def fig3_mcar():
    p = [0, 0.25, 0.5, 0.75]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m, vals in MCAR.items():
        c = ERUF_COLOR if m == "ERUF" else None
        lw = 2.6 if m == "ERUF" else 1.4
        ls = "-" if m == "ERUF" else "--"
        ax.plot(p, vals, marker="o", ms=4, label=m, color=c, lw=lw, ls=ls)
    ax.set_xlabel("Random missingness rate p"); ax.set_ylabel("AUROC")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_mcar_sweep.png", dpi=200); plt.close(fig)

# ---- Fig 4: mechanism ablation -------------------------------------------
def fig4_ablation():
    names = list(ABLATION.keys())
    ext = [ABLATION[n][0] for n in names]
    ece = [ABLATION[n][1] for n in names]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = [ERUF_COLOR if i == 0 else BASELINE_COLOR for i in range(len(names))]
    a1.bar(x, ext, color=cols, alpha=0.9)
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=20, ha="right")
    a1.set_ylabel("External-validation AUROC"); a1.set_title("External ablation")
    for i, v in enumerate(ext):
        a1.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    a2.bar(x, ece, color=cols, alpha=0.9)
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=20, ha="right")
    a2.set_ylabel("ECE (lower is better)"); a2.set_title("Calibration ablation")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_ablation.png", dpi=200); plt.close(fig)

# ---- Fig 5: evidence sufficiency C ---------------------------------------
def fig5_evidence():
    names = list(EVIDENCE.keys())
    corr = [EVIDENCE[n][3] for n in names]
    wrong = [EVIDENCE[n][4] for n in names]
    errdet = [EVIDENCE[n][2] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = [ERUF_COLOR if i == 0 else BASELINE_COLOR for i in range(len(names))]
    a1.bar(x - w/2, corr, w, label="C on correct predictions", color=cols, alpha=0.9)
    a1.bar(x + w/2, wrong, w, label="C on incorrect predictions", color=cols, alpha=0.45)
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=20, ha="right")
    a1.set_ylabel("Mean C"); a1.legend(fontsize=7, frameon=False)
    a1.set_title("Evidence sufficiency C vs correctness")
    a2.bar(x, errdet, color=cols, alpha=0.9)
    a2.axhline(0.5, ls="--", lw=0.8, color="gray")
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=20, ha="right")
    a2.set_ylabel("Error-detection AUROC"); a2.set_title("Error-recognition ability")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_evidence.png", dpi=200); plt.close(fig)

# ---- Fig 6: missing-modality performance matrix --------------------------
def fig6_matrix():
    combos = list(MATRIX.keys())
    vals = [MATRIX[c] for c in combos]
    x = np.arange(len(combos))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    cols = [ERUF_COLOR if c == "all" else BASELINE_COLOR for c in combos]
    ax.bar(x, vals, color=cols, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(combos, rotation=30, ha="right")
    ax.set_ylabel("AUROC"); ax.set_ylim(0.65, 0.82)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_matrix.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig1_main(); fig2_calib_robust(); fig3_mcar()
    fig4_ablation(); fig5_evidence(); fig6_matrix()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
