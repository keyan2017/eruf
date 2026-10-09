"""生成 Figure：外部验证（EchoNext）ROC 曲线，突出 ERUF 最优。

诚实口径：逐样本真实预测概率 + 真实标签（pasp_gte_45），不是拿 Table 1 的标量 AUROC
画假曲线。每个模型加载已保存 checkpoint，在 EchoNext（仅 ECG 零样本）上跑 predict_probs
得到 (y, p)，再用 sklearn.roc_curve 画真实 ROC。

模型口径（与训练/矩阵评估脚本完全一致）：
  - ERUF：4 模态（optional=['ecg','echo','vitals','cxr']，去 ct）+ rectifier 开启；
    外部集用 NoCTDataset（mask 5→4，去掉 ct）。
  - 5 基线：5 模态（optional=['ct','ecg','echo','vitals','cxr']，ct 恒缺失），
    外部集用原始 EchoNext（mask=[0,1,0,0,0]）。

输出：figures/fig_ext_roc.png（300 dpi）+ .pdf（矢量）；并把逐样本预测存到
experiments/mimic/ph_experiment/ext_roc_predictions.npz 以便复用。

用法：
  python scripts/make_ext_roc_curve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

from sklearn.metrics import roc_curve, roc_auc_score  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.config import PROJECT_ROOT, load_config  # noqa: E402
from datasets.ph_experiment import load_fixed_echonext  # noqa: E402
from evaluation.prediction import predict_probs  # noqa: E402
from evaluation.metrics import auroc  # noqa: E402
from models.encoders import MLPEncoder  # noqa: E402
from models.image_encoders import ResECGWaveformEncoder  # noqa: E402
from models.eruf import ERUFModel  # noqa: E402
from models.medfuse import MedFuseModel  # noqa: E402
from models.muse import MUSEModel  # noqa: E402
from models.qamoe import QAMoEModel  # noqa: E402
from models.drfuse import DrFuseModel  # noqa: E402
from models.moddrop import ModDropContrastiveModel  # noqa: E402
from models.base import make_baseline_encoders  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment"
FIG_DIR = PROJECT_ROOT / "figures"
RELEVANT = ["ecg", "echo", "vitals", "cxr"]          # ERUF 的 4 模态（去 ct）
OPTIONAL5 = ["ct", "ecg", "echo", "vitals", "cxr"]    # 基线的 5 模态顺序

# 论文 Table 1 的 Ext-val AUROC（用于核对重算结果，防止画错）
PAPER_EXT_AUROC = {
    "ERUF": 0.7438,
    "QA-MoE": 0.7194,
    "DrFuse": 0.7283,
    "ModDrop": 0.7157,
    "MUSE": 0.7085,
    "MedFuse": 0.6546,
}


class NoCTDataset(Dataset):
    """切掉 ct：mask 5 列 → 4 列，inputs 移除 'ct' 键（与 train_eruf 一致）。"""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        inputs, mask, y = self.base[i]
        inputs = {k: v for k, v in inputs.items() if k != "ct"}
        return inputs, mask[1:], y


def make_eruf_encoders(optional: list[str], d: int, dropout: float) -> dict:
    return {
        "structured": MLPEncoder(6, (64, 128), d, dropout),
        "ecg": ResECGWaveformEncoder(latent_dim=d, dropout=dropout),
        "echo": MLPEncoder(3, (32, 64), d, dropout),
        "vitals": MLPEncoder(6, (32, 64), d, dropout),
        "cxr": MLPEncoder(3, (32, 64), d, dropout),
    }


def collect_predictions() -> dict[str, dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    mcfg = load_config("model")["model"]
    d = int(mcfg.get("latent_dim", 128))
    dropout = float(mcfg.get("dropout", 0.2))
    batch = 256

    ext_ds, ext_meta = load_fixed_echonext()
    print(f"EchoNext n={ext_meta['n_external']}  prev={ext_meta['prev_pasp45']:.4f}", flush=True)

    out: dict[str, dict[str, np.ndarray]] = {}

    # ---- ERUF（4 模态，去 ct，rectifier 开启）----
    print("\n=== ERUF ===", flush=True)
    eruf = ERUFModel(make_eruf_encoders(RELEVANT, d, dropout), RELEVANT,
                     d=d, dropout=dropout, use_delta=True).to(device)
    eruf.load_state_dict(torch.load(OUT_DIR / "eruf_full.pt", map_location=device))
    eruf.use_rectifier = True
    eruf.eval()
    y, p = predict_probs(eruf, DataLoader(NoCTDataset(ext_ds), batch_size=batch,
                                          shuffle=False, num_workers=0), device)
    out["ERUF"] = {"y": y, "p": p}
    del eruf
    torch.cuda.empty_cache()

    # ---- 5 基线（5 模态，ct 位恒 0）----
    baselines = {
        "QA-MoE": (QAMoEModel, "qamoe.pt"),
        "DrFuse": (DrFuseModel, "drfuse.pt"),
        "ModDrop": (ModDropContrastiveModel, "moddrop.pt"),
        "MUSE": (MUSEModel, "muse.pt"),
        "MedFuse": (MedFuseModel, "medfuse.pt"),
    }
    for name, (cls, ckpt) in baselines.items():
        print(f"\n=== {name} ===", flush=True)
        enc = make_baseline_encoders(OPTIONAL5, d, dropout)
        model = cls(enc, OPTIONAL5, d=d, dropout=dropout).to(device)
        model.load_state_dict(torch.load(OUT_DIR / ckpt, map_location=device))
        model.eval()
        y, p = predict_probs(model, DataLoader(ext_ds, batch_size=batch,
                                               shuffle=False, num_workers=0), device)
        out[name] = {"y": y, "p": p}
        del model
        torch.cuda.empty_cache()

    return out


def main() -> None:
    preds = collect_predictions()

    # 核对重算 AUROC 与论文 Table 1 是否一致
    print("\n=== 核对 Ext-val AUROC ===", flush=True)
    aucs: dict[str, float] = {}
    for name, dct in preds.items():
        a = float(roc_auc_score(dct["y"], dct["p"]))
        aucs[name] = a
        paper = PAPER_EXT_AUROC.get(name, float("nan"))
        flag = "OK" if abs(a - paper) < 1e-3 else f"MISMATCH (paper={paper})"
        print(f"  [{name}] auc={a:.4f}  {flag}", flush=True)

    # 保存逐样本预测，供复用
    npz_path = OUT_DIR / "ext_roc_predictions.npz"
    np.savez(npz_path,
             **{f"{k}_y": preds[k]["y"] for k in preds},
             **{f"{k}_p": preds[k]["p"] for k in preds})
    print(f"\n已保存逐样本预测 {npz_path}", flush=True)

    # ---- 画 ROC 曲线 ----
    model_order = sorted(aucs, key=lambda m: -aucs[m])   # 按 AUC 降序，ERUF 第一
    colors = {
        "ERUF": "#d62728", "QA-MoE": "#1f77b4", "DrFuse": "#ff7f0e",
        "ModDrop": "#2ca02c", "MUSE": "#9467bd", "MedFuse": "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for name in model_order:
        fpr, tpr, _ = roc_curve(preds[name]["y"], preds[name]["p"])
        is_eruf = name == "ERUF"
        ax.plot(fpr, tpr, color=colors[name], lw=2.8 if is_eruf else 1.4,
                ls="-" if is_eruf else "--", alpha=1.0 if is_eruf else 0.85,
                label=f"{name} (AUC = {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.9, ls=":", zorder=0)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("External validation ROC (EchoNext, zero-shot ECG-only)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)

    # 高亮 ERUF 最优点（Youden 点）
    fpr_e, tpr_e, thr_e = roc_curve(preds["ERUF"]["y"], preds["ERUF"]["p"])
    j = int(np.argmax(tpr_e - fpr_e))
    ax.scatter([fpr_e[j]], [tpr_e[j]], color="#d62728", s=60, zorder=5,
               edgecolor="white", linewidth=1.2)

    fig.tight_layout()
    dst = FIG_DIR / "fig_ext_roc.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {dst} (300 dpi)")
    print(f"saved {dst.with_suffix('.pdf')}")

    # ---- 小图版（供 PPT 作子图，尺寸/字号更小）----
    fig2, ax2 = plt.subplots(figsize=(4.0, 3.6))
    for name in model_order:
        fpr, tpr, _ = roc_curve(preds[name]["y"], preds[name]["p"])
        is_eruf = name == "ERUF"
        ax2.plot(fpr, tpr, color=colors[name], lw=2.4 if is_eruf else 1.1,
                 ls="-" if is_eruf else "--", alpha=1.0 if is_eruf else 0.8,
                 label=f"{name} ({aucs[name]:.3f})")
    ax2.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":", zorder=0)
    ax2.set_xlim([-0.01, 1.01]); ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("False positive rate", fontsize=7.5)
    ax2.set_ylabel("True positive rate", fontsize=7.5)
    ax2.set_title("External validation ROC (EchoNext)", fontsize=8.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="lower right", fontsize=6, frameon=False)
    ax2.tick_params(labelsize=6.5)
    fig2.tight_layout()
    dst2 = FIG_DIR / "fig_ext_roc_small.png"
    fig2.savefig(dst2, dpi=300, bbox_inches="tight")
    fig2.savefig(dst2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {dst2} (300 dpi, small for PPT)")
    print(f"saved {dst2.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
