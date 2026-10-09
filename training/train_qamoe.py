"""QA-MoE 基线训练 + 缺失模态性能评估（固定 PH 实验数据集上）。

与论文其余四个基线（DrFuse / MedFuse / MUSE / ModDrop，见 train_baselines.py）同协议
（同 backbone、同 5 模态含恒缺 ct、同 seed/划分/评估），可直接与 baselines_summary.json
对比。图像/生存相关方法不做对比。

  QA-MoE —— 质量感知 + 稳定专家混合路由（IJCAI 2026）：Evidential Quality Scorer 出
            每模态可靠性 → 门控专家路由 + 严重缺失 fail-safe。loss=BCE + EDL 正则。

用法：
  python training/train_qamoe.py --smoke
  python training/train_qamoe.py --variants qamoe
"""
from __future__ import annotations

import gc
import json
import sys

sys.path.insert(0, str(__file__ and __import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config import load_config  # noqa: E402
from datasets.ph_experiment import load_fixed_mimic, load_fixed_echonext, OPTIONAL  # noqa: E402
from evaluation.prediction import predict_probs  # noqa: E402
from evaluation.metrics import auroc, compute_all_metrics  # noqa: E402
from evaluation.missing_modality_eval import evaluate_combination, evaluate_mcar_sweep  # noqa: E402
from models.qamoe import QAMoEModel  # noqa: E402
from models.base import make_baseline_encoders  # noqa: E402
from training.train_baselines import (  # noqa: E402
    OUT_DIR, _build_combos, external_val, leave_one_out,
)

BASELINES = {
    "qamoe": dict(cls=QAMoEModel, kind="qamoe"),
}


def _edl_loss(alpha: torch.Tensor, y: torch.Tensor, K: int = 2) -> torch.Tensor:
    """Dirichlet Type-II ML（Bayes risk）正则：证据拟合 ground-truth 并惩罚不确定性。"""
    y_onehot = F.one_hot(y.long(), num_classes=K).float()
    S = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p = alpha / S
    err = (y_onehot - p) ** 2
    var = p * (1.0 - p) / (S + 1.0)
    return (err + var).sum(dim=-1).mean()


def compute_loss(model, inputs, mask, y, criterion, kind, lam) -> torch.Tensor:
    if kind == "qamoe":
        logits, g, alpha, _S = model.forward_aux(inputs, mask)
        loss = criterion(logits, y)
        if lam.get("edl", 0.0) > 0:
            global_alpha = (g.unsqueeze(-1) * alpha).sum(dim=1)   # 门控聚合的全局证据 (B,K)
            loss = loss + lam["edl"] * _edl_loss(global_alpha, y, K=model.K)
        return loss
    raise ValueError(kind)


def train_one(model, train_ds, val_ds, device, kind, lam, tr) -> dict:
    tl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0)
    y_tr = train_ds.y
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos if pos > 0 else 1.0, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["wd"])
    best_auc, best_state, best_ep, cnt = -1.0, None, -1, 0
    for ep in range(1, tr["epochs"] + 1):
        model.train()
        for inputs, mask, y in tl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = compute_loss(model, inputs, mask, y, criterion, kind, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        yv, pv = predict_probs(model, vl, device)
        va = auroc(yv, pv)
        va = 0.0 if np.isnan(va) else va
        if va > best_auc:
            best_auc, best_ep, cnt = va, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
            if cnt >= tr["patience"]:
                break
        if ep % 5 == 0 or ep == 1:
            print(f"    epoch={ep} val_auroc={va:.4f} best={best_auc:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_ep, "val_auroc": best_auc}


def _make_model(name: str, d: int, dropout: float, device: torch.device):
    cfg = BASELINES[name]
    encoders = make_baseline_encoders(OPTIONAL, d, dropout)
    return cfg["cls"](encoders, OPTIONAL, d=d, dropout=dropout).to(device)


def main() -> None:
    smoke = "--smoke" in sys.argv
    only = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--variants" and _i + 1 < len(sys.argv):
            only = {v.strip() for v in sys.argv[_i + 1].split(",")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    mcfg = load_config("model")["model"]
    d = int(mcfg.get("latent_dim", 128))
    dropout = float(mcfg.get("dropout", 0.2))
    trcfg = load_config("training")["training"]
    tr = {
        "epochs": 2 if smoke else int(trcfg["epochs"]),
        "batch_size": int(trcfg["batch_size"]),
        "lr": float(trcfg["optimizer"]["lr"]),
        "wd": float(trcfg["optimizer"]["weight_decay"]),
        "patience": 1 if smoke else int(trcfg["early_stopping"]["patience"]),
    }
    lam = {"edl": 0.1}

    train_ds, val_ds, test_ds, meta = load_fixed_mimic()
    print(f"固定训练集：train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}  "
          f"prev={meta['prev']:.4f}  L={meta['L']}")
    combos = _build_combos(OPTIONAL)
    full_mask_vec = [0 if k == "ct" else 1 for k in OPTIONAL]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {"is_synthetic": False, "data": "fixed ph_experiment dataset (MIMIC train + EchoNext external)",
           "label": "ph_label (echo-PH proxy)",
           "note": "QA-MoE core-mechanism reproduction",
           "baselines": {}}

    for name, cfg in BASELINES.items():
        if only and name not in only:
            continue
        print(f"\n=== 基线 [{name}] kind={cfg['kind']} ===", flush=True)
        model = _make_model(name, d, dropout, device)
        n_params = sum(p.numel() for p in model.parameters())
        res = train_one(model, train_ds, val_ds, device, cfg["kind"], lam, tr)
        model.eval()
        torch.save(model.state_dict(), OUT_DIR / f"{name}.pt")

        rec = {"n_params": n_params, "val_auroc": float(res["val_auroc"]),
               "best_epoch": int(res["best_epoch"])}
        rec["full_metrics"] = compute_all_metrics(
            *predict_probs(model, DataLoader(test_ds, batch_size=tr["batch_size"],
                                             shuffle=False, num_workers=0), device))
        rec["mcar_sweep"] = evaluate_mcar_sweep(
            model, test_ds, device, dropout_rates=(0.0, 0.25, 0.5, 0.75), seed=42,
            batch_size=tr["batch_size"], full_mask_vec=full_mask_vec)
        matrix = {}
        for cname, vec in combos:
            y, p = evaluate_combination(model, test_ds, vec, device, tr["batch_size"])
            matrix[cname] = float(auroc(y, p)) if len(y) > 0 else None
        rec["matrix"] = matrix
        rec["leave_one_out"] = leave_one_out(model, test_ds, device, tr["batch_size"])

        out["baselines"][name] = rec
        print(f"  [{name}] params={n_params:,} full_auroc={rec['full_metrics']['auroc']:.4f}  "
              f"mcar_p0.75={rec['mcar_sweep'].get('p=0.75'):.4f}  "
              f"delta={rec['mcar_sweep'].get('p=0', 0) - rec['mcar_sweep'].get('p=0.75', 0):.4f}",
              flush=True)
        del model
        gc.collect()

    # 外部验证（EchoNext，仅 ECG）
    print("\n=== 外部验证 EchoNext（仅 ECG 零样本）===", flush=True)
    ext_ds, ext_meta = load_fixed_echonext()
    print(f"  外部集：n={ext_meta['n_external']}  pasp>=45 prev={ext_meta['prev_pasp45']:.4f}  "
          f"tr>=3.2 prev={ext_meta['prev_tr32']:.4f}")
    out["external"] = {"meta": ext_meta, "baselines": {}}
    for name in out["baselines"]:
        ckpt = OUT_DIR / f"{name}.pt"
        model = _make_model(name, d, dropout, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        out["external"]["baselines"][name] = external_val(model, ext_ds, device)
        print(f"  [{name}] EchoNext(pasp>=45) AUROC={out['external']['baselines'][name]['auroc']:.4f}")
        del model

    (OUT_DIR / "qamoe_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n结果已保存: {OUT_DIR / 'qamoe_summary.json'}")


if __name__ == "__main__":
    main()
