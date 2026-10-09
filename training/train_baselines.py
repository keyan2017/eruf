"""基线训练 + 缺失模态性能评估（固定 PH 实验数据集上）。

四个基线（核心机制复现，同 backbone 同数据同划分）：
  DrFuse (AAAI 2024)   —— 共享/特有解耦 + 注意力融合
  MedFuse (MLHC 2022)  —— LSTM 融合可变长模态 token 序列
  MUSE   (ICLR 2024)   —— 互一致对比（有监督同标签 + 无监督跨视角）
  ModDrop (2021)       —— 可学习缺失 token + 同步模态丢弃 + 对比对齐

缺失模态性能评估（核心输出）：
  1. 自然缺失 full AUROC/AUPRC/Brier/ECE
  2. MCAR 扫描（p=0→0.75 的 ΔAUROC）
  3. leave-one-out（逐模态 do(M_m=0)）
  4. 外部验证（EchoNext，仅 ECG 可用，零样本跨机构）

用法：
  python training/train_baselines.py --smoke
  python training/train_baselines.py --variants drfuse,medfuse
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from core.config import PROJECT_ROOT, load_config  # noqa: E402
from datasets.ph_experiment import load_fixed_mimic, load_fixed_echonext, OPTIONAL  # noqa: E402
from evaluation.prediction import predict_probs  # noqa: E402
from evaluation.metrics import auroc, compute_all_metrics  # noqa: E402
from evaluation.missing_modality_eval import evaluate_combination, evaluate_mcar_sweep  # noqa: E402
from models.medfuse import MedFuseModel  # noqa: E402
from models.muse import MUSEModel, muse_mutual_consistent_loss  # noqa: E402
from models.drfuse import DrFuseModel  # noqa: E402
from models.moddrop import ModDropContrastiveModel  # noqa: E402
from models.base import make_baseline_encoders  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment"

BASELINES = {
    "drfuse": dict(cls=DrFuseModel, kind="bce"),
    "medfuse": dict(cls=MedFuseModel, kind="bce"),
    "muse": dict(cls=MUSEModel, kind="muse"),
    "moddrop": dict(cls=ModDropContrastiveModel, kind="moddrop"),
}


def _build_combos(optional: list[str]) -> list[tuple[str, list[int]]]:
    idx = {k: j for j, k in enumerate(optional)}

    def vec(*names: str) -> list[int]:
        v = [0] * len(optional)
        for n in names:
            v[idx[n]] = 1
        return v

    return [
        ("structured", vec()),
        ("ecg", vec("ecg")),
        ("echo", vec("echo")),
        ("vitals", vec("vitals")),
        ("cxr", vec("cxr")),
        ("ecg+echo", vec("ecg", "echo")),
        ("ecg+echo+vitals", vec("ecg", "echo", "vitals")),
        ("all", vec("ecg", "echo", "vitals", "cxr")),
    ]


def _random_drop(mask: torch.Tensor, p: float) -> torch.Tensor:
    m = mask.clone()
    keep = (torch.rand_like(m) >= p).float()
    return m * keep


def _infonce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.t() / tau
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0


def compute_loss(model, inputs, mask, y, criterion, kind, lam) -> torch.Tensor:
    if kind == "muse":
        logits, z = model.forward_z(inputs, mask)
        loss = criterion(logits, y)
        if lam["sup"] > 0:
            lsup, _ = muse_mutual_consistent_loss(z, y, tau=lam["tau"])
            loss = loss + lam["sup"] * lsup
        if lam["unsup"] > 0:
            m1 = _random_drop(mask, lam["drop"])
            m2 = _random_drop(mask, lam["drop"])
            _, z1 = model.forward_z(inputs, m1)
            _, z2 = model.forward_z(inputs, m2)
            loss = loss + lam["unsup"] * _infonce(z1, z2, lam["tau"])
        return loss
    if kind == "moddrop":
        logits, z, H = model.forward_loss(inputs, mask)
        loss = criterion(logits, y)
        if lam["align"] > 0:
            m2 = _random_drop(mask, lam["drop"])
            _, z2, _ = model.forward_loss(inputs, m2)
            loss = loss + lam["align"] * _infonce(z, z2, lam["tau"])
        return loss
    # bce (drfuse / medfuse)
    return criterion(model(inputs, mask), y)


def _make_model(name: str, d: int, dropout: float, device: torch.device):
    cfg = BASELINES[name]
    encoders = make_baseline_encoders(OPTIONAL, d, dropout)
    return cfg["cls"](encoders, OPTIONAL, d=d, dropout=dropout).to(device)


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


@torch.no_grad()
def leave_one_out(model, dataset, device, batch_size=256) -> dict[str, float]:
    model.eval()
    optional = model.optional
    y_full, p_full = predict_probs(model, DataLoader(dataset, batch_size=batch_size,
                                                     shuffle=False, num_workers=0), device)
    out = {"full": float(auroc(y_full, p_full))}
    for j, k in enumerate(optional):
        ys, ps = [], []
        for inputs, mask, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            avail_cpu = mask[:, j] > 0.5
            if not avail_cpu.any():
                continue
            inputs = {m: v.to(device) for m, v in inputs.items()}
            mask = mask.to(device)
            avail = mask[:, j] > 0.5
            m_j = mask[avail].clone()
            m_j[:, j] = 0.0
            sub = {m: v[avail] for m, v in inputs.items()}
            sub[k] = sub[k] * 0.0
            ys.append(y[avail_cpu].numpy())
            ps.append(torch.sigmoid(model(sub, m_j)).cpu().numpy())
        if ys:
            out[k] = float(auroc(np.concatenate(ys), np.concatenate(ps)))
    return out


def external_val(model, ext_ds, device, batch_size=64) -> dict:
    model.eval()
    y, p = predict_probs(model, DataLoader(ext_ds, batch_size=batch_size, shuffle=False,
                                           num_workers=0), device)
    m = compute_all_metrics(y, p)
    return {"auroc": m["auroc"], "auprc": m["auprc"]}


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
    lam = {"sup": 0.5, "unsup": 0.5, "align": 0.5, "drop": 0.3, "tau": 0.1}

    train_ds, val_ds, test_ds, meta = load_fixed_mimic()
    print(f"固定训练集：train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}  "
          f"prev={meta['prev']:.4f}  L={meta['L']}")
    combos = _build_combos(OPTIONAL)
    full_mask_vec = [0 if k == "ct" else 1 for k in OPTIONAL]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {"is_synthetic": False, "data": "fixed ph_experiment dataset (MIMIC train + EchoNext external)",
           "label": "ph_label (echo-PH proxy)", "baselines": {}}

    for name, cfg in BASELINES.items():
        if only and name not in only:
            continue
        print(f"\n=== 基线 [{name}] kind={cfg['kind']} ===", flush=True)
        model = _make_model(name, d, dropout, device)
        n_params = sum(p.numel() for p in model.parameters())
        res = train_one(model, train_ds, val_ds, device, cfg["kind"], lam, tr)
        model.eval()
        torch.save(model.state_dict(), OUT_DIR / f"{name}.pt")  # 训练完立即存，eval 崩溃不浪费

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
        if not ckpt.exists():
            continue
        model = _make_model(name, d, dropout, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        out["external"]["baselines"][name] = external_val(model, ext_ds, device)
        print(f"  [{name}] EchoNext(pasp>=45) AUROC={out['external']['baselines'][name]['auroc']:.4f}")
        del model

    (OUT_DIR / "baselines_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n结果已保存: {OUT_DIR / 'baselines_summary.json'}")


if __name__ == "__main__":
    main()
