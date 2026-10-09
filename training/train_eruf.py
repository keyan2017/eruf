"""ERUF（缺失模式感知的证据充分性融合）训练 + 评估。

见 models/eruf.py。方法：掩码拼接 + 缺失模式嵌入（MNAR 感知）+ 证据充分性头 +
后处理 rectifier。训练目标（三路损失）：

    L = L_pred + λ1·L_evidence + λ2·L_recon
      L_pred     = BCE(p, y)                       （风险判别，full + 随机子集）
      L_evidence = BCE(C, y_evidence)              （y_evidence=1[预测正确]，stop-grad）
      L_recon    = Σ_m MSE(recon_m(h), h_m)        （可用模态重建，MNAR 表征稳定）

rectifier g(δ) 为后处理（stage2）：stage1 冻结后，用留出验证集残差拟合 g(δ)，推理时
`logit = f(h) + g(δ)` 做加性偏置校正。

评估强调四点：缺失鲁棒（MCAR ΔAUROC）、校准（ECE）、证据充分性/不确定性识别（C 是否低 C
标记错误样本）、外部泛化（EchoNext）。另输出模态增量价值 Δ_m（派生）诊断。
结果写 experiments/mimic/ph_experiment/eruf_{cfg}_summary.json。

用法：
  python training/train_eruf.py --smoke
  python training/train_eruf.py --config full
  python training/train_eruf.py --config nodelta    # 消融：关 MNAR 感知（δ 嵌入 + rectifier）
  python training/train_eruf.py --config noev       # 消融：关证据充分性损失
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
from torch.utils.data import DataLoader, Dataset  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config import load_config  # noqa: E402
from datasets.ph_experiment import load_fixed_mimic, load_fixed_echonext  # noqa: E402
from evaluation.prediction import predict_probs  # noqa: E402
from evaluation.metrics import auroc, compute_all_metrics  # noqa: E402
from evaluation.missing_modality_eval import evaluate_combination, evaluate_mcar_sweep  # noqa: E402
from models.encoders import MLPEncoder  # noqa: E402
from models.eruf import ERUFModel  # noqa: E402
from models.image_encoders import ResECGWaveformEncoder  # noqa: E402
from training.train_baselines import (  # noqa: E402
    OUT_DIR, external_val, leave_one_out,
)

OPTIONAL = ["ecg", "echo", "vitals", "cxr"]

CONFIGS = {
    "full": dict(n_views=4, p_max=0.75, sub_task=0.5, evidence=0.5, recon=0.1,
                 dropout=0.2, lr=1e-3, wd=1e-4, use_delta=True),
    "nodelta": dict(n_views=4, p_max=0.75, sub_task=0.5, evidence=0.5, recon=0.1,
                    dropout=0.2, lr=1e-3, wd=1e-4, use_delta=False),
    "noev": dict(n_views=4, p_max=0.75, sub_task=0.5, evidence=0.0, recon=0.1,
                 dropout=0.2, lr=1e-3, wd=1e-4, use_delta=True),
    "nosub": dict(n_views=4, p_max=0.75, sub_task=0.0, evidence=0.5, recon=0.1,
                  dropout=0.2, lr=1e-3, wd=1e-4, use_delta=True),
}


class NoCTDataset(Dataset):
    """切掉 ct（恒缺失）模态：mask 5 列 → 4 列，inputs 移除 'ct' 键。"""

    def __init__(self, base: Dataset):
        self.base = base
        self.y = getattr(base, "y", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        inputs, mask, y = self.base[i]
        inputs = {k: v for k, v in inputs.items() if k != "ct"}
        return inputs, mask[1:], y


def make_encoders(optional: list[str], d: int, dropout: float) -> dict[str, nn.Module]:
    return {
        "structured": MLPEncoder(6, (64, 128), d, dropout),
        "ecg": ResECGWaveformEncoder(latent_dim=d, dropout=dropout),
        "echo": MLPEncoder(3, (32, 64), d, dropout),
        "vitals": MLPEncoder(6, (32, 64), d, dropout),
        "cxr": MLPEncoder(3, (32, 64), d, dropout),
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


def _ece(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    acc = np.asarray(acc, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if idx.sum() == 0:
            continue
        ece += (idx.sum() / len(conf)) * abs(conf[idx].mean() - acc[idx].mean())
    return float(ece)


def compute_loss(model, inputs, mask, y, criterion, cfg) -> torch.Tensor:
    """L = L_pred + λ1·L_evidence + λ2·L_recon。"""
    p_full, C_full, p_subs, C_subs, recon_full, H = model.forward_train(
        inputs, mask, n_views=cfg["n_views"], p_max=cfg["p_max"])
    # L_pred：风险判别（full + 子集）
    loss = criterion(p_full, y)
    for ps in p_subs:
        loss = loss + cfg["sub_task"] * criterion(ps, y)
    # L_evidence：证据充分性监督（y_evidence = 1[预测正确]，stop-grad）
    if cfg.get("evidence", 0.0) > 0:
        y_ev_full = (torch.sigmoid(p_full.detach()) > 0.5).float()
        loss = loss + cfg["evidence"] * F.binary_cross_entropy_with_logits(
            C_full, (y_ev_full == y).float())
        for ps, Cs in zip(p_subs, C_subs):
            y_ev = (torch.sigmoid(ps.detach()) > 0.5).float()
            loss = loss + cfg["evidence"] * F.binary_cross_entropy_with_logits(
                Cs, (y_ev == y).float())
    # L_recon：可用模态重建（MNAR 表征稳定）
    if cfg.get("recon", 0.0) > 0:
        rl = 0.0
        for k in model.keys:
            rl = rl + F.mse_loss(recon_full[k], H[k])
        loss = loss + cfg["recon"] * rl
    return loss


def train_one(model, train_ds, val_ds, device, cfg, tr) -> dict:
    tl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0)
    y_tr = train_ds.y
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos if pos > 0 else 1.0, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_auc, best_state, best_ep, cnt = -1.0, None, -1, 0
    for ep in range(1, tr["epochs"] + 1):
        model.train()
        for inputs, mask, y in tl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = compute_loss(model, inputs, mask, y, criterion, cfg)
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


def fit_rectifier(model, val_ds, device, batch_size, epochs=200, lr=1e-3) -> float:
    """stage2：冻结 f，用留出验证集残差 y−σ(f(h)) 拟合 g(δ)（MSE），再开启 rectifier。"""
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    masks, resid = [], []
    with torch.no_grad():
        for inputs, mask, y in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask = mask.to(device)
            y = y.to(device)
            H = model._encode_full(inputs)
            h, _ = model._fuse(H, mask)
            p = model.risk_head(h).squeeze(-1)
            r = y - torch.sigmoid(p)
            masks.append(mask.cpu())
            resid.append(r.cpu())
    masks = torch.cat(masks, dim=0)
    resid = torch.cat(resid, dim=0)
    opt = torch.optim.AdamW(model.rectifier.parameters(), lr=lr, weight_decay=1e-4)
    last = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        g = model.rectifier(masks.to(device)).squeeze(-1)
        loss = F.mse_loss(g, resid.to(device))
        loss.backward()
        opt.step()
        last = float(loss.item())
    model.use_rectifier = True
    return last


@torch.no_grad()
def evidence_diag(model, dataset, device, batch_size) -> dict:
    """证据充分性 C 是否校准、是否低 C 标记错误风险样本（不确定性识别）。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ps, Cs, ys = [], [], []
    for inputs, mask, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask = mask.to(device)
        p_logit, C_logit, _ = model.forward_outputs(inputs, mask)
        ps.append(torch.sigmoid(p_logit).cpu().numpy())
        Cs.append(torch.sigmoid(C_logit).cpu().numpy())
        ys.append(y.numpy())
    p = np.concatenate(ps)
    C = np.concatenate(Cs)
    y = np.concatenate(ys).astype(float)
    pred = (p > 0.5).astype(float)
    correct = (pred == y).astype(float)
    wrong = 1.0 - correct
    ece_C = _ece(C, correct)
    err_auroc = float(auroc(wrong, 1.0 - C)) if (wrong.min() < wrong.max()) else float("nan")
    return {
        "C_mean": float(C.mean()),
        "C_correct_mean": float(C[correct == 1].mean()) if (correct == 1).any() else None,
        "C_wrong_mean": float(C[wrong == 1].mean()) if (wrong == 1).any() else None,
        "ece_C": ece_C,
        "err_det_auroc": err_auroc,
    }


@torch.no_grad()
def derived_voi_diag(model, dataset, device, batch_size) -> dict:
    """模态增量价值（派生）：Δ_m = C(S∪{m}) − C(S)，对缺失样本求均值。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    opt = model.optional
    acc = {k: [] for k in opt}
    cnt = {k: 0 for k in opt}
    for inputs, mask, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask = mask.to(device)
        D = model._derived_voi(model._encode_full(inputs), mask)  # (B, M)
        m = mask.cpu().numpy()
        Dn = D.cpu().numpy()
        for j, k in enumerate(opt):
            sel = m[:, j] <= 0.5
            if sel.any():
                acc[k].append(Dn[sel, j])
                cnt[k] += int(sel.sum())
    out = {}
    for k in opt:
        if acc[k]:
            v = np.concatenate(acc[k])
            out[f"delta_{k}"] = float(v.mean())
            out[f"delta_{k}_n"] = int(cnt[k])
        else:
            out[f"delta_{k}"] = None
            out[f"delta_{k}_n"] = 0
    return out


def main() -> None:
    smoke = "--smoke" in sys.argv
    cfg_name = "full"
    for _i, _a in enumerate(sys.argv):
        if _a == "--config" and _i + 1 < len(sys.argv):
            cfg_name = sys.argv[_i + 1]
    cfg = dict(CONFIGS[cfg_name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    mcfg = load_config("model")["model"]
    d = int(mcfg.get("latent_dim", 128))
    cfg["dropout"] = cfg.get("dropout", float(mcfg.get("dropout", 0.2)))
    trcfg = load_config("training")["training"]
    tr = {
        "epochs": 2 if smoke else int(trcfg["epochs"]),
        "batch_size": int(trcfg["batch_size"]),
        "patience": 1 if smoke else int(trcfg["early_stopping"]["patience"]),
    }

    train_ds, val_ds, test_ds, meta = load_fixed_mimic()
    train_ds, val_ds, test_ds = NoCTDataset(train_ds), NoCTDataset(val_ds), NoCTDataset(test_ds)
    print(f"固定训练集：train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}  "
          f"prev={meta['prev']:.4f}  L={meta['L']}  (已去除 ct，optional={OPTIONAL})")
    combos = _build_combos(OPTIONAL)
    full_mask_vec = [1] * len(OPTIONAL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    use_delta = cfg.get("use_delta", True)
    model = ERUFModel(make_encoders(OPTIONAL, d, cfg["dropout"]), OPTIONAL, d=d,
                      dropout=cfg["dropout"], use_delta=use_delta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== ERUF config={cfg_name} params={n_params:,} ===", flush=True)
    print(f"    cfg={cfg}", flush=True)

    resume = "--resume" in sys.argv
    if resume:
        ckpt = OUT_DIR / f"eruf_{cfg_name}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        res = {"val_auroc": None, "best_epoch": None}
        print(f"[resume] 已加载 {ckpt}，跳过训练，仅重跑评估", flush=True)
    else:
        res = train_one(model, train_ds, val_ds, device, cfg, tr)
        model.eval()

    # stage2：拟合 rectifier（仅 use_delta=True 时）
    rect_loss = None
    if use_delta and not smoke:
        rect_loss = fit_rectifier(model, val_ds, device, tr["batch_size"])
        print(f"  rectifier 拟合完成（val 残差 MSE={rect_loss:.4f}），已开启加性校正", flush=True)

    # 保存 checkpoint（含已拟合的 rectifier，供外部验证重载）
    torch.save(model.state_dict(), OUT_DIR / f"eruf_{cfg_name}.pt")

    rec = {"config": cfg, "n_params": n_params,
           "val_auroc": float(res["val_auroc"]) if res["val_auroc"] is not None else None,
           "best_epoch": int(res["best_epoch"]) if res["best_epoch"] is not None else None,
           "rectifier_val_mse": rect_loss}
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
    rec["evidence"] = evidence_diag(model, test_ds, device, tr["batch_size"])
    rec["voi"] = derived_voi_diag(model, test_ds, device, tr["batch_size"])

    print(f"  full_auroc={rec['full_metrics']['auroc']:.4f}  ece={rec['full_metrics']['ece']:.4f}  "
          f"mcar_p0={rec['mcar_sweep'].get('p=0'):.4f}  p0.75={rec['mcar_sweep'].get('p=0.75'):.4f}  "
          f"delta={rec['mcar_sweep'].get('p=0', 0) - rec['mcar_sweep'].get('p=0.75', 0):.4f}",
          flush=True)
    print(f"  evidence: C_mean={rec['evidence']['C_mean']:.4f}  ece_C={rec['evidence']['ece_C']:.4f}  "
          f"err_det_auroc={rec['evidence']['err_det_auroc']:.4f}  "
          f"C_correct={rec['evidence']['C_correct_mean']:.4f}  C_wrong={rec['evidence']['C_wrong_mean']:.4f}",
          flush=True)
    print(f"  voi(derived): { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec['voi'].items()} }",
          flush=True)
    del model
    gc.collect()

    # 外部验证（EchoNext，仅 ECG 零样本）
    print("\n=== 外部验证 EchoNext（仅 ECG 零样本）===", flush=True)
    ext_ds, ext_meta = load_fixed_echonext()
    ext_ds = NoCTDataset(ext_ds)
    print(f"  外部集：n={ext_meta['n_external']}  pasp>=45 prev={ext_meta['prev_pasp45']:.4f}")
    model = ERUFModel(make_encoders(OPTIONAL, d, cfg["dropout"]), OPTIONAL, d=d,
                      dropout=cfg["dropout"], use_delta=use_delta).to(device)
    model.load_state_dict(torch.load(OUT_DIR / f"eruf_{cfg_name}.pt", map_location=device))
    model.use_rectifier = use_delta and not smoke
    rec["external"] = {"meta": ext_meta, "metrics": external_val(model, ext_ds, device)}
    print(f"  ERUF({cfg_name}) EchoNext(pasp>=45) AUROC={rec['external']['metrics']['auroc']:.4f}", flush=True)

    out = {"is_synthetic": False, "model": f"ERUF({cfg_name})", "config": cfg, "result": rec}
    (OUT_DIR / f"eruf_{cfg_name}_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n结果已保存: {OUT_DIR / f'eruf_{cfg_name}_summary.json'}")


if __name__ == "__main__":
    main()
