"""缺失模态性能矩阵评估（第 15 节要求）。

对同一个（缺失模态鲁棒）模型，在多种模态可用组合下评估其性能，
回答：“模型在不同临床可用数据条件下是否仍然稳定？”

组合（mask 顺序 [ct, ecg, echo]）：
  no_imaging / ct / ecg / echo / ct+ecg / ct+echo / ecg+echo / ct+ecg+echo
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import compute_all_metrics

# 模态组合（name, mask vector）
COMBINATIONS: list[tuple[str, list[int]]] = [
    ("no_imaging", [0, 0, 0]),
    ("ct", [1, 0, 0]),
    ("ecg", [0, 1, 0]),
    ("echo", [0, 0, 1]),
    ("ct_ecg", [1, 1, 0]),
    ("ct_echo", [1, 0, 1]),
    ("ecg_echo", [0, 1, 1]),
    ("ct_ecg_echo", [1, 1, 1]),
]


@torch.no_grad()
def evaluate_combination(
    model: torch.nn.Module,
    dataset,
    mask_vec: list[int],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """在“强制该模态组合”的条件下评估模型。

    只选取该组合所需模态均真实可用的样本，然后把其它模态特征置零并强制 mask。
    返回 (y_true, y_prob)。mask 长度从 model.optional 读取（可变）。
    """
    model.eval()
    optional = model.optional
    n_opt = len(optional)
    vec = np.array(mask_vec, dtype=np.float32)
    required = [j for j in range(n_opt) if vec[j] == 1]

    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for inputs, mask, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask = mask.to(device)
        y = y.to(device)
        ok = torch.ones(mask.shape[0], dtype=torch.bool, device=device)
        for j in required:
            ok &= (mask[:, j] == 1)
        if not ok.any():
            continue

        forced = torch.zeros_like(mask)
        for j in range(n_opt):
            if vec[j] == 1:
                forced[:, j] = 1.0
        # 非组合内模态特征置零（视为缺失）；structured 不受影响
        sub = {k: v[ok] for k, v in inputs.items()}
        for j, k in enumerate(optional):
            sub[k] = sub[k] * vec[j]

        logits = model(sub, forced[ok])
        ys.append(y[ok].cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


@torch.no_grad()
def evaluate_mcar_sweep(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    dropout_rates: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7),
    seed: int = 42,
    batch_size: int = 256,
    full_mask_vec: list[int] | None = None,
) -> dict[str, float]:
    """MCAR（随机缺失）压力测试：在全模态样本上按丢弃率随机置缺，度量 AUROC 退化。

    全模态基 = full_mask_vec（默认全 1；MIMIC 用 [0,1,1,1,1]，ct 恒缺）。对每个丢弃率
    p，用固定 seed 的 rng 把每个「本就可用的」可选模态以概率 p 随机置 0（特征 + mask
    一致），计算 AUROC。回答：「从 MNAR 医院数据训练 → 体检 MCAR 场景部署」的鲁棒性。
    样本量 = 全模态子集（可能较小），如实标注。
    """
    from evaluation.metrics import auroc

    optional = model.optional
    n_opt = len(optional)
    full_vec = np.asarray(
        full_mask_vec if full_mask_vec is not None else [1] * n_opt, dtype=np.float32)

    # 收集全模态样本（自然 mask == full_vec）
    xs: dict[str, list[np.ndarray]] = {k: [] for k in model.keys}
    ys: list[np.ndarray] = []
    for inputs, mask, y in DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=0):
        keep = torch.all(mask == torch.as_tensor(full_vec, dtype=mask.dtype), dim=1)
        if not keep.any():
            continue
        keep = keep.cpu().numpy()
        for k in model.keys:
            xs[k].append(inputs[k][keep].numpy())
        ys.append(y[keep].numpy())

    if not ys:
        return {}

    base = {k: np.concatenate(xs[k], axis=0) for k in model.keys}
    y_all = np.concatenate(ys, axis=0)
    n = len(y_all)

    out: dict[str, float] = {}
    for p in dropout_rates:
        rng = np.random.default_rng(seed)
        feats = {k: v.copy() for k, v in base.items()}
        mask_arr = np.tile(full_vec, (n, 1))
        for j, k in enumerate(optional):
            if full_vec[j] == 0:
                continue
            drop = rng.random(n) < p
            mask_arr[drop, j] = 0.0
            feats[k][drop] = 0.0

        preds: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            inp = {k: torch.as_tensor(feats[k][sl], dtype=torch.float32, device=device)
                   for k in model.keys}
            m = torch.as_tensor(mask_arr[sl], dtype=torch.float32, device=device)
            preds.append(torch.sigmoid(model(inp, m)).cpu().numpy())
        out[f"p={p:g}"] = auroc(y_all, np.concatenate(preds))
    return out


def evaluate_matrix(model, dataset, device) -> dict[str, dict]:
    """返回 {组合名: 完整指标 dict}。"""
    out: dict[str, dict] = {}
    for name, vec in COMBINATIONS:
        y, p = evaluate_combination(model, dataset, vec, device)
        out[name] = compute_all_metrics(y, p)
    return out


def matrix_auroc(results: dict[str, dict]) -> dict[str, float]:
    """从矩阵结果中抽取 AUROC（便于打印/保存）。"""
    return {name: results[name]["auroc"] for name in results}
