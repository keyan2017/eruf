"""模型评价指标。

PH 早筛 / 二分类：AUROC / AUPRC / Sensitivity / Specificity / Accuracy / F1 /
PPV / NPV / Brier / ECE(校准)。
动态风险预测的 C-index 等在时序模型阶段（Phase 7）补充。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """在给定阈值下的混淆矩阵相关指标。"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error（等频分箱）。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return float("nan")
    # 等频分箱（每箱样本数尽量相等）
    order = np.argsort(y_prob)
    n = len(y_true)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if lo >= hi:
            continue
        idx = order[lo:hi]
        frac_pos = y_true[idx].mean()
        mean_conf = y_prob[idx].mean()
        ece_val += (hi - lo) / n * abs(mean_conf - frac_pos)
    return float(ece_val)


def calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """校准曲线数据（用于绘图 / 分析）。"""
    if len(np.unique(y_true)) < 2:
        return {"fraction_positive": [], "mean_predicted": []}
    fp, mp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {"fraction_positive": fp.tolist(), "mean_predicted": mp.tolist()}


def compute_all_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """一次性计算全套早筛指标。"""
    out: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "ece": ece(y_true, y_prob),
        "calibration": calibration_curve_data(y_true, y_prob),
    }
    out.update(threshold_metrics(y_true, y_prob, threshold=threshold))
    return out


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index（时序风险，Phase 7）。

    risk : 预测风险（越大 -> 越早发生事件）
    time : 到事件或删失的时间
    event: 1=观察到事件，0=删失
    """
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(risk)
    conc = 0.0
    total = 0
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j or time[j] <= time[i]:
                continue
            total += 1
            if risk[i] > risk[j]:
                conc += 1.0
            elif risk[i] == risk[j]:
                conc += 0.5
    return float(conc / total) if total else float("nan")
