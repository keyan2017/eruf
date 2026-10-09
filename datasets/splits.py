"""数据划分与泄漏检查。

核心原则：患者级划分 —— 同一患者的全部访视必须落在同一划分内，
严禁同一患者的不同访视同时进入 train/test。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int = 42,
    stratify_by: Optional[str] = None,
    patient_col: str = "patient_id",
) -> dict[str, pd.DataFrame]:
    """按患者级划分数据集。

    stratify_by: 可选的标签列（患者级标签，如该患者任意访视的 ph_label），
    用于尽量平衡类别。返回 {name: 该划分对应的访视 DataFrame}。
    """
    patients = df[patient_col].unique().tolist()
    rng = np.random.default_rng(seed)

    if stratify_by is not None:
        # 患者级标签 = 任一访视阳性即记为阳性
        patient_label = (
            df.groupby(patient_col)[stratify_by].max().reindex(patients).fillna(0).astype(int).to_numpy()
        )
        pos = [p for p, lab in zip(patients, patient_label) if lab == 1]
        neg = [p for p, lab in zip(patients, patient_label) if lab == 0]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        neg = [neg[i] for i in rng.permutation(len(neg))]
        pos_splits = _split_indices(len(pos), ratios)
        neg_splits = _split_indices(len(neg), ratios)
        assign: dict[str, list[str]] = {name: [] for name in ratios}
        for name in ratios:
            assign[name].extend([pos[i] for i in pos_splits[name]])
            assign[name].extend([neg[i] for i in neg_splits[name]])
    else:
        shuffled = [patients[i] for i in rng.permutation(len(patients))]
        idx = _split_indices(len(patients), ratios)
        assign = {name: [shuffled[i] for i in idx[name]] for name in ratios}

    return {
        name: df[df[patient_col].isin(assign[name])].reset_index(drop=True)
        for name in ratios
    }


def _split_indices(n: int, ratios: dict[str, float]) -> dict[str, list[int]]:
    """把 0..n-1 按 ratios 顺序切分（不重叠），返回各划分的索引列表。"""
    total = sum(ratios.values())
    bounds: dict[str, list[int]] = {}
    names = list(ratios.keys())
    start = 0
    for k, name in enumerate(names):
        if k == len(names) - 1:
            end = n  # 最后一段拿走余数
        else:
            end = start + int(round(n * ratios[name] / total))
        bounds[name] = list(range(start, end))
        start = end
    return bounds


def check_patient_overlap(
    *frames: pd.DataFrame, patient_col: str = "patient_id"
) -> dict[str, object]:
    """检查多个划分之间是否存在患者重叠（数据泄漏检查器之一）。"""
    result: dict[str, object] = {"overlap": False, "details": []}
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a = set(frames[i][patient_col].unique())
            b = set(frames[j][patient_col].unique())
            inter = a & b
            if inter:
                result["overlap"] = True
                result["details"].append({"split_pair": (i, j), "n_overlap": len(inter)})
    return result


def check_future_label_leakage(
    df: pd.DataFrame,
    label_col: str = "cor_pulmonale_label",
    patient_col: str = "patient_id",
    visit_col: str = "visit_index",
) -> dict[str, object]:
    """检查时序标签泄漏：预测用特征是否包含未来标签（简单静态检查）。"""
    n_issue = 0
    for _, g in df.groupby(patient_col):
        g = g.sort_values(visit_col)
        # 若早期访视已含晚期阳性标签而晚期尚未发生，说明标签定义跨时间泄漏
        # 此处做保守提示：报告患者级标签是否随时间单调（不强制）
        labels = g[label_col].tolist()
        if any(labels) and not labels[-1]:
            n_issue += 1
    return {"suspicious_non_monotonic_label_patients": n_issue}
