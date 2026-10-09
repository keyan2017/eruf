"""多模态数据集 / 数据模块（PyTorch）。

Phase 3：提供单模态（或结构化组）baseline 的数据准备：
  - 患者级划分（复用 datasets.splits）
  - 按模态可用性过滤（单模态 baseline 仅在“该模态可用”的访视上训练/评估）
  - train-only 标准化（防止预处理用 test 统计量）
  - 输入列排除血流动力学/治疗（标签泄漏防护）

Phase 4~5 在此扩展多模态融合与缺失模态鲁棒输入。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.constants import FEATURE_GROUPS, IMAGING_MODALITIES, OPTIONAL_MODALITIES
from datasets.splits import patient_level_split


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def group_columns(group: str) -> list[str]:
    return list(FEATURE_GROUPS[group])


def input_columns(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return cols


def available_mask(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """返回各访视“所选组全部可用”的布尔掩码。

    结构化组（clinical/biochemical）在 synthetic v1 中默认可用；
    成像组则要求 avail_<mod> == 1。
    """
    mask = np.ones(len(df), dtype=bool)
    for g in groups:
        if g in IMAGING_MODALITIES:
            mask &= (df[f"avail_{g}"].to_numpy() == 1)
    return mask


def standardize(X: np.ndarray, mean: Optional[np.ndarray] = None,
                std: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准化。若未给 mean/std，则从 X 拟合（仅用于训练集）。"""
    if mean is None:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0) + 1e-8
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs, mean, std


class TabularDataset(Dataset):
    """把 (特征, 标签, 元信息) 包装为 PyTorch Dataset。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, meta: list[tuple[str, str]]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.meta[i]


def build_split_dataframes(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int = 42,
    stratify_by: str = "ph_label",
) -> dict[str, pd.DataFrame]:
    """患者级划分。"""
    return patient_level_split(df, ratios=ratios, seed=seed, stratify_by=stratify_by)


def prepare_single_modality(
    df: pd.DataFrame,
    groups: list[str],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, object]:
    """为单模态/结构化组 baseline 准备 train/val/test。

    返回 dict:
      - datasets: {split: TabularDataset}
      - scaler: (mean, std)   # 仅由训练集拟合
      - columns: 输入列名
      - n_input: 输入维度
    """
    sub = df[available_mask(df, groups)].reset_index(drop=True)
    splits = patient_level_split(sub, ratios=ratios, seed=seed, stratify_by="ph_label")

    cols = input_columns(groups)
    label_col = "ph_label"

    X_train, mean, std = standardize(splits["train"][cols].to_numpy(dtype=np.float32))
    datasets: dict[str, TabularDataset] = {}
    for name in ["train", "val", "test"]:
        sdf = splits[name]
        X, _, _ = standardize(sdf[cols].to_numpy(dtype=np.float32), mean, std)
        y = sdf[label_col].to_numpy(dtype=np.float32)
        meta = list(zip(sdf["patient_id"], sdf["visit_id"]))
        datasets[name] = TabularDataset(X, y, meta)

    return {
        "datasets": datasets,
        "splits": splits,
        "scaler": (mean, std),
        "columns": cols,
        "n_input": len(cols),
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
    }


# =====================================================================
# 多模态（缺失模态鲁棒）数据准备
# =====================================================================

# 模态组 -> 底层特征组。structured（临床+生化）作为“始终可用”的锚。
MULTIMODAL_GROUPS: dict[str, list[str]] = {
    "structured": ["clinical", "biochemical"],
    "ct": ["ct"],
    "ecg": ["ecg"],
    "echo": ["echo"],
    "vitals": ["vitals"],
    "cxr": ["cxr"],
}
MULTIMODAL_KEYS = list(MULTIMODAL_GROUPS.keys())          # 顺序固定（全 6 模态参考）
IMAGING_KEYS = ["ct", "ecg", "echo"]                      # 合成数据默认可选模态


class MultimodalDataset(Dataset):
    """多模态样本数据集。

    __getitem__ 返回 (inputs_dict, mask, y)。inputs_dict 键 = ["structured", *optional]，
    mask 顺序 = optional（默认 [ct, ecg, echo]）。train=True 时按 mask_prob 随机丢弃
    可用可选模态（A4）。
    """

    def __init__(
        self,
        features: dict[str, np.ndarray],
        mask: np.ndarray,
        y: np.ndarray,
        train: bool = False,
        mask_prob: float = 0.3,
        seed: int = 0,
        optional: list[str] | None = None,
    ):
        self.optional = list(optional) if optional is not None else list(IMAGING_KEYS)
        self.keys = ["structured", *self.optional]
        self.features = features
        self.mask = mask.astype(np.float32)
        self.y = y.astype(np.float32)
        self.train = train
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed) if train else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        m = self.mask[i].copy()
        feats = {k: self.features[k][i].copy() for k in self.keys}

        if self.train and self.mask_prob > 0 and self.rng is not None:
            # 随机丢弃可用可选模态（模拟更丰富的缺失组合）
            for j, g in enumerate(self.optional):
                if m[j] == 1.0 and self.rng.random() < self.mask_prob:
                    m[j] = 0.0
                    feats[g] = np.zeros_like(feats[g])

        return (
            feats,
            torch.tensor(m),
            torch.tensor(self.y[i]),
        )


def prepare_multimodal_data(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int = 42,
    mask_prob: float = 0.3,
    label_col: str = "ph_label",
    optional: list[str] | None = None,
) -> dict[str, object]:
    """准备缺失模态鲁棒模型的数据（全部访视，不按可用性过滤）。

    各模态组独立做 train-only 标准化，缺失模态特征 NaN -> 0（由 mask 语义区分）。
    `optional` 指定可缺失（显式 mask）模态，默认 ["ct","ecg","echo"]（合成 4 模态）；
    MIMIC 6 模态传 ["ct","ecg","echo","vitals","cxr"]。
    """
    optional = list(optional) if optional is not None else list(IMAGING_KEYS)
    keys = ["structured", *optional]
    splits = patient_level_split(df, ratios=ratios, seed=seed, stratify_by=label_col)

    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feats: dict[str, dict[str, np.ndarray]] = {name: {} for name in ["train", "val", "test"]}

    # 仅用训练集拟合标准化（组名 -> 特征列名，再取列）
    for g in keys:
        cols = input_columns(MULTIMODAL_GROUPS[g])
        Xtr, mean, std = standardize(splits["train"][cols].to_numpy(dtype=np.float32))
        scalers[g] = (mean, std)
        feats["train"][g] = Xtr
        for name in ["val", "test"]:
            X, _, _ = standardize(splits[name][cols].to_numpy(dtype=np.float32), mean, std)
            feats[name][g] = X

    datasets: dict[str, MultimodalDataset] = {}
    for name in ["train", "val", "test"]:
        sdf = splits[name]
        mask = sdf[[f"avail_{k}" for k in optional]].to_numpy().astype(np.float32)
        y = sdf[label_col].to_numpy().astype(np.float32)
        datasets[name] = MultimodalDataset(
            feats[name], mask, y,
            train=(name == "train"), mask_prob=mask_prob, seed=seed, optional=optional,
        )

    return {
        "datasets": datasets,
        "splits": splits,
        "scalers": scalers,
        "columns": {g: input_columns(MULTIMODAL_GROUPS[g]) for g in keys},
        "optional": optional,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
    }


# =====================================================================
# 真实数据（MIMIC）原始 ECG 波形多模态准备 —— CNN 编码器架构重设计
# =====================================================================

# 降维列集合：去掉无判别力的维度。
#   structured 9 -> 6（去 who_fc / six_mwd / bnp —— MIMIC 恒缺失，纯哑元）
#   echo       5 -> 3（去 echo_tr_velocity / echo_rvsp —— 泄漏防护置 0；注意二者并非
#                    原始数据恒 0，实测可用率 ~75%，是 ph_label 的定义源，必须排除）
REDUCED_STRUCTURED_COLS = ["age", "sex", "bmi", "nt_probnp", "hemoglobin", "uric_acid"]
REDUCED_ECHO_COLS = ["echo_tapse", "echo_rv_basal_diameter", "echo_pericardial_effusion"]

# 模态 -> 降维列（ecg 为原始波形 3D，不在此处）
REDUCED_GROUP_COLS: dict[str, list[str]] = {
    "structured": REDUCED_STRUCTURED_COLS,
    "ct": list(FEATURE_GROUPS["ct"]),
    "echo": REDUCED_ECHO_COLS,
    "vitals": list(FEATURE_GROUPS["vitals"]),
    "cxr": list(FEATURE_GROUPS["cxr"]),
}

# ---- 审计扩展列：补入未使用的 PH 相关化验与 CXR PH 定性发现 ----
EXPANDED_LAB_COLS = ["troponin", "crp", "ldh", "bilirubin", "creatinine", "platelet"]
EXPANDED_STRUCTURED_COLS = REDUCED_STRUCTURED_COLS + EXPANDED_LAB_COLS      # 12 维
EXPANDED_CXR_COLS = [
    "cxr_cardiomegaly", "cxr_pleural_effusion", "cxr_pulmonary_edema",
    "cxr_pulmonary_hypertension", "cxr_pa_enlargement", "cxr_rv_enlargement",
]                                                                         # 6 维
EXPANDED_GROUP_COLS: dict[str, list[str]] = {
    "structured": EXPANDED_STRUCTURED_COLS,
    "ct": list(FEATURE_GROUPS["ct"]),
    "echo": REDUCED_ECHO_COLS,
    "vitals": list(FEATURE_GROUPS["vitals"]),
    "cxr": EXPANDED_CXR_COLS,
}


def _build_ecg_waves(
    sdf: pd.DataFrame,
    ecg_waveforms: dict[int, np.ndarray],
    L: int,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 (n, 12, L) float16 波形数组。

    avail_ecg=1 且有波形 -> per-lead z-score；否则全 0（缺失语义由 mask 区分）。
    mean/std 为 None 时（train）从可用波形拟合（仅训练集统计，防预处理泄漏）。
    波形按 hadm_id 查表；sdf 需含 hadm_id 列（训练脚本合并 admissions_structured）。
    """
    n = len(sdf)
    avail = (sdf["avail_ecg"].to_numpy() == 1)
    hadm = sdf["hadm_id"].to_numpy() if "hadm_id" in sdf.columns else None

    def _wave(i: int) -> np.ndarray | None:
        h = int(hadm[i]) if hadm is not None else -1
        w = ecg_waveforms.get(h)
        if w is None:
            return None
        w = w.astype(np.float32)
        if w.shape[1] > L:
            w = w[:, :L]
        elif w.shape[1] < L:
            w = np.pad(w, ((0, 0), (0, L - w.shape[1])))
        return w

    # 流式计算 per-lead mean/std（避免大临时数组）
    if mean is None or std is None:
        sum_ = np.zeros(12, dtype=np.float64)
        sumsq_ = np.zeros(12, dtype=np.float64)
        cnt = 0
        for i in range(n):
            if not avail[i]:
                continue
            w = _wave(i)
            if w is None:
                avail[i] = False  # 标记可用但无波形（解析失败）——按缺失处理
                continue
            sum_ += w.sum(axis=1)
            sumsq_ += (w.astype(np.float64) ** 2).sum(axis=1)
            cnt += w.shape[1]
        if cnt > 0:
            mean = (sum_ / cnt).astype(np.float32)[:, None]   # (12, 1) per-lead
            var = np.clip(sumsq_ / cnt - (sum_ / cnt) ** 2, 0.0, None)
            std = (np.sqrt(var) + 1e-6).astype(np.float32)[:, None]  # (12, 1)
        else:
            mean = np.zeros((12, 1), dtype=np.float32)
            std = np.ones((12, 1), dtype=np.float32)

    out = np.zeros((n, 12, L), dtype=np.float16)
    for i in range(n):
        if not avail[i]:
            continue
        w = _wave(i)
        if w is None:
            continue
        out[i] = ((w - mean) / std).astype(np.float16)
    return out, mean, std


def prepare_mimic_raw_ecg_data(
    df: pd.DataFrame,
    ecg_waveforms: dict[int, np.ndarray],
    ratios: dict[str, float],
    seed: int = 42,
    mask_prob: float = 0.3,
    label_col: str = "ph_label",
    optional: list[str] | None = None,
    group_cols: dict[str, list[str]] | None = None,
    winsorize: bool = False,
) -> dict[str, object]:
    """真实数据（MIMIC）多模态准备 —— ECG 用原始波形（CNN），其余模态降维。

    与 prepare_multimodal_data 同构，差异：
      * structured/echo 用降维列集合（去无判别力/泄漏防护维度）；
      * ecg 模态 = 原始波形 3D (n, 12, L) float16（per-lead z-score，缺失全 0）。
    返回 dict 结构与 prepare_multimodal_data 完全一致（datasets/splits/scalers/
    columns/optional/n_*），故训练脚本可无缝切换。
    `df` 需含 hadm_id 列（由训练脚本合并 admissions_structured.pkl 得到）。
    `group_cols` 缺省 = REDUCED_GROUP_COLS（CNN 基线）；前沿扩展实验传 EXPANDED_GROUP_COLS。
    `winsorize`=True 时对标量模态做 train-only [0.5, 99.5] 分位截尾（去数据录入错误/
    重尾异常值，如 bmi=126540），防训练梯度爆炸；默认 False（基线行为不变）。
    """
    optional = list(optional) if optional is not None else list(OPTIONAL_MODALITIES)
    keys = ["structured", *optional]
    gcols = group_cols or REDUCED_GROUP_COLS
    splits = patient_level_split(df, ratios=ratios, seed=seed, stratify_by=label_col)

    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feats: dict[str, dict[str, np.ndarray]] = {name: {} for name in ["train", "val", "test"]}

    # 标量模态：train-only 标准化（NaN -> 0）；可选 train-only winsorize
    for g in keys:
        if g == "ecg":
            continue
        Xtr_raw = splits["train"][gcols[g]].to_numpy(dtype=np.float32)
        lo = hi = None
        if winsorize:
            lo = np.nanpercentile(Xtr_raw, 0.5, axis=0).astype(np.float32)
            hi = np.nanpercentile(Xtr_raw, 99.5, axis=0).astype(np.float32)
        Xtr, mean, std = standardize(np.clip(Xtr_raw, lo, hi) if winsorize else Xtr_raw)
        scalers[g] = (mean, std)
        feats["train"][g] = Xtr
        for name in ["val", "test"]:
            X_raw = splits[name][gcols[g]].to_numpy(dtype=np.float32)
            if winsorize:
                X_raw = np.clip(X_raw, lo, hi)
            X, _, _ = standardize(X_raw, mean, std)
            feats[name][g] = X

    # ECG 原始波形（3D，per-lead z-score，train-only 统计）
    sample = next(iter(ecg_waveforms.values()))
    L = int(sample.shape[1])
    feats["train"]["ecg"], ecg_mean, ecg_std = _build_ecg_waves(
        splits["train"], ecg_waveforms, L, None, None)
    for name in ["val", "test"]:
        feats[name]["ecg"], _, _ = _build_ecg_waves(
            splits[name], ecg_waveforms, L, ecg_mean, ecg_std)
    scalers["ecg"] = (ecg_mean, ecg_std)

    datasets: dict[str, MultimodalDataset] = {}
    for name in ["train", "val", "test"]:
        sdf = splits[name]
        mask = sdf[[f"avail_{k}" for k in optional]].to_numpy().astype(np.float32)
        y = sdf[label_col].to_numpy().astype(np.float32)
        datasets[name] = MultimodalDataset(
            feats[name], mask, y,
            train=(name == "train"), mask_prob=mask_prob, seed=seed, optional=optional,
        )

    return {
        "datasets": datasets,
        "splits": splits,
        "scalers": scalers,
        "columns": {g: (gcols[g] if g != "ecg" else [f"ecg_wave_12x{L}"])
                    for g in keys},
        "optional": optional,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
    }
