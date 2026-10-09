"""PH 固定实验数据集加载器（scripts/build_ph_experiment_dataset.py 的产物）。

读取 data/ph_experiment/ 下固定数据（不再各自重划分/重标准化），产出与既有训练/评估
代码兼容的 Dataset（__getitem__ -> (inputs, mask, y)）。

  inputs: {"structured": (6,), "ct": (3,), "ecg": (12, L) float32,
           "echo": (3,), "vitals": (6,), "cxr": (3,)}
  mask  : (5,) float32，顺序 = [ct, ecg, echo, vitals, cxr]；structured 恒可用。
  y     : ph_label

ECG 波形：MIMIC 引用 mimic/extracted/processed/ecg_raw_waveforms.pkl（hadm_id 键），per-lead z-score
用 train-only 统计（首调用时流式计算并缓存到 ecg_scalers.json）；EchoNext 用官方 npy + 其
train per-lead 统计（缓存 echonext_train_perlead_stats.npz）。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.config import PROJECT_ROOT
from core.constants import OPTIONAL_MODALITIES
from datasets.mimic_loader import PROCESSED_DIR
from datasets.multimodal_dataset import REDUCED_GROUP_COLS

EXP_DIR = PROJECT_ROOT / "data" / "ph_experiment"
MIMIC_CSV = PROJECT_ROOT / "data" / "processed" / "mimic_ph_dataset.csv"
ECG_WAVE_PKL = PROCESSED_DIR / "ecg_raw_waveforms.pkl"
ECHONEXT_DIR = PROJECT_ROOT / "echonext"

OPTIONAL = list(OPTIONAL_MODALITIES)


def _z_cols() -> dict[str, list[str]]:
    return {m: [f"z_{c}" for c in REDUCED_GROUP_COLS[m]]
            for m in ["structured", "echo", "vitals", "cxr"]}


def _load_ecg_waveforms() -> dict[int, np.ndarray]:
    with open(ECG_WAVE_PKL, "rb") as f:
        return pickle.load(f)


def _ecg_perlead_stats(ecg: dict[int, np.ndarray], hadm_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """train-only per-lead mean/std（流式，避免大临时数组）。返回 (mean, std, L)。"""
    cache = EXP_DIR / "ecg_scalers.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return (np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32), int(d["L"]))
    L = int(next(iter(ecg.values())).shape[1])
    hadm_set = set(int(h) for h in hadm_ids if pd.notna(h))
    sum_ = np.zeros(12, np.float64)
    sumsq_ = np.zeros(12, np.float64)
    cnt = 0
    for h in hadm_set:
        w = ecg.get(h)
        if w is None:
            continue
        w = w.astype(np.float32)
        if w.shape[1] > L:
            w = w[:, :L]
        elif w.shape[1] < L:
            w = np.pad(w, ((0, 0), (0, L - w.shape[1])))
        sum_ += w.sum(axis=1)
        sumsq_ += (w.astype(np.float64) ** 2).sum(axis=1)
        cnt += w.shape[1]
    mean = (sum_ / cnt).astype(np.float32)
    std = (np.sqrt(np.clip(sumsq_ / cnt - mean ** 2, 0, None)) + 1e-6).astype(np.float32)
    cache.write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist(), "L": L}), encoding="utf-8")
    return mean, std, L


def _wave(ecg: dict[int, np.ndarray], hadm, L: int) -> np.ndarray | None:
    if pd.isna(hadm):
        return None
    w = ecg.get(int(hadm))
    if w is None:
        return None
    w = w.astype(np.float32)
    if w.shape[1] > L:
        w = w[:, :L]
    elif w.shape[1] < L:
        w = np.pad(w, ((0, 0), (0, L - w.shape[1])))
    return w


class FixedPHDataset(Dataset):
    """MIMIC 固定数据集（标准化 tabular + mask + 标签 + ECG 波形）。"""

    def __init__(self, df: pd.DataFrame, ecg: dict[int, np.ndarray],
                 mean: np.ndarray, std: np.ndarray, L: int):
        self.df = df.reset_index(drop=True)
        self.ecg = ecg
        self.mean = mean
        self.std = std
        self.L = L
        self.zcols = _z_cols()
        self.y = df["ph_label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        inputs = {
            "structured": r[self.zcols["structured"]].to_numpy(np.float32),
            "ct": np.zeros(3, np.float32),
            "echo": r[self.zcols["echo"]].to_numpy(np.float32),
            "vitals": r[self.zcols["vitals"]].to_numpy(np.float32),
            "cxr": r[self.zcols["cxr"]].to_numpy(np.float32),
        }
        w = _wave(self.ecg, r["hadm_id"], self.L) if r["avail_ecg"] == 1 else None
        if w is None:
            inputs["ecg"] = np.zeros((12, self.L), np.float32)
        else:
            inputs["ecg"] = ((w - self.mean[:, None]) / self.std[:, None]).astype(np.float32)
        mask = np.array([r[f"avail_{m}"] for m in OPTIONAL], np.float32)
        return inputs, mask, self.y[i]


def load_fixed_mimic() -> tuple[FixedPHDataset, FixedPHDataset, FixedPHDataset, dict]:
    """返回 (train_ds, val_ds, test_ds, meta)。ECG per-lead 统计 train-only 首调用缓存。"""
    df = pd.read_pickle(EXP_DIR / "mimic_train.pkl")
    ecg = _load_ecg_waveforms()
    train_hadm = df.loc[df["split"] == "train", "hadm_id"].to_numpy()
    mean, std, L = _ecg_perlead_stats(ecg, train_hadm)
    meta = {
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "prev": float(df["ph_label"].mean()),
        "L": L,
    }
    return (
        FixedPHDataset(df[df["split"] == "train"], ecg, mean, std, L),
        FixedPHDataset(df[df["split"] == "val"], ecg, mean, std, L),
        FixedPHDataset(df[df["split"] == "test"], ecg, mean, std, L),
        meta,
    )


class EchoNextExternalDataset(Dataset):
    """EchoNext 外部验证（ECG 波形 + age/sex；echo/vitals/cxr/ct 全缺）。"""

    def __init__(self, ext: pd.DataFrame, wave_path: Path, wmean, wstd,
                 s_mean, s_std, L: int = 2500):
        self.ext = ext.reset_index(drop=True)
        self.wave = np.load(str(wave_path), mmap_mode="r")  # (N,1,2500,12)
        self.wmean = wmean
        self.wstd = wstd
        self.s_mean = s_mean
        self.s_std = s_std
        self.L = L
        assert self.wave.shape[0] >= len(self.ext)

    def __len__(self) -> int:
        return len(self.ext)

    def __getitem__(self, i: int):
        r = self.ext.iloc[i]
        x = self.wave[i][0].T.astype(np.float32)  # (12, 2500)
        if x.shape[1] > self.L:
            x = x[:, :self.L]
        elif x.shape[1] < self.L:
            x = np.pad(x, ((0, 0), (0, self.L - x.shape[1])))
        x = (x - self.wmean[:, None]) / self.wstd[:, None]
        age = float(r["age_at_ecg"]) if pd.notna(r["age_at_ecg"]) else 0.0
        sex_bin = 1.0 if str(r["sex"]).lower() == "female" else 0.0
        structured = np.zeros(6, np.float32)
        structured[0] = (age - self.s_mean[0]) / self.s_std[0]      # age
        structured[1] = (sex_bin - self.s_mean[1]) / self.s_std[1]  # sex (F=1/M=0)
        inputs = {
            "structured": structured,
            "ct": np.zeros(3, np.float32),
            "ecg": x,
            "echo": np.zeros(3, np.float32),
            "vitals": np.zeros(6, np.float32),
            "cxr": np.zeros(3, np.float32),
        }
        mask = np.array([0.0, 1.0, 0.0, 0.0, 0.0], np.float32)  # 仅 ecg 可用
        return inputs, mask, float(r["ph_label_external"])


def load_fixed_echonext() -> tuple[EchoNextExternalDataset, dict]:
    """返回 (external_ds, meta)。外部标签 = pasp_gte_45（次标签 tr>=3.2）。"""
    ext = pd.read_pickle(EXP_DIR / "echonext_external.pkl")
    scalers = json.loads((EXP_DIR / "mimic_scalers.json").read_text(encoding="utf-8"))
    s_mean = np.array([scalers[c]["mean"] for c in ["age", "sex", "bmi", "nt_probnp",
                                                    "hemoglobin", "uric_acid"]], np.float32)
    s_std = np.array([scalers[c]["std"] for c in ["age", "sex", "bmi", "nt_probnp",
                                                  "hemoglobin", "uric_acid"]], np.float32)
    # EchoNext train per-lead 统计（缓存；train 17.4GB 只流式算一次）
    stats_path = ECHONEXT_DIR.parent / "experiments" / "mimic" / "ph_experiment" / "echonext_train_perlead_stats.npz"
    if stats_path.exists():
        d = np.load(str(stats_path))
        wmean, wstd = d["mean"], d["std"]
    else:
        wmean, wstd = _echonext_perlead_stats(ECHONEXT_DIR / "EchoNext_train_waveforms.npy")
        np.savez(str(stats_path), mean=wmean, std=wstd)
    meta = {
        "n_external": int(len(ext)),
        "prev_pasp45": float(ext["ph_label_external"].mean()),
        "prev_tr32": float(ext["ph_label_tr32"].mean()),
    }
    return (EchoNextExternalDataset(
        ext, ECHONEXT_DIR / "EchoNext_test_waveforms.npy", wmean, wstd, s_mean, s_std),
        meta)


def _echonext_perlead_stats(wave_path: Path, chunk: int = 1024):
    w = np.load(str(wave_path), mmap_mode="r")
    n = w.shape[0]
    sum_ = np.zeros(12, np.float64)
    sumsq_ = np.zeros(12, np.float64)
    cnt = 0
    for start in range(0, n, chunk):
        x = w[start:start + chunk][:, 0, :, :].reshape(-1, 12)
        sum_ += x.sum(axis=0)
        sumsq_ += (x ** 2).sum(axis=0)
        cnt += x.shape[0]
    mean = (sum_ / cnt).astype(np.float32)
    std = (np.sqrt(np.clip(sumsq_ / cnt - mean ** 2, 0, None)) + 1e-6).astype(np.float32)
    return mean, std
