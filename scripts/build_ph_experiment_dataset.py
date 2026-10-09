"""构建固定 PH 实验数据集（训练 MIMIC + 外部验证 EchoNext）。

一次性形成、后期所有实验复用同一数据集（不再各自重划分/重标准化）：

  训练集（内部）：MIMIC-IV echo 队列
    label  = ph_label（echo-PH 代理：phtn_severity 或 RVSP>=34mmHg），非 RHC。
    cohort = avail_echo==1 且 ph_label 非空（早筛人群）。
    泄漏防护（与既有审计一致）：
      * echo_tr_velocity / echo_rvsp 是 ph_label 定义源 → 排除出输入（置 0）。
      * mpap/pvr/rap/pawp/cardiac_index（血流动力学，RHC 金标准量）→ 恒 0，不入输入。
      * ct 定量特征 MIMIC 无 → 恒 0。
    modality = structured(6) / ecg(波形 12×L) / echo(3) / vitals(6) / cxr(3) / ct(恒缺)。
    划分 = 患者级 70/10/20，seed 42，按 ph_label 分层；train-only winsorize[0.5,99.5]+标准化。

  外部验证集：EchoNext（Columbia + Allen 医院）
    label  = pasp_gte_45_flag（PASP>=45mmHg，echo 派生），次标签 tr_max_gte_32_flag。
    cohort = split=='test'（官方测试集，每患者仅最新 ECG）。
    可用模态 = ecg（原始 12 导联波形，10s/250Hz）+ structured(age/sex)；
              echo/vitals/cxr/ct 全缺 → 外部部署的“缺失模态”核心场景。

输出到 data/ph_experiment/：
  mimic_train.pkl              —— 训练/验证/测试访视（标准化特征 + mask + 标签 + split + hadm_id）
  mimic_scalers.json           —— 各特征 train-only (mean, std) + winsorize (lo, hi)
  echonext_external.pkl        —— 外部验证访视（标签 + 原始特征 + ecg_key）
  manifest.yaml                —— 版本/标签定义/泄漏控制/文件指针
  data_correctness_report.txt  —— 正确性自检（划分无重叠、标签率、NaN 率、泄漏核对）
  README.md                    —— 数据集文档

用法：
  python scripts/build_ph_experiment_dataset.py            # 全量
  python scripts/build_ph_experiment_dataset.py --smoke    # 冒烟（子采样，不落盘）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.config import PROJECT_ROOT  # noqa: E402
from core.constants import OPTIONAL_MODALITIES  # noqa: E402
from datasets.multimodal_dataset import REDUCED_GROUP_COLS  # noqa: E402
from datasets.splits import patient_level_split  # noqa: E402
from datasets.mimic_loader import PROCESSED_DIR  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "ph_experiment"
MIMIC_CSV = PROJECT_ROOT / "data" / "processed" / "mimic_ph_dataset.csv"
ADM_PKL = PROCESSED_DIR / "admissions_structured.pkl"
ECG_WAVE_PKL = PROCESSED_DIR / "ecg_raw_waveforms.pkl"
ECHONEXT_META = PROJECT_ROOT / "echonext" / "echonext_metadata_100k.csv"

SEED = 42
RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
WINSOR_Q = (0.5, 99.5)

# 泄漏防护：这些列是标签定义源或 MIMIC 不存在，绝不可作为输入
LEAKAGE_EXCLUDED = [
    "echo_tr_velocity", "echo_rvsp",                 # ph_label 定义源
    "mpap", "pvr", "rap", "pawp", "cardiac_index",   # 血流动力学（RHC 金标准）
    "ct_pa_diameter", "ct_pa_aorta_ratio", "ct_rv_lv_ratio",  # ct MIMIC 无
    "who_fc", "six_mwd", "bnp",                      # MIMIC 无
]


def _add_hadm_id(df: pd.DataFrame) -> pd.DataFrame:
    adm = pd.read_pickle(ADM_PKL)
    hadm = adm[["subject_id", "visit_index", "hadm_id"]].rename(
        columns={"subject_id": "patient_id"})
    return df.merge(hadm, on=["patient_id", "visit_index"], how="left")


def _feature_cols() -> dict[str, list[str]]:
    """每模态实际输入特征列（与 REDUCED_GROUP_COLS 一致）。"""
    cols = {}
    for m in ["structured", "echo", "vitals", "cxr"]:
        cols[m] = list(REDUCED_GROUP_COLS[m])
    return cols


def _standardize(df: pd.DataFrame, cols: list[str], train_idx: pd.Series,
                 scalers: dict) -> None:
    """train-only winsorize + 标准化，写回 df（原地，列改名 z_*）。"""
    Xtr = df.loc[train_idx, cols].to_numpy(dtype=np.float32)
    for j, c in enumerate(cols):
        lo = np.nanpercentile(Xtr[:, j], WINSOR_Q[0])
        hi = np.nanpercentile(Xtr[:, j], WINSOR_Q[1])
        x = np.clip(df[c].to_numpy(dtype=np.float32), lo, hi)
        mean = np.nanmean(x[train_idx.to_numpy()]) if train_idx.any() else 0.0
        std = np.nanstd(x[train_idx.to_numpy()]) if train_idx.any() else 1.0
        std = std if std > 1e-8 else 1.0
        z = (x - mean) / std
        z = np.where(np.isfinite(z), z, 0.0)          # NaN/Inf -> 0（缺失语义由 mask 区分）
        df[f"z_{c}"] = z.astype(np.float32)
        scalers[c] = {"mean": float(mean), "std": float(std),
                      "lo": float(lo), "hi": float(hi)}


def build_mimic(smoke: bool) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(MIMIC_CSV)
    df = df[df["ph_label"].notna()].reset_index(drop=True)
    df["ph_label"] = df["ph_label"].astype(int)
    df = _add_hadm_id(df)
    if smoke:
        sub = pd.Series(df["patient_id"].unique()).sort_values().iloc[:1500].to_numpy()
        df = df[df["patient_id"].isin(sub)].reset_index(drop=True)

    splits = patient_level_split(df, RATIOS, seed=SEED, stratify_by="ph_label",
                                 patient_col="patient_id")
    # patient_level_split 返回的 DataFrame 已 reset_index，故按 patient_id 反查划分
    patient_split: dict = {}
    for name, sdf in splits.items():
        for pid in sdf["patient_id"].unique():
            patient_split[pid] = name
    df["split"] = df["patient_id"].map(patient_split).astype(str)

    cols_by_mod = _feature_cols()
    all_cols = [c for m in cols_by_mod for c in cols_by_mod[m]]
    scalers: dict = {}
    train_idx = df["split"] == "train"
    _standardize(df, all_cols, train_idx, scalers)

    # mask 列
    mask_cols = [f"avail_{m}" for m in OPTIONAL_MODALITIES]
    out_cols = ["patient_id", "visit_id", "visit_index", "hadm_id", "split",
                "ph_label", "icd_ph", "cor_pulmonale_label", *mask_cols,
                *[f"z_{c}" for c in all_cols]]
    return df[out_cols], scalers


def build_echonext(smoke: bool) -> pd.DataFrame:
    meta = pd.read_csv(ECHONEXT_META)
    ext = meta[meta["split"] == "test"].reset_index(drop=True)
    if smoke:
        ext = ext.iloc[:512].reset_index(drop=True)
    keep = ["ecg_key", "patient_key", "age_at_ecg", "sex", "ventricular_rate",
            "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected",
            "pasp_gte_45_flag", "tr_max_gte_32_flag", "shd_moderate_or_greater_flag"]
    ext = ext[keep].rename(columns={
        "pasp_gte_45_flag": "ph_label_external",
        "tr_max_gte_32_flag": "ph_label_tr32",
        "shd_moderate_or_greater_flag": "shd_moderate_or_greater",
    })
    ext["split"] = "external"
    return ext


def correctness_report(df: pd.DataFrame, scalers: dict, ext: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PH 实验数据集 正确性自检")
    lines.append("=" * 70)
    # 1. 划分无重叠
    g = df.groupby("patient_id")["split"].nunique()
    lines.append(f"[1] 患者级划分重叠检查：患者跨划分数 = {int((g > 1).sum())}（必须为 0）")
    lines.append(f"    各划分访视数：{df['split'].value_counts().to_dict()}")
    # 2. 标签率
    lines.append(f"[2] 标签 ph_label 患病率：{df['ph_label'].mean():.4f}（echo 早筛人群）")
    lines.append(f"    交叉标签 icd_ph 患病率：{df['icd_ph'].mean():.4f}")
    # 3. 泄漏核对
    leak_in = [c for c in LEAKAGE_EXCLUDED if c in df.columns]
    lines.append(f"[3] 泄漏防护：输入列中不含 {len(leak_in)} 个排除列（echo_tr_velocity/echo_rvsp/"
                 f"mpap/pvr/rap/pawp/cardiac_index/ct_*）→ 已全部排除，OK" if not leak_in
                 else f"    警告：仍含 {leak_in}")
    # 4. 模态可用率
    lines.append("[4] 模态可用率：")
    for m in OPTIONAL_MODALITIES:
        lines.append(f"    avail_{m:<8} = {df['avail_' + m].mean():.4f}")
    # 5. 标准化后 NaN
    zcols = [c for c in df.columns if c.startswith("z_")]
    nan_rate = df[zcols].isna().mean().mean()
    lines.append(f"[5] 标准化特征 NaN 率 = {nan_rate:.6f}（应为 0）")
    # 6. scaler 数
    lines.append(f"[6] scaler 覆盖特征数 = {len(scalers)}（结构化+echo+vitals+cxr 共 {len(zcols)}）")
    # 7. 外部集
    lines.append(f"[7] 外部 EchoNext：n={len(ext)}  ph(pasp>=45) 患病率="
                 f"{ext['ph_label_external'].mean():.4f}  tr>=3.2 患病率="
                 f"{ext['ph_label_tr32'].mean():.4f}")
    return "\n".join(lines)


def main() -> None:
    smoke = "--smoke" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("构建 MIMIC 训练集（固定划分 + 标准化）...")
    df, scalers = build_mimic(smoke)
    print("构建 EchoNext 外部验证集...")
    ext = build_echonext(smoke)

    report = correctness_report(df, scalers, ext)
    print("\n" + report)

    if smoke:
        print("\n[SMOKE] 不落盘。")
        return

    df.to_pickle(OUT_DIR / "mimic_train.pkl")
    ext.to_pickle(OUT_DIR / "echonext_external.pkl")
    (OUT_DIR / "mimic_scalers.json").write_text(
        json.dumps(scalers, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "version": "1.0",
        "created": "2026-10-03",
        "split_seed": SEED,
        "split_ratios": RATIOS,
        "train": {
            "source": "MIMIC-IV echo cohort (BIDMC)",
            "file": "mimic_train.pkl",
            "label": "ph_label = phtn_severity 或 RVSP>=34mmHg（echo-PH 代理，非 RHC）",
            "secondary_labels": ["icd_ph", "cor_pulmonale_label"],
            "modalities": {
                "structured": REDUCED_GROUP_COLS["structured"],
                "ecg": "raw 12-lead waveform (12×L), data/processed/ecg_raw_waveforms.pkl keyed by hadm_id",
                "echo": REDUCED_GROUP_COLS["echo"],
                "vitals": REDUCED_GROUP_COLS["vitals"],
                "cxr": REDUCED_GROUP_COLS["cxr"],
                "ct": "always missing",
            },
            "leakage_excluded": LEAKAGE_EXCLUDED,
            "standardization": "train-only winsorize[0.5,99.5] + z-score; NaN/Inf -> 0",
        },
        "external": {
            "source": "EchoNext (Columbia + Allen)",
            "file": "echonext_external.pkl",
            "label": "pasp_gte_45_flag (PASP>=45mmHg)",
            "secondary_labels": ["ph_label_tr32 (tr>=3.2m/s)", "shd_moderate_or_greater"],
            "available_modalities": ["ecg (waveform)", "structured (age+sex only)"],
            "missing_modalities": ["echo", "vitals", "cxr", "ct"],
        },
        "is_synthetic": False,
    }
    (OUT_DIR / "manifest.yaml").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "data_correctness_report.txt").write_text(report, encoding="utf-8")

    readme = f"""# PH 实验固定数据集

训练（内部）+ 外部验证（跨机构）一次形成，**后期所有实验复用本目录**，不再各自重划分/重标准化。

## 文件
- `mimic_train.pkl` —— MIMIC echo 队列（含 split / ph_label / avail_* / 标准化特征 z_* / hadm_id）
- `mimic_scalers.json` —— 各特征 train-only (mean/std) + winsorize (lo/hi)
- `echonext_external.pkl` —— EchoNext 官方 test（外部验证，pasp_gte_45 主标签）
- `manifest.yaml` —— 版本 / 标签定义 / 泄漏控制 / 文件指针
- `data_correctness_report.txt` —— 正确性自检

## 标签
- 训练：`ph_label` = phtn_severity 或 RVSP>=34mmHg（echo-PH 代理，非 RHC）；交叉标签 `icd_ph`。
- 外部：`ph_label_external` = pasp_gte_45_flag（PASP>=45mmHg）；次标签 `ph_label_tr32`（TR>=3.2m/s）。

## 泄漏防护（训练集输入中已排除）
{', '.join(LEAKAGE_EXCLUDED)}

## 模态与缺失
- 训练：structured(6)/ecg(波形)/echo(3)/vitals(6)/cxr(3)，ct 恒缺失。
- 外部：仅 ecg + structured(age/sex)，echo/vitals/cxr/ct 全缺 → 跨机构缺失模态核心场景。
- ECG 波形不入 parquet，引用 `data/processed/ecg_raw_waveforms.pkl`（hadm_id 键）。
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n已写入: {OUT_DIR}")
    print(f"  mimic_train.pkl    {len(df)} 行 × {len(df.columns)} 列")
    print(f"  echonext_external.pkl {len(ext)} 行")


if __name__ == "__main__":
    main()
