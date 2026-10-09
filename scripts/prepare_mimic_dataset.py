"""Phase 4 — 组装 MIMIC 多模态 DataFrame 并落盘。

把 Phase 1/2（structured + ICD + echo）合并为项目 38 列 schema，落盘
data/processed/mimic_ph_dataset.csv。ECG 特征若已存在（Phase 3 回填
ecg_features.pkl）则一并 join，否则 avail_ecg=0 / ecg 特征为 NaN。

标签诚实边界：
  * ph_label           = echo-PH 代理（phtn_severity / tr_mmhg），非 RHC。
  * cor_pulmonale_label = echo 右心功能恶化代理（rv_function / rv_diam）。
  * echo 输入排除 echo_tr_velocity / echo_rvsp（泄漏防护，两列置 0）。
  * who_fc / six_mwd / bnp / 血流动力学 / treatment / ct 均置 0（MIMIC 无）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.constants import (  # noqa: E402
    ALL_FEATURE_COLUMNS, FEATURE_GROUPS, LABEL_COLUMNS, METADATA_COLUMNS,
)
from datasets.mimic_loader import (  # noqa: E402
    PROCESSED_DIR, join_echo_to_admissions,
)
from datasets.mimic_ed import load_vitals  # noqa: E402
from datasets.mimic_cxr import join_cxr_to_admissions  # noqa: E402

VITALS_FEATURES = FEATURE_GROUPS["vitals"]
CXR_FEATURES = FEATURE_GROUPS["cxr"]

OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "mimic_ph_dataset.csv"

# 恒缺失列（MIMIC 无此信息） -> 0
ZERO_FEATURES = [
    "who_fc", "six_mwd", "bnp",
    "mpap", "pvr", "rap", "pawp", "cardiac_index",
    "ct_pa_diameter", "ct_pa_aorta_ratio", "ct_rv_lv_ratio",
    "tx_vasodilator", "tx_diuretic", "tx_anticoagulant",
    # 泄漏防护：echo 压力特征恒 0（其派生 ph_label）
    "echo_tr_velocity", "echo_rvsp",
]


def load_ecg_features() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "ecg_admission_features.pkl"
    if not p.exists():
        return None
    return pd.read_pickle(p)


def assemble() -> pd.DataFrame:
    adm = pd.read_pickle(PROCESSED_DIR / "admissions_structured.pkl")
    echo_feat = pd.read_pickle(PROCESSED_DIR / "echo_features.pkl")

    # CXR join 需原始 subject_id/admittime/dischtime 列（rename 前调用）
    cxr = join_cxr_to_admissions(adm)

    echo_join = join_echo_to_admissions(adm, echo_feat)
    adm = adm.rename(columns={"subject_id": "patient_id", "admittime": "timestamp"})
    df = adm.merge(echo_join, on="hadm_id", how="left")
    df["avail_echo"] = df["avail_echo"].fillna(0).astype(int)

    # ---- ECG join（若已回填）----
    ecg = load_ecg_features()
    if ecg is not None:
        df = df.merge(ecg, on="hadm_id", how="left")
        df["avail_ecg"] = df["avail_ecg"].fillna(0).astype(int)
    else:
        df["avail_ecg"] = 0
        for c in ["ecg_right_axis_degree", "ecg_rs_v1", "ecg_rv1_sv5", "ecg_p_pulmonale"]:
            df[c] = np.nan

    # ---- ED vitals join（新可选模态）----
    vitals = load_vitals()
    if vitals is not None and len(vitals):
        df = df.merge(vitals, on="hadm_id", how="left")
        df["avail_vitals"] = df["avail_vitals"].fillna(0).astype(int)
    else:
        df["avail_vitals"] = 0
        for c in VITALS_FEATURES:
            df[c] = np.nan

    # ---- CXR findings join（新可选模态；无 metadata.csv.gz -> 患者级 subject 关联）----
    if cxr is not None and len(cxr):
        df = df.merge(cxr, on="hadm_id", how="left")
        df["avail_cxr"] = df["avail_cxr"].fillna(0).astype(int)
    else:
        df["avail_cxr"] = 0
        for c in CXR_FEATURES:
            df[c] = np.nan

    df["avail_ct"] = 0  # ct 恒缺失

    # ---- 恒缺失列置 0 ----
    for c in ZERO_FEATURES:
        df[c] = 0.0

    # ---- 列序对齐 canonical schema（38 列） + 额外敏感性标签列 ----
    cols = METADATA_COLUMNS + ALL_FEATURE_COLUMNS + LABEL_COLUMNS
    df = df[cols + ["icd_ph", "icd_corpulmonale"]]

    # 元数据类型
    df["patient_id"] = df["patient_id"].astype(int)
    df["visit_index"] = df["visit_index"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def smoke_check(df: pd.DataFrame) -> None:
    print("\n=== [assemble smoke] ===")
    print(f"n_rows (admissions) = {len(df)}")
    print(f"n_patients          = {df['patient_id'].nunique()}")
    print(f"avail_echo rate     = {df['avail_echo'].mean():.3f}")
    print(f"avail_ecg rate      = {df['avail_ecg'].mean():.3f}")
    print(f"avail_ct  rate      = {df['avail_ct'].mean():.3f} (must be 0)")
    print(f"avail_vitals rate   = {df['avail_vitals'].mean():.3f}")
    print(f"avail_cxr rate      = {df['avail_cxr'].mean():.3f}")

    # 标签
    echo = df[df["avail_echo"] == 1]
    print(f"\n-- 早筛人群（echo 访视 n={len(echo)}）--")
    print(f"  ph_label prevalence = {echo['ph_label'].mean():.4f} "
          f"(non-null {echo['ph_label'].notna().sum()})")
    print(f"  cor_pulm prevalence = {echo['cor_pulmonale_label'].mean():.4f} "
          f"(non-null {echo['cor_pulmonale_label'].notna().sum()})")
    # ICD 敏感性标签（无泄漏，独立于 echo）
    print(f"  ICD-PH prevalence   = {echo['icd_ph'].mean():.4f}")
    print(f"  ICD-cor pulm prev   = {echo['icd_corpulmonale'].mean():.4f}")

    # 特征缺失率（关键列）
    print("\n-- 特征缺失率（NaN 比例）--")
    for c in ["bmi", "nt_probnp", "hemoglobin", "uric_acid",
              "echo_tapse", "echo_rv_basal_diameter", "echo_pericardial_effusion",
              "ecg_right_axis_degree",
              "vitals_temperature", "vitals_o2sat", "vitals_sbp",
              "cxr_cardiomegaly", "cxr_pleural_effusion", "cxr_pulmonary_edema"]:
        print(f"  {c:<28} = {df[c].isna().mean():.3f}")

    # 泄漏自检：echo 结构特征 vs ph_label 的相关（右心结构应携带 PH 信号，但非自证）
    print("\n-- 泄漏/信号自检（echo 人群）--")
    for c in ["echo_tapse", "echo_rv_basal_diameter", "echo_pericardial_effusion"]:
        v = echo[c]
        pos = echo.loc[echo["ph_label"] == 1, c]
        neg = echo.loc[echo["ph_label"] == 0, c]
        print(f"  {c:<28} pos_mean={pos.mean():.3f} neg_mean={neg.mean():.3f}")

    # 纵向：≥2 admission 的患者数
    vc = df.groupby("patient_id")["visit_index"].max() + 1
    print(f"\n-- 纵向 --")
    print(f"  patients with >=2 admissions = {(vc >= 2).sum()}")
    print(f"  max visits/patient           = {vc.max()}")


def main() -> None:
    df = assemble()
    smoke_check(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
