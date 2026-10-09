"""常量定义：模态、特征组、列名。

这是整个系统的 canonical feature schema 的单一来源。
真实数据接入时，只需保证列名与本文件一致（或在 ETL 层做映射）。
"""
from __future__ import annotations

# ---- 参与“缺失模态鲁棒建模”研究的成像模态（可缺失） ----
IMAGING_MODALITIES: list[str] = ["ct", "ecg", "echo"]

# ---- 全部可选（可缺失、显式 mask）模态，mask 规范顺序 ----
# 前 3 个 = 成像模态（合成数据沿用）；vitals/cxr = MIMIC 扩展的新可选模态。
OPTIONAL_MODALITIES: list[str] = ["ct", "ecg", "echo", "vitals", "cxr"]

# ---- 特征组 -> 列名 ----
FEATURE_GROUPS: dict[str, list[str]] = {
    "clinical": ["age", "sex", "bmi", "who_fc", "six_mwd"],
    "biochemical": ["nt_probnp", "bnp", "hemoglobin", "uric_acid"],
    "hemodynamic": ["mpap", "pvr", "rap", "pawp", "cardiac_index"],
    "ct": ["ct_pa_diameter", "ct_pa_aorta_ratio", "ct_rv_lv_ratio"],
    "ecg": ["ecg_right_axis_degree", "ecg_rs_v1", "ecg_rv1_sv5", "ecg_p_pulmonale"],
    "echo": [
        "echo_tr_velocity",
        "echo_rvsp",
        "echo_tapse",
        "echo_rv_basal_diameter",
        "echo_pericardial_effusion",
    ],
    "treatment": ["tx_vasodilator", "tx_diuretic", "tx_anticoagulant"],
    # ---- MIMIC 扩展：新可选模态（显式 mask，缺失时特征=0 + avail_*=0） ----
    "vitals": [
        "vitals_temperature",
        "vitals_heartrate",
        "vitals_resprate",
        "vitals_o2sat",
        "vitals_sbp",
        "vitals_dbp",
    ],
    "cxr": [
        "cxr_cardiomegaly",
        "cxr_pleural_effusion",
        "cxr_pulmonary_edema",
    ],
}

# ---- 标签列 ----
LABEL_COLUMNS: list[str] = ["ph_label", "cor_pulmonale_label"]

# ---- 元数据列（非特征、非标签） ----
METADATA_COLUMNS: list[str] = [
    "patient_id",
    "visit_id",
    "visit_index",
    "timestamp",
    "avail_ct",
    "avail_ecg",
    "avail_echo",
    "avail_vitals",
    "avail_cxr",
]


def modality_columns(modality: str) -> list[str]:
    """返回某个模态对应的特征列名。"""
    if modality not in FEATURE_GROUPS:
        raise KeyError(f"unknown modality/group: {modality}")
    return FEATURE_GROUPS[modality]


def all_feature_columns() -> list[str]:
    """返回全部特征列（按固定顺序）。"""
    cols: list[str] = []
    for name in FEATURE_GROUPS:
        cols.extend(FEATURE_GROUPS[name])
    return cols


ALL_FEATURE_COLUMNS: list[str] = all_feature_columns()

# 结构化（非成像）特征组：第一版 synthetic 数据默认总是可用
STRUCTURED_GROUPS: list[str] = [
    "clinical",
    "biochemical",
    "hemodynamic",
    "treatment",
]
