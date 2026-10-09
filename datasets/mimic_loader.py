"""MIMIC-IV real-data loader: extract structured + echo features into the project schema.

The project schema (core/constants.py) is hardcoded to 4 modalities
(structured/ct/ecg/echo). MIMIC-IV has no CT imaging and no RHC mPAP gold
standard, so:
  * ct modality is ALWAYS missing (avail_ct=0, ct features = 0).
  * who_fc / six_mwd / bnp are not present in MIMIC -> filled 0 (documented limit).
  * hemodynamic + treatment features are not mapped -> filled 0.
  * ph_label is an ECHO proxy (phtn_severity / tr_mmhg), NOT an RHC diagnosis.
  * cor_pulmonale_label is an ECHO RV-dysfunction proxy (rv_function / rv_diam).

Visit anchor = hospital ADMISSION (not echo study): each admission is one
visit, echo/ecg/lab are aggregated inside that admission's time window, so
echo is naturally missing for some admissions (MNAR referral bias).

Label-leakage guard: echo input EXCLUDES echo_tr_velocity / echo_rvsp (the two
pressure features that define ph_label). We keep only right-heart STRUCTURE
(tapse / rv_basal_diameter / pericardial_effusion); the pressure columns are
zeroed at assembly time (columns retained to keep the schema intact). ICD I27.x
is kept as an independent sensitivity label.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MIMIC_DIR = Path(__file__).resolve().parents[1] / "mimic"
EXTRACTED_DIR = MIMIC_DIR / "extracted"
HOSP_DIR = EXTRACTED_DIR / "mimic-iv-3.1" / "hosp"
PROCESSED_DIR = EXTRACTED_DIR / "processed"
ECHO_RAW_PATH = MIMIC_DIR / "structured-measurement.csv.gz"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (verified against d_labitems / d_icd_diagnoses)
# ---------------------------------------------------------------------------
LAB_ITEMIDS = {50963: "nt_probnp", 51007: "uric_acid", 51222: "hemoglobin"}

PH_ICD_CODES = {"I270", "I272", "I2720", "I2722", "I2723", "I2724", "I2729"}
CORPULMONALE_ICD_CODES = {"I260", "I2609", "I2781", "4150"}

ECHO_TARGETS = [
    "tr_velocity", "tr_mmhg", "tapse", "rv_diam",
    "pericardial_effusion", "rv_function", "phtn_severity",
]

# Qualitative -> numeric maps (exact strings observed in structured-measurement).
PERICARDIAL_MAP = {
    None: 0.0,
    "Trivial": 0.0,
    "Very small (<0.5cm)": 0.0,
    "Anterior fat pad": 0.0,
    "Small (<1.0cm)": 1.0,
    "Small-moderate": 1.0,
    "Moderate (1.0-2.0cm)": 2.0,
    "Moderate-large": 2.0,
    "Large (>2.0cm)": 3.0,
}

# phtn_severity -> echo-PH proxy
PH_SEVERITY_POS = {
    "Mild (TR 26-36mmHg)", "Mod (TR 37-60mmHg)", "Severe (TR >60mmHg)",
    "Mild-mod (TR 35-40mmHg)", "Mod-severe (50-60mmHg)", "Borderline (TR 26-30mmHg)",
}
PH_SEVERITY_NEG = {
    "Normal (TR<26mmHg)", "HIgh normal (TR 21-25 mmHg)", "PA HTN unlikely",
}
PH_TR_MMHG = 34.0  # TR gradient >= 34 mmHg -> echo-PH (per plan)

# rv_function -> cor-pulmonale proxy (severe RV dysfunction)
RV_FUNCTION_POS = {"Severe global hypo", "RV function depressed"}
RV_FUNCTION_NEG = {
    "Nl RV function", "Hyperdynamic", "Low normal function", "Mild global RV hypo",
}
RV_DIAM_ENLARGED_CM = 4.1  # RV basal diameter > 4.1 cm -> dilated


def _combine_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Positive if either signal positive; negative if either negative and
    neither positive; NaN only when both are indeterminate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pos = (a == 1) | (b == 1)
    neg = (~pos) & ((a == 0) | (b == 0))
    return np.where(pos, 1.0, np.where(neg, 0.0, np.nan))


# ---------------------------------------------------------------------------
# Phase 1: admissions + ICD + labs + BMI
# ---------------------------------------------------------------------------
def load_admissions() -> pd.DataFrame:
    """Admission-level table: subject_id, hadm_id, admittime, dischtime, age, sex."""
    patients = pd.read_csv(HOSP_DIR / "patients.csv.gz")
    admissions = pd.read_csv(
        HOSP_DIR / "admissions.csv.gz", parse_dates=["admittime", "dischtime"]
    )
    adm = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id", how="left",
    )
    # age at admission = anchor_age + (admission year - anchor year)
    adm["age"] = adm["anchor_age"] + (adm["admittime"].dt.year - adm["anchor_year"])
    adm["age"] = adm["age"].fillna(adm["anchor_age"])
    adm["sex"] = (adm["gender"].str.upper() == "F").astype(int)
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    # visit identity
    adm["visit_index"] = adm.groupby("subject_id").cumcount()
    adm["visit_id"] = adm["subject_id"].astype(str) + "_a" + adm["hadm_id"].astype(str)
    return adm


def load_icd_labels(adm: pd.DataFrame) -> pd.DataFrame:
    """Attach icd_ph / icd_corpulmonale (independent sensitivity labels)."""
    dx = pd.read_csv(HOSP_DIR / "diagnoses_icd.csv.gz")
    ph = dx[dx["icd_code"].isin(PH_ICD_CODES)][["subject_id", "hadm_id"]].drop_duplicates()
    ph["icd_ph"] = 1
    cp = dx[dx["icd_code"].isin(CORPULMONALE_ICD_CODES)][["subject_id", "hadm_id"]].drop_duplicates()
    cp["icd_corpulmonale"] = 1
    out = adm.merge(ph, on=["subject_id", "hadm_id"], how="left")
    out = out.merge(cp, on=["subject_id", "hadm_id"], how="left")
    for c in ["icd_ph", "icd_corpulmonale"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


def load_labs(adm: pd.DataFrame) -> pd.DataFrame:
    """Most recent nt_probnp / hemoglobin / uric_acid inside each admission."""
    cache = PROCESSED_DIR / "labs_admission.pkl"
    if cache.exists():
        lab_wide = pd.read_pickle(cache)
    else:
        cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
        chunks = []
        reader = pd.read_csv(
            HOSP_DIR / "labevents.csv.gz", usecols=cols, chunksize=2_000_000,
            dtype={"itemid": "float32"},
        )
        for chunk in reader:
            chunk = chunk[chunk["itemid"].isin(list(LAB_ITEMIDS))]
            if len(chunk):
                chunks.append(chunk)
        lab = pd.concat(chunks, ignore_index=True)
        lab["charttime"] = pd.to_datetime(lab["charttime"])
        lab = lab.merge(
            adm[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"], how="inner",
        )
        win = pd.Timedelta(days=1)
        lab = lab[
            (lab["charttime"] >= lab["admittime"] - win)
            & (lab["charttime"] <= lab["dischtime"] + win)
        ]
        lab = lab.sort_values("charttime").drop_duplicates(["hadm_id", "itemid"], keep="last")
        lab_wide = lab.pivot_table(index="hadm_id", columns="itemid", values="valuenum", aggfunc="last")
        lab_wide = lab_wide.rename(columns=LAB_ITEMIDS).reset_index()
        lab_wide.to_pickle(cache)
    return adm.merge(lab_wide, on="hadm_id", how="left")


def load_bmi(adm: pd.DataFrame) -> pd.DataFrame:
    """Most recent 'BMI (kg/m2)' per subject (omr)."""
    cache = PROCESSED_DIR / "bmi_subject.pkl"
    if cache.exists():
        bmi = pd.read_pickle(cache)
    else:
        omr = pd.read_csv(HOSP_DIR / "omr.csv.gz")
        bmi = omr[omr["result_name"] == "BMI (kg/m2)"].copy()
        bmi["chartdate"] = pd.to_datetime(bmi["chartdate"])
        bmi["bmi"] = pd.to_numeric(bmi["result_value"], errors="coerce")
        bmi = bmi.dropna(subset=["bmi"])
        bmi = bmi.sort_values("chartdate").groupby("subject_id").tail(1)[["subject_id", "bmi"]]
        bmi.to_pickle(cache)
    return adm.merge(bmi, on="subject_id", how="left")


# ---------------------------------------------------------------------------
# Phase 2: echo -> echo features + proxy labels
# ---------------------------------------------------------------------------
def load_echo_studies() -> pd.DataFrame:
    """One row per TTE study with raw echo measurements (long->wide)."""
    cache = PROCESSED_DIR / "echo_studies.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    cols = ["subject_id", "measurement_id", "measurement_datetime", "test_type",
            "measurement", "result"]
    chunks = []
    reader = pd.read_csv(ECHO_RAW_PATH, usecols=cols, chunksize=2_000_000, dtype={"result": str})
    for chunk in reader:
        chunk = chunk[(chunk["test_type"] == "tte") & (chunk["measurement"].isin(ECHO_TARGETS))]
        if len(chunk):
            chunks.append(chunk)
    echo = pd.concat(chunks, ignore_index=True)

    meta = echo.groupby(["subject_id", "measurement_id"])["measurement_datetime"].first().reset_index()
    wide = echo.pivot_table(
        index=["subject_id", "measurement_id"], columns="measurement", values="result", aggfunc="first"
    ).reset_index()
    wide = wide.merge(meta, on=["subject_id", "measurement_id"], how="left")
    wide["measurement_datetime"] = pd.to_datetime(wide["measurement_datetime"])
    wide.to_pickle(cache)
    return wide


def derive_echo_features_and_labels(echo: pd.DataFrame) -> pd.DataFrame:
    """Numeric echo features + ph_label / cor_pulmonale_label per study."""
    out = echo[["subject_id", "measurement_id", "measurement_datetime"]].copy()

    out["echo_tr_velocity"] = pd.to_numeric(echo.get("tr_velocity"), errors="coerce")
    out["echo_rvsp"] = pd.to_numeric(echo.get("tr_mmhg"), errors="coerce")
    out["echo_tapse"] = pd.to_numeric(echo.get("tapse"), errors="coerce")
    out["echo_rv_basal_diameter"] = pd.to_numeric(echo.get("rv_diam"), errors="coerce")
    out["echo_pericardial_effusion"] = (
        echo.get("pericardial_effusion").map(PERICARDIAL_MAP)
    )

    # --- ph_label (echo-PH proxy) ---
    sev = echo.get("phtn_severity")
    sev_ph = np.where(
        sev.isin(PH_SEVERITY_POS), 1.0,
        np.where(sev.isin(PH_SEVERITY_NEG), 0.0, np.nan),
    )
    tr_mmhg = out["echo_rvsp"].to_numpy()
    pres_ph = np.where(tr_mmhg >= PH_TR_MMHG, 1.0, np.where(tr_mmhg < PH_TR_MMHG, 0.0, np.nan))
    out["ph_label"] = _combine_or(sev_ph, pres_ph)

    # --- cor_pulmonale_label (echo RV-dysfunction proxy) ---
    rv = echo.get("rv_function")
    rv_sev = np.where(
        rv.isin(RV_FUNCTION_POS), 1.0,
        np.where(rv.isin(RV_FUNCTION_NEG), 0.0, np.nan),
    )
    rv_diam = out["echo_rv_basal_diameter"].to_numpy()
    rv_enl = np.where(
        rv_diam > RV_DIAM_ENLARGED_CM, 1.0,
        np.where(rv_diam <= RV_DIAM_ENLARGED_CM, 0.0, np.nan),
    )
    out["cor_pulmonale_label"] = _combine_or(rv_sev, rv_enl)
    return out


def join_echo_to_admissions(adm: pd.DataFrame, echo_feat: pd.DataFrame) -> pd.DataFrame:
    """Attach the LAST echo study inside each admission (most recent before
    dischtime + 1d) to the admission row."""
    echo_feat = echo_feat.dropna(subset=["measurement_datetime"])
    adm_e = adm[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    merged = echo_feat.merge(adm_e, on="subject_id", how="inner")
    win = pd.Timedelta(days=1)
    merged = merged[
        (merged["measurement_datetime"] >= merged["admittime"] - win)
        & (merged["measurement_datetime"] <= merged["dischtime"] + win)
    ]
    merged = merged.sort_values("measurement_datetime").drop_duplicates(
        "hadm_id", keep="last"
    )
    echo_cols = [
        "echo_tr_velocity", "echo_rvsp", "echo_tapse",
        "echo_rv_basal_diameter", "echo_pericardial_effusion",
        "ph_label", "cor_pulmonale_label",
    ]
    keep = ["hadm_id"] + echo_cols
    merged = merged[keep]
    # flag echo availability
    merged["avail_echo"] = 1
    return merged


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def smoke_check_admissions(adm: pd.DataFrame) -> None:
    print("\n=== [smoke] admissions ===")
    print(f"n_admissions        = {len(adm)}")
    print(f"n_patients          = {adm['subject_id'].nunique()}")
    print(f"visits/patient      = {adm.groupby('subject_id')['visit_index'].max().mean():.2f} (max)")
    print(f"age mean/std        = {adm['age'].mean():.1f} / {adm['age'].std():.1f}")
    print(f"sex (F=1) rate      = {adm['sex'].mean():.3f}")
    print(f"ICD PH prevalence   = {adm['icd_ph'].mean():.4f}")
    print(f"ICD cor pulm prev   = {adm['icd_corpulmonale'].mean():.4f}")
    for c in ["nt_probnp", "hemoglobin", "uric_acid", "bmi"]:
        print(f"  missing {c:<12}= {adm[c].isna().mean():.3f}")


def smoke_check_echo(echo_feat: pd.DataFrame) -> None:
    print("\n=== [smoke] echo ===")
    print(f"n_tte_studies       = {len(echo_feat)}")
    print(f"ph_label prevalence = {echo_feat['ph_label'].mean():.4f} "
          f"(non-null {echo_feat['ph_label'].notna().sum()})")
    print(f"cor_pulm prevalence = {echo_feat['cor_pulmonale_label'].mean():.4f} "
          f"(non-null {echo_feat['cor_pulmonale_label'].notna().sum()})")
    for c in ["echo_tr_velocity", "echo_rvsp", "echo_tapse",
              "echo_rv_basal_diameter", "echo_pericardial_effusion"]:
        print(f"  missing {c:<26}= {echo_feat[c].isna().mean():.3f}")


def run_phase12() -> pd.DataFrame:
    """Build the admission-level structured + ICD table (Phase 1), and the
    per-study echo table with labels (Phase 2). Returns the admissions table."""
    print("== Phase 1: admissions + ICD + labs + BMI ==")
    adm = load_admissions()
    adm = load_icd_labels(adm)
    adm = load_labs(adm)
    adm = load_bmi(adm)
    smoke_check_admissions(adm)

    print("\n== Phase 2: echo -> features + proxy labels ==")
    echo = load_echo_studies()
    echo_feat = derive_echo_features_and_labels(echo)
    smoke_check_echo(echo_feat)

    adm.to_pickle(PROCESSED_DIR / "admissions_structured.pkl")
    echo_feat.to_pickle(PROCESSED_DIR / "echo_features.pkl")
    print("\nSaved admissions_structured.pkl + echo_features.pkl")
    return adm


if __name__ == "__main__":
    run_phase12()
