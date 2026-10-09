"""MIMIC-IV-ED 2.2 -> vitals 可选模态（6 个连续 triage 生命体征）。

访视锚点 = 医院 admission。ED triage 体征经 `edstays.stay_id -> hadm_id` 关联到
admission；仅保留 hadm_id 非空的 ED stay（即确实入院者），"treat-and-release"
（未入院）的 ED stay 丢弃。

6 特征构成 "vitals" 可选模态（显式 mask，缺失时 NaN -> 组装期置 0 + avail_vitals=0）：
  vitals_temperature / vitals_heartrate / vitals_resprate /
  vitals_o2sat / vitals_sbp / vitals_dbp

落盘 extracted/processed/ed_vitals_admission.pkl（hadm_id -> 6 特征 + avail_vitals）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from datasets.mimic_loader import PROCESSED_DIR  # noqa: E402

MIMIC_DIR = Path(__file__).resolve().parents[1] / "mimic"
ED_DIR = MIMIC_DIR / "mimic-iv-ed-2.2" / "mimic-iv-ed-2.2" / "ed"

VITALS_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

# ED 列名 -> 项目 schema 特征名
VITALS_RENAME = {
    "temperature": "vitals_temperature",
    "heartrate": "vitals_heartrate",
    "resprate": "vitals_resprate",
    "o2sat": "vitals_o2sat",
    "sbp": "vitals_sbp",
    "dbp": "vitals_dbp",
}


def load_vitals() -> pd.DataFrame:
    """返回 admission 级 vitals 表（hadm_id, 6 特征, avail_vitals=1）。"""
    cache = PROCESSED_DIR / "ed_vitals_admission.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    triage = pd.read_csv(ED_DIR / "triage.csv.gz")
    edstays = pd.read_csv(
        ED_DIR / "edstays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime"],
        parse_dates=["intime"],
    )
    edstays = edstays.dropna(subset=["hadm_id"])
    edstays["hadm_id"] = edstays["hadm_id"].astype(int)

    merged = triage.merge(edstays, on=["subject_id", "stay_id"], how="inner")
    # 每个 hadm_id 取 intime 最早的一条 triage 体征（一个 admission 通常对应一条 ED stay）
    merged = merged.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    for c in VITALS_COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    out = merged[["hadm_id"] + VITALS_COLS].rename(columns=VITALS_RENAME)
    out["avail_vitals"] = 1
    out = out.reset_index(drop=True)
    out.to_pickle(cache)
    return out


def smoke_check(vitals: pd.DataFrame) -> None:
    print("\n=== [smoke] ED vitals ===")
    print(f"n_admissions with ED vitals = {len(vitals)}")
    for c in ["vitals_temperature", "vitals_heartrate", "vitals_resprate",
              "vitals_o2sat", "vitals_sbp", "vitals_dbp"]:
        v = vitals[c]
        print(f"  {c:<24} n={v.notna().sum():>6}  mean={v.mean():.2f}  "
              f"p5={v.quantile(.05):.1f}  p95={v.quantile(.95):.1f}")


if __name__ == "__main__":
    smoke_check(load_vitals())
