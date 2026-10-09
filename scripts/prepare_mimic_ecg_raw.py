"""MIMIC-IV-ECG 原始波形 -> (12, 2500) float16，供 CNN 编码器（Module A/B 共用）。

复用 Stage1 的 ecg_meta.pkl + Stage2 的匹配逻辑，只对 echo 人群（ph_label 有定义）
里 avail_ecg=1 的 admission 抽原始波形（约 39k），落盘 ecg_raw_waveforms.pkl =
{hadm_id: np.ndarray((12, 2500), float16)}（约 2.3GB）。

用法（项目根目录）：
    python scripts/prepare_mimic_ecg_raw.py
"""

from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from datasets.mimic_ecg import read_record_raw  # noqa: E402
from datasets.mimic_loader import PROCESSED_DIR, MIMIC_DIR, join_echo_to_admissions  # noqa: E402

ECG_ZIP = MIMIC_DIR / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
META_PKL = PROCESSED_DIR / "ecg_meta.pkl"
ADM_PKL = PROCESSED_DIR / "admissions_structured.pkl"
ECHO_PKL = PROCESSED_DIR / "echo_features.pkl"
OUT_PKL = PROCESSED_DIR / "ecg_raw_waveforms.pkl"


def _zip_path(record: str, subject_id: int) -> str:
    bucket = subject_id // 10000
    return f"{PREFIX}files/p{bucket}/p{subject_id}/s{record}/{record}"


def main() -> None:
    adm = pd.read_pickle(ADM_PKL)
    echo_feat = pd.read_pickle(ECHO_PKL)
    meta = pd.read_pickle(META_PKL)

    # echo 人群（avail_echo=1 的 hadm_id）——只抽这些 admission 的波形
    echo_join = join_echo_to_admissions(adm, echo_feat)
    echo_hadm = set(echo_join["hadm_id"].astype(int))
    print(f"echo 人群 hadm_id = {len(echo_hadm)}")

    # 匹配 ECG -> admission（复用 Stage2 逻辑，时间窗 ±1d，每 admission 取最后一条）
    win = pd.Timedelta(days=1)
    m = meta.merge(adm[["subject_id", "hadm_id", "admittime", "dischtime"]],
                   on="subject_id", how="inner")
    m = m[(m["datetime"] >= m["admittime"] - win) & (m["datetime"] <= m["dischtime"] + win)]
    m = m.sort_values("datetime").drop_duplicates("hadm_id", keep="last")
    m = m[m["hadm_id"].isin(echo_hadm)].reset_index(drop=True)
    print(f"echo 人群里有 ECG 的 (admission, ECG) 匹配 = {len(m)} 条")

    zf = zipfile.ZipFile(ECG_ZIP)
    out: dict[int, np.ndarray] = {}
    n_fail = 0
    for i, row in enumerate(m.itertuples()):
        base = _zip_path(row.record, row.subject_id)
        try:
            hea = zf.read(base + ".hea").decode("utf-8", errors="replace")
            dat = zf.read(base + ".dat")
            out[int(row.hadm_id)] = read_record_raw(dat, hea)  # (12, 2500) float16
        except Exception:
            n_fail += 1
        if (i + 1) % 5000 == 0:
            print(f"  处理 {i + 1}/{len(m)}（fail={n_fail}）…", flush=True)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done: {len(out)} 条波形 -> {OUT_PKL.name}（fail={n_fail}）")

    # 冒烟
    arr = np.stack(list(out.values())[:100])
    print(f"  样例 shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.3f} max={arr.max():.3f} nan={int(np.isnan(arr).sum())}")
    n_const = sum(1 for a in out.values() if float(np.std(a)) < 1e-6)
    print(f"  近常量记录（std<1e-6）={n_const}")


if __name__ == "__main__":
    main()
