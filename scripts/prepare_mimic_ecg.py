"""Phase 3 — MIMIC-IV-ECG 波形 -> 4 个 ECG 特征，回填到 admission。

两阶段：
  Stage 1 (meta)  : 用 zipfile 直读全部 .hea（不落盘），解析 record/subject_id/
                    datetime -> ecg_meta.pkl（约 80 万条）。
  Stage 2 (match) : 载入 admissions_structured.pkl，按 subject_id + 时间窗把 ECG
                    匹配到 admission（每 admission 取窗内最后一条），读 .dat 算 4
                    特征 -> ecg_admission_features.pkl（hadm_id + 4 特征）。

用法（项目根目录）：
    python scripts/prepare_mimic_ecg.py meta     # Stage 1（可独立跑）
    python scripts/prepare_mimic_ecg.py match    # Stage 2（需先跑 Phase 1/2）
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from datasets.mimic_ecg import parse_hea, process_record  # noqa: E402
from datasets.mimic_loader import PROCESSED_DIR, MIMIC_DIR  # noqa: E402

ECG_ZIP = MIMIC_DIR / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
META_PKL = PROCESSED_DIR / "ecg_meta.pkl"
OUT_PKL = PROCESSED_DIR / "ecg_admission_features.pkl"


def _zip_path(record: str, subject_id: int) -> str:
    """从 record_id + subject_id 重构 zip 内路径（files/p{bucket}/p{subj}/s{rec}/{rec}）。"""
    bucket = subject_id // 10000
    return f"{PREFIX}files/p{bucket}/p{subject_id}/s{record}/{record}"


def stage1_meta() -> None:
    """读全部 .hea -> 元数据 (record, subject_id, datetime)。"""
    print(f"Stage 1: 解析 .hea 元数据（zip: {ECG_ZIP.name}）……")
    zf = zipfile.ZipFile(ECG_ZIP)
    rows = []
    n_hea = 0
    for info in zf.infolist():
        if not info.filename.endswith(".hea"):
            continue
        n_hea += 1
        text = zf.read(info).decode("utf-8", errors="replace")
        try:
            h = parse_hea(text)
        except Exception:
            continue
        rows.append((h["record"], h["subject_id"], h["datetime"]))
        if n_hea % 100000 == 0:
            print(f"  parsed {n_hea} .hea …")
    meta = pd.DataFrame(rows, columns=["record", "subject_id", "datetime"])
    meta.to_pickle(META_PKL)
    print(f"  done: {len(meta)} records, {meta['subject_id'].nunique()} subjects -> {META_PKL.name}")


def stage2_match() -> None:
    """匹配 ECG 到 admission，算特征。"""
    adm_pkl = PROCESSED_DIR / "admissions_structured.pkl"
    if not adm_pkl.exists():
        raise SystemExit("未找到 admissions_structured.pkl，请先跑 Phase 1/2。")
    meta = pd.read_pickle(META_PKL)
    adm = pd.read_pickle(adm_pkl)
    print(f"Stage 2: 匹配 ECG -> admission（meta={len(meta)}，adm={len(adm)}）……")

    win = pd.Timedelta(days=1)
    m = meta.merge(adm[["subject_id", "hadm_id", "admittime", "dischtime"]],
                   on="subject_id", how="inner")
    m = m[(m["datetime"] >= m["admittime"] - win) & (m["datetime"] <= m["dischtime"] + win)]
    m = m.sort_values("datetime").drop_duplicates("hadm_id", keep="last")
    print(f"  匹配到 {len(m)} 条 (admission, ECG)，覆盖 {m['hadm_id'].nunique()} admission")

    zf = zipfile.ZipFile(ECG_ZIP)
    feats = []
    for i, row in enumerate(m.itertuples()):
        base = _zip_path(row.record, row.subject_id)
        try:
            hea = zf.read(base + ".hea").decode("utf-8", errors="replace")
            dat = zf.read(base + ".dat")
            f = process_record(dat, hea)
        except Exception:
            continue
        feats.append((row.hadm_id, f["ecg_right_axis_degree"], f["ecg_rs_v1"],
                      f["ecg_rv1_sv5"], f["ecg_p_pulmonale"]))
        if (i + 1) % 20000 == 0:
            print(f"  处理 {i + 1}/{len(m)} …")

    out = pd.DataFrame(feats, columns=[
        "hadm_id", "ecg_right_axis_degree", "ecg_rs_v1",
        "ecg_rv1_sv5", "ecg_p_pulmonale",
    ])
    out["avail_ecg"] = 1
    out.to_pickle(OUT_PKL)
    print(f"  done: {len(out)} admission 有 ECG 特征 -> {OUT_PKL.name}")
    _smoke(out)


def _smoke(out: pd.DataFrame) -> None:
    print("\n=== [ecg smoke] ===")
    for c in ["ecg_right_axis_degree", "ecg_rs_v1", "ecg_rv1_sv5", "ecg_p_pulmonale"]:
        v = out[c]
        print(f"  {c:<26} mean={v.mean():.3f} std={v.std():.3f} "
              f"min={v.min():.2f} max={v.max():.2f}")
    # 临床合理性：右偏轴比例、RV1+SV5>1.05mV（右室肥厚）比例、P 波>0.25mV（P 肺型）比例
    print(f"  右偏轴 (>+100°) 比例     = {(out['ecg_right_axis_degree'] > 100).mean():.3f}")
    print(f"  RV1+SV5 > 1.05mV 比例   = {(out['ecg_rv1_sv5'] > 1.05).mean():.3f}")
    print(f"  P(II) > 0.25mV 比例      = {(out['ecg_p_pulmonale'] > 0.25).mean():.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "meta"
    if mode == "meta":
        stage1_meta()
    elif mode == "match":
        stage2_match()
    else:
        raise SystemExit("usage: prepare_mimic_ecg.py [meta|match]")
