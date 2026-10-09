"""MIMIC-CXR 2.1.0 -> cxr 可选模态（3 个二值发现，从放射报告否定感知提取）。

两段式：
  1) load_study_findings(): 解析 `mimic-cxr-reports.zip` 的 22.8 万份报告，用
     否定感知规则从 FINDINGS/IMPRESSION 提取每份 study 的 3 个二值发现
     （cxr_cardiomegaly / cxr_pleural_effusion / cxr_pulmonary_edema）。落盘
     mimic/extracted/processed/cxr_study_findings.pkl（study_id 键）。此步无需 metadata。
  2) join_cxr_to_admissions(): 当前目录无 metadata.csv.gz（无 StudyDate/StudyTime），
     无法做 admission 时间窗关联，退化为**患者级（subject_id）关联**：任一 study
     阳性 -> 该患者所有 admission 的发现=1，avail_cxr=1（有 ≥1 份 CXR 报告）。落盘
     cxr_findings_admission.pkl（hadm_id 键）。

时序泄漏提示：subject 级粗关联不区分 study 时间，可能把「未来」CXR 发现挂到较早
admission；对肺心病动态预测构成潜在未来泄漏，报告显式标注此局限。

近似标签声明：规则提取 ≠ CheXpert/NegBio 精度；仅作输入特征，报告显式标注。
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from datasets.mimic_loader import MIMIC_DIR, PROCESSED_DIR  # noqa: E402

CXR_DIR = MIMIC_DIR / "MIMIC-CXR Database 2.1.0"
REPORTS_ZIP = CXR_DIR / "mimic-cxr-reports.zip"

# 报告路径形如 files/p<subject>/s<study>.txt
_PATH_RE = re.compile(r"p(?P<subject>\d+)/s(?P<study>\d+)\.txt$")

# 否定词（出现在提及之前 -> 否定；出现在之后 -> 否定）
_NEG_BEFORE = re.compile(
    r"\b(no|not|without|free of|absent|no evidence of|without evidence of|negative for)\b",
    re.IGNORECASE,
)
_NEG_AFTER = re.compile(r"\b(absent|not seen|not present|negative)\b", re.IGNORECASE)

# 各发现的阳性提及模式
POS_PATTERNS: dict[str, list[str]] = {
    "cxr_cardiomegaly": [
        r"cardiomegaly", r"enlarged heart", r"heart (?:is |appears )?enlarged",
        r"enlargement of the heart", r"enlarged cardiac silhouette",
        r"cardiac enlargement", r"enlarged cardiomediastinal silhouette",
        r"enlarged heart size",
    ],
    "cxr_pleural_effusion": [
        r"(?<!pericardial )(?<!pericardial\-)effusion", r"pleural effusion",
    ],
    "cxr_pulmonary_edema": [
        r"pulmonary edema", r"pulmonary oedema", r"interstitial edema",
        r"alveolar edema", r"airspace edema", r"pulmonary vascular congestion",
        r"pulmonary venous congestion", r"vascular congestion",
    ],
    # ---- 审计新增：PH 直接定性信号（此前未提取的自由文本） ----
    "cxr_pulmonary_hypertension": [
        r"pulmonary arterial hypertension", r"pulmonary hypertension",
        r"pulmonary artery hypertension", r"pulmonary-artery hypertension",
    ],
    "cxr_pa_enlargement": [
        r"pulmonary artery enlargement", r"enlargement of the pulmonary artery",
        r"prominent pulmonary artery", r"dilated pulmonary artery",
        r"enlarged pulmonary artery", r"enlarged main pulmonary artery",
        r"prominent main pulmonary artery", r"dilated main pulmonary artery",
        r"pulmonary artery (?:is |appears )?(?:enlarged|dilated|prominent)",
    ],
    "cxr_rv_enlargement": [
        r"right ventricular enlargement", r"right ventricular hypertrophy",
        r"right ventricular dilatation", r"right ventricular dilation",
        r"enlarged right ventricle", r"dilated right ventricle",
        r"right ventricle (?:is |appears )?(?:enlarged|dilated)",
        r"RV enlargement", r"RV dilatation",
    ],
}
POS_RE = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in POS_PATTERNS.items()}
FINDING_COLS = list(POS_PATTERNS.keys())


def _is_negated(text: str, start: int, end: int) -> bool:
    """提及前后窗口内是否存在否定词。"""
    before = text[max(0, start - 60):start]
    after = text[end:end + 40]
    if _NEG_BEFORE.search(before):
        return True
    if _NEG_AFTER.search(after):
        return True
    return False


def extract_findings(text: str) -> dict[str, float]:
    """单份报告 -> 3 个发现（1=阳性 / 0=阴性或未提及）。"""
    # 只取 FINDINGS 与 IMPRESSION 段（避免 HISTORY 里"既往病史"污染）
    seg = _findings_impression(text)
    out: dict[str, float] = {}
    for name, pats in POS_RE.items():
        val = 0.0
        for p in pats:
            hit = False
            for m in p.finditer(seg):
                if not _is_negated(seg, m.start(), m.end()):
                    hit = True
                    break
            if hit:
                val = 1.0
                break
        out[name] = val
    return out


def _findings_impression(text: str) -> str:
    """截取 FINDINGS 与 IMPRESSION 段（缺省则用全文）。"""
    m = re.search(r"FINDINGS\s*:?\s*(.*?)(?:IMPRESSION|$)", text, re.IGNORECASE | re.DOTALL)
    parts = []
    if m:
        parts.append(m.group(1))
    m2 = re.search(r"IMPRESSION\s*:?\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)
    if m2:
        parts.append(m2.group(1))
    return "\n".join(parts) if parts else text


def load_study_findings(cache_name: str = "cxr_study_findings.pkl") -> pd.DataFrame:
    """study_id -> 各发现（缓存）。cache_name 区分 v1(3)/v2(6) 发现集。"""
    cache = PROCESSED_DIR / cache_name
    if cache.exists():
        return pd.read_pickle(cache)

    rows: list[dict] = []
    with zipfile.ZipFile(REPORTS_ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".txt")]
        for i, n in enumerate(names):
            pm = _PATH_RE.search(n)
            if not pm:
                continue
            study_id = int(pm.group("study"))
            subject_id = int(pm.group("subject"))
            try:
                text = z.read(n).decode("utf-8", errors="replace")
            except Exception:
                continue
            feats = extract_findings(text)
            feats["subject_id"] = subject_id
            feats["study_id"] = study_id
            rows.append(feats)
            if (i + 1) % 50000 == 0:
                print(f"  ... parsed {i + 1}/{len(names)} reports")

    out = pd.DataFrame(rows)[["subject_id", "study_id", *FINDING_COLS]]
    out.to_pickle(cache)
    return out


def join_cxr_to_admissions(
    adm: pd.DataFrame,
    study_cache: str = "cxr_study_findings.pkl",
    out_cache: str = "cxr_findings_admission.pkl",
) -> pd.DataFrame:
    """把 study 级 CXR 发现按 subject_id（患者级）聚合后关联到 admission。

    无 metadata.csv.gz（无 StudyDate/StudyTime）→ 无法做 admission 时间窗关联，
    退化为患者级关联：任一 study 阳性 -> 该患者所有 admission 的发现=1。

    返回 admission 级表（hadm_id, FINDING_COLS, avail_cxr=1）。
    """
    findings = load_study_findings(study_cache)
    feat_cols = list(FINDING_COLS)
    # 患者级聚合：任一 study 阳性 = 1（0/1 取 max）
    pt = findings.groupby("subject_id")[feat_cols].max().reset_index()
    pt["avail_cxr"] = 1
    # subject_id -> hadm_id：该患者所有 admission 共享同一患者级发现
    agg = adm[["subject_id", "hadm_id"]].merge(pt, on="subject_id", how="inner")
    agg = agg[["hadm_id", *feat_cols, "avail_cxr"]].drop_duplicates("hadm_id")
    agg.to_pickle(PROCESSED_DIR / out_cache)
    return agg


def smoke_check(findings: pd.DataFrame) -> None:
    print("\n=== [smoke] CXR study findings ===")
    print(f"n_studies = {len(findings)}")
    for c in ["cxr_cardiomegaly", "cxr_pleural_effusion", "cxr_pulmonary_edema"]:
        print(f"  {c:<24} prevalence = {findings[c].mean():.4f}")


if __name__ == "__main__":
    smoke_check(load_study_findings())
