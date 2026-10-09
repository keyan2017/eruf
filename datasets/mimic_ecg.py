"""MIMIC-IV-ECG waveform DSP (pure numpy, no new dependencies).

Reads a WFDB record (.hea + .dat) and computes 4 scalar ECG features used by
the project schema:

  ecg_right_axis_degree : frontal QRS axis from net deflection of leads I / aVF
  ecg_rs_v1            : R/S ratio in V1
  ecg_rv1_sv5          : R(V1) + S(V5) in mV (Sokolow-Lyon RV-hypertrophy term)
  ecg_p_pulmonale      : P-wave amplitude in lead II in mV

The .dat files are 12-lead, 500 Hz, 10 s, int16, multiplexed (frame-by-frame)
samples. Physical mV = sample / gain, gain=200.0/mV, baseline 0 (format 16).

The measurements are APPROXIMATE (median-beat amplitude) and are intended as
input features, not as a diagnostic-grade ECG reader. This is documented in the
comparison report.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# WFDB header / data parsing
# ---------------------------------------------------------------------------
HEADER_RE = re.compile(
    r"^(?P<record>\S+)\s+(?P<nleads>\d+)\s+(?P<fs>\d+)\s+(?P<nsamp>\d+)"
    r"\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<date>\d{2}/\d{2}/\d{4})"
)
SIG_RE = re.compile(
    r"^(?P<fname>\S+)\s+(?P<fmt>\d+)\s+(?P<gain>[\d.]+)(?:\((?P<base>-?[\d.]+)\))?/(?P<unit>\S+)"
    r"(?:\s+(?P<adcres>\d+))?(?:\s+(?P<adczero>\d+))?"
    r"(?:\s+(?P<init>-?\d+))?(?:\s+(?P<checksum>-?\d+))?"
    r"(?:\s+(?P<blocksize>\d+))?\s+(?P<lead>\S+)"
)


def parse_hea(text: str) -> dict:
    lines = text.strip().splitlines()
    m = HEADER_RE.match(lines[0])
    if not m:
        raise ValueError(f"cannot parse header line: {lines[0]!r}")
    leads, gains = [], []
    for ln in lines[1:]:
        s = SIG_RE.match(ln.strip())
        if not s:
            continue
        leads.append(s.group("lead"))
        gains.append(float(s.group("gain")))
    dt = datetime.strptime(
        f"{m.group('date')} {m.group('time')}", "%d/%m/%Y %H:%M:%S"
    )
    subject_id = None
    for ln in lines:
        mm = re.search(r"#\s*<subject_id>:\s*(\d+)", ln)
        if mm:
            subject_id = int(mm.group(1))
            break
    return {
        "record": m.group("record"),
        "n_leads": int(m.group("nleads")),
        "fs": int(m.group("fs")),
        "n_samples": int(m.group("nsamp")),
        "datetime": dt,
        "leads": leads,
        "gains": np.asarray(gains, dtype=np.float32),
        "subject_id": subject_id,
    }


def read_dat(dat: bytes, n_leads: int, n_samples: int) -> np.ndarray:
    """Return (n_samples, n_leads) float32 array in mV (gain applied)."""
    x = np.frombuffer(dat, dtype="<i2").astype(np.float32)
    if x.size != n_leads * n_samples:
        raise ValueError(
            f"dat size mismatch: {x.size} != {n_leads}*{n_samples}"
        )
    return x.reshape(n_samples, n_leads)


# 规范 12 导联顺序（CNN 需要跨记录一致的导联排列）
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]


def read_record_raw(dat: bytes, hea_text: str, target_fs: int = 250) -> np.ndarray:
    """读一条 ECG 原始波形 -> (12, n_samples//2) float16 mV（500Hz 抽稀到 250Hz）。

    复用 parse_hea + read_dat + gain；按 CANONICAL_LEADS 重排为规范 12 导联顺序
    （跨记录一致），供 1D CNN 编码器直接吃原始波形。非 12 导联记录抛 ValueError。
    """
    hdr = parse_hea(hea_text)
    x = read_dat(dat, hdr["n_leads"], hdr["n_samples"])  # (n_samples, n_leads)
    x = x / hdr["gains"][None, :]                        # mV
    idx = {name: i for i, name in enumerate(hdr["leads"])}
    if not set(CANONICAL_LEADS).issubset(idx):
        raise ValueError(f"missing canonical leads: {sorted(set(CANONICAL_LEADS) - set(idx))}")
    x = x[:, [idx[l] for l in CANONICAL_LEADS]]          # (n_samples, 12)
    x = x.T                                               # (12, n_samples)
    if target_fs == 250:
        x = x[:, ::2]                                     # 500Hz -> 250Hz
    return x.astype(np.float16)


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------
def median_filter(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding median filter (edge-padded). w must be odd."""
    if w % 2 == 0:
        w += 1
    n = len(x)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    idx = np.arange(w)[None, :] + np.arange(n)[:, None]
    return np.median(xp[idx], axis=1)


def detrend(x: np.ndarray, fs: int, w_ms: int = 200) -> np.ndarray:
    """Baseline-wander removal via boxcar-mean subtraction (O(n), 用于 R 峰检测)。"""
    w = int(round(fs * w_ms / 1000.0))
    w = max(w, 2)
    x = x.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    box = (cs[w:] - cs[:-w]) / w
    pad = w // 2
    base = np.concatenate([np.full(pad, box[0]), box, np.full(w - 1 - pad, box[-1])])
    return x - base[: len(x)]


def detect_r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    """R-peak indices on a single lead (already high-passed)."""
    thresh = 0.5 * np.max(x) if np.max(x) > 0 else 0.0
    if thresh <= 0:
        return np.array([], dtype=int)
    # local maxima above threshold
    above = x > thresh
    peaks = []
    i = 0
    n = len(x)
    min_dist = int(round(fs * 0.2))  # 200 ms refractory
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and x[j + 1] >= x[j]:
                j += 1
            peaks.append(j)
            i = j + min_dist
        else:
            i += 1
    return np.array(peaks, dtype=int)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _median_beat(leads: np.ndarray, peaks: np.ndarray, pre: int, post: int) -> np.ndarray:
    """(n_leads, pre+post) median beat aligned at R peaks."""
    win = pre + post
    beats = np.empty((len(peaks), leads.shape[1], win), dtype=np.float32)
    for k, p in enumerate(peaks):
        a, b = p - pre, p + post
        if a < 0 or b > leads.shape[0]:
            # pad with edge for out-of-range windows
            seg = np.pad(leads[max(a, 0): min(b, leads.shape[0])],
                         ((max(-a, 0), max(b - leads.shape[0], 0)), (0, 0)),
                         mode="edge")
            beats[k] = seg.T
        else:
            beats[k] = leads[a:b].T
    return np.median(beats, axis=0)


def compute_features(x: np.ndarray, leads: list[str], fs: int) -> dict:
    """x: (n_samples, n_leads) mV. Returns the 4 ECG features (+qc flags)."""
    out = {
        "ecg_right_axis_degree": np.nan,
        "ecg_rs_v1": np.nan,
        "ecg_rv1_sv5": np.nan,
        "ecg_p_pulmonale": np.nan,
        "n_r_peaks": 0,
    }
    idx = {name: i for i, name in enumerate(leads)}
    need = {"I", "II", "aVF", "V1", "V5"}
    if not need.issubset(idx):
        return out  # missing a required lead -> NaN

    # R-peak detection on lead II
    ii = detrend(x[:, idx["II"]], fs)
    peaks = detect_r_peaks(ii, fs)
    out["n_r_peaks"] = len(peaks)
    if len(peaks) < 3:
        return out  # too few beats -> unreliable

    # median beat (200ms pre, 320ms post -> 520ms window)
    pre = int(round(fs * 0.20))
    post = int(round(fs * 0.32))
    b = _median_beat(x, peaks, pre, post)  # (n_leads, win), R peak at index `pre`
    b = b - np.median(b, axis=1, keepdims=True)  # 每导联去除基线偏移

    # QRS window: 60ms before to 80ms after R peak
    q0 = pre - int(round(fs * 0.06))
    q1 = pre + int(round(fs * 0.08))
    # P window: 200ms..60ms before R peak
    p0 = pre - int(round(fs * 0.20))
    p1 = pre - int(round(fs * 0.06))

    def qrs_rs(lead):
        seg = b[idx[lead], q0:q1]
        r = float(np.max(seg))
        s = float(-np.min(seg))
        return max(r, 0.0), max(s, 0.0)

    r_i, s_i = qrs_rs("I")
    r_avf, s_avf = qrs_rs("aVF")
    net_i = r_i - s_i
    net_avf = r_avf - s_avf
    out["ecg_right_axis_degree"] = float(np.degrees(np.arctan2(net_avf, net_i)))

    r_v1, s_v1 = qrs_rs("V1")
    r_v5, s_v5 = qrs_rs("V5")
    out["ecg_rs_v1"] = float(r_v1 / (s_v1 + 0.05))
    out["ecg_rv1_sv5"] = float(r_v1 + s_v5)

    p_ii = float(np.max(np.abs(b[idx["II"], p0:p1])))
    out["ecg_p_pulmonale"] = p_ii
    return out


def process_record(dat: bytes, hea_text: str) -> dict:
    """End-to-end: parse .hea + .dat -> feature dict."""
    hdr = parse_hea(hea_text)
    x = read_dat(dat, hdr["n_leads"], hdr["n_samples"])
    # apply per-lead gain -> mV (gain in adc/mV)
    x = x / hdr["gains"][None, :]
    feats = compute_features(x, hdr["leads"], hdr["fs"])
    feats["record"] = hdr["record"]
    feats["datetime"] = hdr["datetime"]
    return feats
