"""评估全部 2^4=16 种模态组合（在 {ecg,echo,vitals,cxr} 上）的性能矩阵。

把现有只覆盖 8 组合的 matrix（structured + 4 单模态 + ecg+echo + ecg+echo+vitals + all）
补齐到完整 16 组合，供「Full Modality Synergy Matrix」热力图使用。

关键点（诚实口径）：
  - 纯推理，不重训：加载已保存 checkpoint，用 evaluate_combination（强制掩码）评估。
  - 每个格子是「实际具备该组合所需模态的子人群」上的 AUROC（非同一固定人群），
    缺失的其它模态内容置零、mask 强制。这与现有 matrix 口径完全一致。
  - ERUF 是 4 模态（optional=['ecg','echo','vitals','cxr']，已去 ct），
    基线是 5 模态（optional=['ct','ecg','echo','vitals','cxr']，ct 恒缺失），
    故各自组合掩码长度不同（ERUF 4 位 / 基线 5 位且 ct 位恒 0）。
  - 结果会与既有 8 组合 summary JSON 对照打印，确认一致后才可信。

用法：
  python scripts/eval_synergy_matrix_16.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from core.config import PROJECT_ROOT, load_config  # noqa: E402
from datasets.ph_experiment import load_fixed_mimic  # noqa: E402
from evaluation.metrics import auroc  # noqa: E402
from evaluation.missing_modality_eval import evaluate_combination  # noqa: E402
from models.encoders import MLPEncoder  # noqa: E402
from models.image_encoders import ResECGWaveformEncoder  # noqa: E402
from models.eruf import ERUFModel  # noqa: E402
from models.medfuse import MedFuseModel  # noqa: E402
from models.muse import MUSEModel  # noqa: E402
from models.qamoe import QAMoEModel  # noqa: E402
from models.drfuse import DrFuseModel  # noqa: E402
from models.moddrop import ModDropContrastiveModel  # noqa: E402
from models.base import make_baseline_encoders  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "mimic" / "ph_experiment"
RELEVANT = ["ecg", "echo", "vitals", "cxr"]          # 4 个相关可选模态（去 ct）
OPTIONAL5 = ["ct", "ecg", "echo", "vitals", "cxr"]    # 基线的 5 模态顺序


class NoCTDataset(Dataset):
    """切掉 ct：mask 5 列 → 4 列，inputs 移除 'ct' 键（与 train_eruf 一致）。"""

    def __init__(self, base: Dataset):
        self.base = base
        self.y = getattr(base, "y", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        inputs, mask, y = self.base[i]
        inputs = {k: v for k, v in inputs.items() if k != "ct"}
        return inputs, mask[1:], y


def make_eruf_encoders(optional: list[str], d: int, dropout: float) -> dict:
    return {
        "structured": MLPEncoder(6, (64, 128), d, dropout),
        "ecg": ResECGWaveformEncoder(latent_dim=d, dropout=dropout),
        "echo": MLPEncoder(3, (32, 64), d, dropout),
        "vitals": MLPEncoder(6, (32, 64), d, dropout),
        "cxr": MLPEncoder(3, (32, 64), d, dropout),
    }


def build_combos() -> list[tuple[str, list[int]]]:
    """16 组合按位掩码升序：bits = [ecg, echo, vitals, cxr]。"""
    out = []
    for m in range(16):
        bits = [(m >> i) & 1 for i in range(4)]
        name = "+".join(RELEVANT[i] for i in range(4) if bits[i]) or "structured"
        out.append((name, bits))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    mcfg = load_config("model")["model"]
    d = int(mcfg.get("latent_dim", 128))
    dropout = float(mcfg.get("dropout", 0.2))
    batch = 256

    train_ds, val_ds, test_ds, meta = load_fixed_mimic()
    test_eruf = NoCTDataset(test_ds)
    print(f"测试集 n={meta['n_test']}  prev={meta['prev']:.4f}  d={d} dropout={dropout}", flush=True)

    combos = build_combos()
    models: dict[str, dict[str, float | None]] = {}

    # ---- ERUF（4 模态，rectifier 开启，与 eruf_full_summary 的 matrix 口径一致）----
    print("\n=== ERUF ===", flush=True)
    eruf = ERUFModel(make_eruf_encoders(RELEVANT, d, dropout), RELEVANT,
                     d=d, dropout=dropout, use_delta=True).to(device)
    eruf.load_state_dict(torch.load(OUT_DIR / "eruf_full.pt", map_location=device))
    eruf.use_rectifier = True
    eruf.eval()
    models["ERUF"] = {}
    for name, bits in combos:
        y, p = evaluate_combination(eruf, test_eruf, bits, device, batch)
        models["ERUF"][name] = float(auroc(y, p)) if len(y) else None
    del eruf
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 5 个基线（5 模态，ct 位恒 0）----
    baselines = {
        "QA-MoE": (QAMoEModel, "qamoe.pt"),
        "DrFuse": (DrFuseModel, "drfuse.pt"),
        "ModDrop": (ModDropContrastiveModel, "moddrop.pt"),
        "MUSE": (MUSEModel, "muse.pt"),
        "MedFuse": (MedFuseModel, "medfuse.pt"),
    }
    for name, (cls, ckpt) in baselines.items():
        print(f"\n=== {name} ===", flush=True)
        enc = make_baseline_encoders(OPTIONAL5, d, dropout)
        model = cls(enc, OPTIONAL5, d=d, dropout=dropout).to(device)
        model.load_state_dict(torch.load(OUT_DIR / ckpt, map_location=device))
        model.eval()
        models[name] = {}
        for cname, bits in combos:
            vec5 = [0] + bits                       # ct=0, ecg, echo, vitals, cxr
            y, p = evaluate_combination(model, test_ds, vec5, device, batch)
            models[name][cname] = float(auroc(y, p)) if len(y) else None
        del model
        torch.cuda.empty_cache()
        gc.collect()

    out = {"combos": [c[0] for c in combos], "models": models}
    dst = OUT_DIR / "synergy_matrix_16.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n已保存 {dst}", flush=True)

    # ---- 与既有 8 组合 summary JSON 对照，确认口径一致 ----
    print("\n=== 对照既有 8 组合 ===", flush=True)
    refs = {
        "ERUF": ("eruf_full_summary.json", "result"),
        "QA-MoE": ("qamoe_summary.json", "baselines", "qamoe"),
        "DrFuse": ("baselines_summary.json", "baselines", "drfuse"),
        "ModDrop": ("baselines_summary.json", "baselines", "moddrop"),
        "MUSE": ("baselines_summary.json", "baselines", "muse"),
        "MedFuse": ("baselines_summary.json", "baselines", "medfuse"),
    }
    for name, (fname, *path) in refs.items():
        ref = json.load(open(OUT_DIR / fname, encoding="utf-8"))
        for p in path:
            ref = ref[p]
        ref_matrix = ref["matrix"]
        bad = []
        for cname in ["structured", "ecg", "echo", "vitals", "cxr",
                      "ecg+echo", "ecg+echo+vitals", "all"]:
            new_v = models[name].get(cname)
            old_v = ref_matrix.get(cname)
            if new_v is None or old_v is None:
                bad.append((cname, new_v, old_v))
                continue
            if abs(new_v - old_v) > 1e-3:
                bad.append((cname, new_v, old_v))
        status = "OK" if not bad else f"MISMATCH {bad}"
        print(f"  [{name}] {status}", flush=True)


if __name__ == "__main__":
    main()
