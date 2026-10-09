"""YAML 配置加载。

用法：
    cfg = load_config("model")["model"]   # 读取 configs/model.yaml 的 model 节
    d = int(cfg.get("latent_dim", 128))
"""
from __future__ import annotations

from pathlib import Path

import yaml

# 项目根目录 = 本文件向上两级（core/config.py -> core -> 根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """加载 configs/<name>.yaml 并返回顶层字典。"""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
