"""预测工具：模型在给定 DataLoader 上的逐样本风险概率。

``predict_probs`` 为训练/评估脚本共用：把模型切 eval、前向 ``sigmoid(model(inputs, mask))``，
返回 (y_true, y_prob)。自然缺失下评估（缺失模态由 mask 显式给出，模型内部处理）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """自然缺失下的预测，返回 (y_true, y_prob)。"""
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for inputs, mask, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask = mask.to(device)
        p = torch.sigmoid(model(inputs, mask)).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)
