"""通用 MLP 特征编码器。

[Research Baseline] 第一阶段各模态编码器共用此 MLP 结构；
原始影像编码器（3D CNN / waveform CNN / video encoder）需真实数据，后续替换。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """多层感知机编码器：input_dim -> hidden_dims... -> output_dim。

    每个隐层后接激活 + Dropout，末层线性映射到 output_dim（不带激活）。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 256),
        output_dim: int = 128,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
