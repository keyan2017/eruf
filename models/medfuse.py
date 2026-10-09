"""MedFuse 基线（LSTM 融合可变长模态 token 序列）—— 核心机制复现。

在特征化模态（无原始影像）上按方法描述复现核心机制（非逐行复现）：可用模态 latent 按
slot 序组成序列，LSTM 循环聚合，取「最后一个可用 token」的隐状态分类；缺失模态 latent
置零（等价于从序列中省略）。

接口：``forward(inputs, mask) -> logits``；``model.optional / model.keys`` 暴露。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.base import _Baseline, make_baseline_encoders


class MedFuseModel(_Baseline):
    """MedFuse 核心机制复现：LSTM 聚合可用模态 token 序列，取末个可用隐状态分类。

    原文为「time-series LSTM 编码 + 图像编码 + LSTM 融合模块」；此处各模态已统一为
    d 维 latent（同 backbone），融合模块用 LSTM 把 [structured, ct, ecg, echo, vitals,
    cxr] 的可用子序列循环聚合，末个「可用」token 的隐状态送入分类头（缺失槽 latent
    置零，等价于从序列中省略）。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2):
        super().__init__(encoders, optional)
        self.d = d
        self.fusion_lstm = nn.LSTM(d, d, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        H = self._encode(inputs, mask)                            # 缺失槽 latent 已置零
        s = torch.stack([H[k] for k in self.keys], dim=1)         # (B, S, d)
        z_seq, _ = self.fusion_lstm(s)                            # (B, S, d)
        avail = self._avail(mask)                                 # (B, S) bool
        B = s.shape[0]
        last_idx = avail.long().sum(dim=1) - 1                    # 末个可用 slot 下标
        last_idx = last_idx.clamp(min=0)
        z = z_seq[torch.arange(B), last_idx]                      # (B, d)
        return self.head(z).squeeze(-1)                           # (B,)


def _smoke() -> None:
    optional = ["ct", "ecg", "echo", "vitals", "cxr"]
    d = 128
    encoders = make_baseline_encoders(optional, d, 0.1)
    B, L = 8, 500
    inputs = {
        "structured": torch.randn(B, 6),
        "ct": torch.randn(B, 3),
        "ecg": torch.randn(B, 12, L),
        "echo": torch.randn(B, 3),
        "vitals": torch.randn(B, 6),
        "cxr": torch.randn(B, 3),
    }
    mask = torch.tensor([
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    m = MedFuseModel(encoders, optional, d=d, dropout=0.1)
    logits = m(inputs, mask)
    assert logits.shape == (B,) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print(f"[smoke] MedFuseModel OK  params={sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    _smoke()
