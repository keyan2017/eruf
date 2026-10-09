"""DrFuse 基线（解耦共享/特有表征 + 注意力融合）—— 核心机制复现。

在特征化模态（无原始影像）上按方法描述复现核心机制（非逐行复现）：每模态 latent 拆
shared/distinct 两部分，shared 经 masked attention 融合、distinct 经可用槽均值融合。
原方法的「疾病感知注意力排序损失」简化为可学习注意力（ranking 损失未落地，如实标注）。

接口：``forward(inputs, mask) -> logits``；``model.optional / model.keys`` 暴露。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.base import MaskedModalityAttention, _Baseline, make_baseline_encoders


class DrFuseModel(_Baseline):
    """DrFuse 核心机制复现：shared/distinct 解耦 + 共享部分注意力融合。

    原方法额外有「疾病感知注意力排序损失」（用标签构造模态重要性排序），此处简化为
    可学习 masked attention（排序损失未落地，代码与论文均如实标注该简化）。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2, n_heads: int = 4):
        super().__init__(encoders, optional)
        self.d = d
        self.shared_dim = d // 2
        self.distinct_dim = d - self.shared_dim
        self.shared_proj = nn.Linear(d, self.shared_dim)
        self.distinct_proj = nn.Linear(d, self.distinct_dim)
        self.shared_attn = MaskedModalityAttention(self.shared_dim, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        H = self._encode(inputs, mask)
        s = torch.stack([H[k] for k in self.keys], dim=1)          # (B, S, d)
        shared = self.shared_proj(s)                               # (B, S, d/2)
        distinct = self.distinct_proj(s)                           # (B, S, d/2)

        avail = self._avail(mask)                                  # (B, S) bool
        fused_shared = self.shared_attn(shared, avail)             # (B, d/2)

        a = avail.unsqueeze(-1).float()
        fused_distinct = (distinct * a).sum(dim=1) / a.sum(dim=1).clamp(min=1.0)  # (B, d/2)

        z = torch.cat([fused_shared, fused_distinct], dim=-1)      # (B, d)
        return self.head(z).squeeze(-1)                            # (B,)


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

    m = DrFuseModel(encoders, optional, d=d, dropout=0.1)
    logits = m(inputs, mask)
    assert logits.shape == (B,) and torch.isfinite(logits).all()
    assert m.optional == optional and m.keys == ["structured", *optional]
    logits.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print(f"[smoke] DrFuseModel OK  params={sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    _smoke()
