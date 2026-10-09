"""ModDrop-Contrastive 基线（可学习缺失 token + 模态丢弃 + 对比对齐）—— 核心机制复现。

在特征化模态（无原始影像）上按方法描述复现核心机制（非逐行复现）：缺失槽用可学习
per-modality missing token 替换（非零）后融合；训练时同步模态丢弃 + 融合表征与各可用
单模态表征的 InfoNCE 对比（见 training/train_baselines.py）。

接口：``forward(inputs, mask) -> logits``；``model.optional / model.keys`` 暴露。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.base import MaskedModalityAttention, _Baseline, make_baseline_encoders


class ModDropContrastiveModel(_Baseline):
    """ModDrop-Contrastive 核心机制复现：缺失槽用可学习 token 替换 + 对比对齐。

    缺失模态的 latent 不置零，而是替换为该模态的可学习 missing token（区别于本方法的
    -inf 屏蔽）；融合对所有槽位注意力。训练脚本施加同步模态丢弃，并用 InfoNCE 对齐
    融合表征与各可用单模态表征（见 training/train_baselines.py）。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2, n_heads: int = 4):
        super().__init__(encoders, optional)
        self.d = d
        self.miss_token = nn.Parameter(torch.randn(len(optional), d) * 0.02)  # (M, d)
        self.mod_type_emb = nn.Parameter(torch.randn(self.n_slots, d) * 0.02)
        self.fusion = MaskedModalityAttention(d, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )

    def _tokens(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """组装 token：可用槽 = 内容 + 类型嵌入；缺失槽 = 可学习 missing token + 类型嵌入。"""
        toks = [self.encoders["structured"](inputs["structured"]) + self.mod_type_emb[0]]
        for j, k in enumerate(self.optional):
            h = self.encoders[k](inputs[k])
            present = (mask[:, j] > 0.5).float().unsqueeze(-1)   # (B, 1)
            tok = h * present + self.miss_token[j] * (1.0 - present)
            toks.append(tok + self.mod_type_emb[1 + j])
        return torch.stack(toks, dim=1)                          # (B, S, d)

    def forward_z(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        toks = self._tokens(inputs, mask)
        B = mask.shape[0]
        avail = torch.ones(B, self.n_slots, dtype=torch.bool, device=mask.device)
        z = self.fusion(toks, avail)                             # (B, d)
        return self.head(z).squeeze(-1), z

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        return self.forward_z(inputs, mask)[0]

    def unimodal(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """各模态内容 latent（缺失置零），供融合-单模态 InfoNCE 对齐。"""
        return self._encode(inputs, mask)

    def forward_loss(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """单次前向返回 (logits, z, H)，训练时避免二次编码。H 为内容 latent（缺失置零）。"""
        H = self._encode(inputs, mask)
        toks = [H["structured"] + self.mod_type_emb[0]]
        for j, k in enumerate(self.optional):
            present = (mask[:, j] > 0.5).float().unsqueeze(-1)
            tok = H[k] + self.miss_token[j] * (1.0 - present)      # H[k] 缺失已置零
            toks.append(tok + self.mod_type_emb[1 + j])
        toks = torch.stack(toks, dim=1)
        B = mask.shape[0]
        avail = torch.ones(B, self.n_slots, dtype=torch.bool, device=mask.device)
        z = self.fusion(toks, avail)
        return self.head(z).squeeze(-1), z, H


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

    m = ModDropContrastiveModel(encoders, optional, d=d, dropout=0.1)
    logits = m(inputs, mask)
    assert logits.shape == (B,) and torch.isfinite(logits).all()
    assert m.optional == optional and m.keys == ["structured", *optional]
    logits.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    _, z = m.forward_z(inputs, mask)
    assert z.shape == (B, d)
    h = m.unimodal(inputs, mask)
    assert set(h.keys()) == set(["structured", *optional])
    print(f"[smoke] ModDropContrastiveModel OK  params={sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    _smoke()
