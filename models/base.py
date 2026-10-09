"""共享基座：各模型共用的编码器构造、masked 注意力与缺失感知骨架。

所有模型（ERUF 与 5 个基线）共用同一 encoder backbone（``make_baseline_encoders``）与
缺失感知公共骨架（``_Baseline``），保证「同数据同 backbone」的公平对比。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import MLPEncoder
from models.image_encoders import MimicECGWaveformEncoder


def make_baseline_encoders(optional: list[str], d: int, dropout: float) -> dict[str, nn.Module]:
    """基线编码器构造（同 backbone 保证公平）。"""
    encoders: dict[str, nn.Module] = {
        "structured": MLPEncoder(6, (64, 128), d, dropout),
        "ecg": MimicECGWaveformEncoder(latent_dim=d, dropout=dropout),
        "echo": MLPEncoder(3, (32, 64), d, dropout),
        "vitals": MLPEncoder(6, (32, 64), d, dropout),
        "cxr": MLPEncoder(3, (32, 64), d, dropout),
    }
    if "ct" in optional:
        encoders["ct"] = MLPEncoder(3, (32, 64), d, dropout)
    return encoders


class MaskedModalityAttention(nn.Module):
    """可学习 patient query 的 masked 多头注意力（共享可学习查询，缺失槽 -inf）。

    ``Q`` 为**共享可学习查询**（泛化的「患者状态探针」，非 patient_id 输入——patient_id
    严禁作为模型输入）。缺失模态的注意力 logits 置 -inf 使其不参与融合。
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(d_model) * 0.02)   # 共享患者查询
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        """tokens: (B, S, d)；avail: (B, S) bool（True=可 attend，structured 恒 True）。

        返回 (B, d)。
        """
        B, S, _ = tokens.shape
        q = self.w_q(self.query).view(1, 1, self.n_heads, self.head_dim).expand(B, -1, -1, -1)
        k = self.w_k(tokens).view(B, S, self.n_heads, self.head_dim)   # (B, S, H, hd)
        v = self.w_v(tokens).view(B, S, self.n_heads, self.head_dim)

        scores = torch.einsum("bqhd,bshd->bhqs", q, k) * self.scale    # (B, H, 1, S)
        scores = scores.masked_fill(~avail.view(B, 1, 1, S), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)                            # (B, H, 1, S)
        attn = self.dropout(attn)
        out = torch.einsum("bhqs,bshd->bqhd", attn, v)                  # (B, 1, H, hd)
        out = out.reshape(B, 1, self.d_model)
        return self.out(out).squeeze(1)                                 # (B, d)


class _Baseline(nn.Module):
    """模型公共骨架：编码 + 缺失 latent 置零 + optional/keys 接口。"""

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str]):
        super().__init__()
        self.optional = list(optional)
        self.keys = ["structured", *self.optional]
        self.n_slots = len(self.keys)
        self.encoders = nn.ModuleDict(encoders)

    def _encode(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """各模态编码；缺失可选模态 latent 置零（structured 恒可用不置零）。"""
        H = {"structured": self.encoders["structured"](inputs["structured"])}
        for j, k in enumerate(self.optional):
            h = self.encoders[k](inputs[k])
            H[k] = h * mask[:, j:j + 1]
        return H

    def _avail(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, S) bool：structured 恒 True，可选模态 = mask>0.5。"""
        B = mask.shape[0]
        avail = torch.ones(B, self.n_slots, dtype=torch.bool, device=mask.device)
        avail[:, 1:] = mask > 0.5
        return avail
