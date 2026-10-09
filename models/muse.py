"""MUSE 基线（互一致对比学习）—— 核心机制复现。

在特征化模态（无原始影像）上按方法描述复现核心机制（非逐行复现）：编码 + 可用模态融合 +
投影，互一致对比损失由训练脚本施加（见 training/train_baselines.py）。原文用「患者-模态
二部图 + 边丢弃 + 互一致对比」；本数据无图结构，故边丢弃降为随机模态丢弃，无监督一致
降为同一患者两个丢弃视角表征的 InfoNCE 对齐，有监督一致为同标签患者拉近（batch 内），
如实标注该简化。

接口：``forward(inputs, mask) -> logits``；``model.optional / model.keys`` 暴露。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import _Baseline, make_baseline_encoders


class MUSEModel(_Baseline):
    """MUSE 核心机制复现：编码 + 可用模态融合 + 投影；互一致对比损失由训练脚本施加。

    原文用「患者-模态二部图 + 边丢弃 + 互一致对比（无监督跨模态一致 / 有监督同标签
    相近）」。本数据无图结构，故：
      * 边丢弃 → 随机模态丢弃（训练脚本生成两视角）；
      * 无监督一致 → 同一患者两个丢弃视角的表征 InfoNCE 对齐；
      * 有监督一致 → 同标签患者表征拉近、异标签推远（batch 内）。
    如实标注该简化。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2):
        super().__init__(encoders, optional)
        self.d = d
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )

    def forward_z(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        H = self._encode(inputs, mask)                            # 缺失槽 latent 已置零
        s = torch.stack([H[k] for k in self.keys], dim=1)         # (B, S, d)
        a = self._avail(mask).unsqueeze(-1).float()               # (B, S, 1)
        z = (s * a).sum(dim=1) / a.sum(dim=1).clamp(min=1.0)      # (B, d)
        z = self.proj(z)
        return self.head(z).squeeze(-1), z

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        return self.forward_z(inputs, mask)[0]


def muse_mutual_consistent_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """MUSE 互一致对比损失（batch 内）。

    有监督：同标签患者为正、异标签为负（label-decisive）。
    返回 (L_sup, L_unsup)；L_unsup 需要两个视角，由训练脚本单独对齐（见 train_baselines）。
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau                                       # (B, B)
    pos = (y[:, None] == y[None, :]).float()                    # 同标签
    eye = torch.eye(y.shape[0], device=y.device)
    pos = pos * (1.0 - eye)
    neg_mask = 1.0 - (y[:, None] == y[None, :]).float() - eye
    log_sum_pos = torch.logsumexp(sim + torch.log(pos.clamp(min=1e-9)), dim=1)
    log_sum_all = torch.logsumexp(sim + torch.log((pos + neg_mask).clamp(min=1e-9)), dim=1)
    L_sup = -(log_sum_pos - log_sum_all).mean()
    return L_sup, None


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

    m = MUSEModel(encoders, optional, d=d, dropout=0.1)
    logits, z = m.forward_z(inputs, mask)
    assert logits.shape == (B,) and z.shape == (B, d) and torch.isfinite(logits).all()
    y = torch.randint(0, 2, (B,), dtype=torch.float32)
    lsup, _ = muse_mutual_consistent_loss(z, y)
    assert torch.isfinite(lsup)
    (logits.sum() + lsup).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print(f"[smoke] MUSEModel OK  params={sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    _smoke()
