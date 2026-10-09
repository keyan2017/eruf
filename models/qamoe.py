"""QA-MoE 基线（质量感知 + 稳定的专家混合路由）—— 核心机制复现。

在特征化模态（无原始影像）上按方法描述复现核心机制（非逐行复现）：把「模态可靠性估计」
与「专家路由」解耦，使噪声/缺失模态无法扰动专家路由（见类 docstring）。图像/生存相关
方法不做对比。

接口：``forward(inputs, mask) -> logits``；``model.optional / model.keys`` 暴露。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import _Baseline, make_baseline_encoders


class QAMoEModel(_Baseline):
    """QA-MoE (Linpeng Sun & Victor S. Sheng, IJCAI 2026) 核心机制复现。

    核心 = 把「模态可靠性估计」与「专家路由」解耦，使噪声/缺失模态无法扰动专家路由：

      1. Evidential Quality Scorer —— 每模态 latent 经 Dirichlet 证据头得 K 类证据 e，
         可靠性 r = S/(S+K)（S=Σ(e+1)）。可靠性只由该模态内容决定、不含路由信号；
         缺失模态 r 置 0。
      2. Stability-Enhanced Subset Selector —— 对「可用模态的可靠性」做 masked softmax
         得门控 g，动态丢弃低可靠模态。
      3. 专家聚合 —— 每模态一个专家 MLP 出标量贡献，g 加权求和；另有 global prior 专家
         对可用 latent 的 mean-pool 出标量，权重随「平均可靠性下降」增大，作为严重缺失
         时的 fail-safe（稳定性来源）。

    loss = BCE(最终 logit) + λ_edl·EDL 正则（门控聚合的全局证据对 ground-truth 的
    Dirichlet Type-II ML 损失）。

    如实标注的简化：原方法的「三元专家聚合」分支内部结构未公开，此处以「global prior
    专家 + 可靠性门控 fail-safe 权重」近似；evidential scorer 取 K=2（二分类）证据。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2, K: int = 2):
        super().__init__(encoders, optional)
        self.d = d
        self.K = K
        # 每模态 Dirichlet 证据头（质量打分器），structured 恒可用也参与
        self.evidence_head = nn.ModuleDict({k: nn.Linear(d, K) for k in self.keys})
        # 每模态专家 → 标量贡献
        self.experts = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))
            for k in self.keys
        })
        # global prior 专家（严重缺失 fail-safe）
        self.global_expert = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )
        self.failsafe_w = nn.Parameter(torch.tensor(1.0))

    def _reliability(self, H: dict[str, torch.Tensor], mask: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """每模态 Dirichlet 可靠性 r ∈ [0.5,1]，缺失置 0。返回 (r, alpha, S)。"""
        evid_list, alpha_list, S_list = [], [], []
        for k in self.keys:
            e = self.evidence_head[k](H[k])          # (B, K)
            evid = F.softplus(e)                     # 证据 > 0
            alpha = evid + 1.0
            S = alpha.sum(dim=-1, keepdim=True)      # (B, 1)
            evid_list.append(evid)
            alpha_list.append(alpha)
            S_list.append(S)
        alpha = torch.stack(alpha_list, dim=1)       # (B, S, K)
        S = torch.stack(S_list, dim=1)               # (B, S, 1)
        r = (S / (S + self.K)).squeeze(-1)           # (B, S) ∈ [0.5, 1]
        r = r * self._avail(mask).float()            # 缺失 → 0
        return r, alpha, S

    def forward_aux(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (logits, g, alpha, S)，供 BCE + EDL 正则。"""
        H = self._encode(inputs, mask)               # 缺失 latent 置零
        r, alpha, S = self._reliability(H, mask)
        avail = self._avail(mask)                    # (B, S) bool

        # 子集选择器：可用模态可靠性 softmax（缺失 → -inf → g=0）
        g_logits = r.masked_fill(~avail, float("-inf"))
        g = F.softmax(g_logits, dim=-1)              # (B, S)

        expert_out = torch.stack(
            [self.experts[k](H[k]).squeeze(-1) for k in self.keys], dim=1)  # (B, S)
        logits_sel = (g * expert_out).sum(dim=1)     # (B,)

        # fail-safe：可用 latent mean-pool，权重随平均可靠性下降而增大
        s = torch.stack([H[k] for k in self.keys], dim=1)   # (B, S, d)
        a = avail.unsqueeze(-1).float()
        mean_pool = (s * a).sum(dim=1) / a.sum(dim=1).clamp(min=1.0)          # (B, d)
        g_out = self.global_expert(mean_pool).squeeze(-1)                     # (B,)
        mean_r = r.sum(dim=1) / a.sum(dim=1).clamp(min=1.0).squeeze(-1)       # (B,)
        fail_w = self.failsafe_w * (1.0 - mean_r).clamp(min=0.0)      # (B,)
        logits = logits_sel + fail_w * g_out
        return logits, g, alpha, S

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        return self.forward_aux(inputs, mask)[0]


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

    m = QAMoEModel(encoders, optional, d=d, dropout=0.1)
    logits, g, alpha, S = m.forward_aux(inputs, mask)
    assert logits.shape == (B,), logits.shape
    assert g.shape == (B, m.n_slots) and alpha.shape == (B, m.n_slots, m.K)
    assert torch.isfinite(logits).all()
    # 缺失槽门控为 0
    assert (g[:, 1] <= 1e-6).all()  # ct 恒缺失
    assert m.optional == optional and m.keys == ["structured", *optional]
    logits.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print(f"[smoke] QAMoEModel OK  params={sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    _smoke()
