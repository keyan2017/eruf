"""ERUF —— 缺失模式感知的证据充分性融合模型。

Evidence-guided multimodal Representation with Uncertainty-aware Fusion（ERUF）。显式建模
缺失模式（缺失非随机 MNAR），与内容解耦，一个模型输出三个量：

    h_m    = E_m(x_m)                            # Step1 编码（全量，缺失槽内容为 0 填充）
    s      = [h_0 ; h_m·a_m]                     # Step2a 掩码拼接（内容，缺失槽置 0）
    δ_emb  = MaskEmb(a)                          # Step2b 缺失模式嵌入（MNAR：δ 是信息性输入）
    h      = Fusion([s ; δ_emb])                 # Step2c MNAR 感知融合
    p      = σ(f(h))                             # 输出1 疾病风险
    C      = σ(f_C([h ; a ; |S_t|]))             # 输出2 证据充分性（学习头）
    Δ_m    = C(S_t∪{m}) − C(S_t)   （do-add 派生） # 输出3 模态增量价值（派生，无 L_voi）
    logit  = f(h) + g(δ)   （推理时加性 rectifier） # 后处理：减去观测模式的直接效应

三个损失（见 training/train_eruf.py）：

    L = L_pred + λ1·L_evidence + λ2·L_recon
      L_pred     = BCE(p, y)                       （风险判别，full + 随机子集）
      L_evidence = BCE(C, y_evidence)              （y_evidence=1[预测正确]，stop-grad）
      L_recon    = Σ_m MSE(recon_m(h), h_m)        （可用模态重建，MNAR 表征稳定）

rectifier g(δ) 为后处理（stage2）：stage1 冻结后，用留出验证集残差 y−σ(f(h)) 拟合 g(δ)（MSE），
推理时 logit = f(h) + g(δ) 做加性偏置校正。

接口与基线一致：``forward(inputs, mask) -> 风险 logits``；``forward_outputs`` 返回
``(p_logit, C_logit, Δ (B,M))``；``forward_train`` 返回
``(p_full, C_full, [p_sub], [C_sub], recon_full, H_full)``。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import _Baseline, make_baseline_encoders


def _random_drop(mask: torch.Tensor, p: float) -> torch.Tensor:
    """随机置缺 p 比例的可用模态（0=缺失，1=可用；structured 不在 mask 内故恒可用）。"""
    keep = (torch.rand_like(mask) >= p).float()
    return mask * keep


class ERUFModel(_Baseline):
    """ERUF：掩码拼接 + 缺失模式嵌入 + 证据充分性头 + 后处理 rectifier。

    use_delta=False 时去掉缺失模式嵌入与 rectifier（消融 MNAR 感知），退化为「掩码拼接 +
    证据充分性头」。
    """

    def __init__(self, encoders: dict[str, nn.Module], optional: list[str],
                 d: int = 128, dropout: float = 0.2, use_delta: bool = True):
        super().__init__(encoders, optional)
        self.d = d
        self.n_opt = len(optional)
        self.use_delta = use_delta

        # 缺失模式嵌入 δ_emb = MaskEmb(a)
        self.mask_emb = nn.Sequential(nn.Linear(self.n_opt, d), nn.ReLU(), nn.Linear(d, d))
        # MNAR 感知融合：[掩码拼接 (S·d) ; δ_emb (d)] → h
        in_dim = self.n_slots * d + (d if use_delta else 0)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d),
        )
        # 输出1 疾病风险 p = σ(f(h))
        self.risk_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )
        # 输出2 证据充分性 C = σ(f_C([h ; a ; |S_t|]))  输入 d + M + 1 维
        self.evidence_head = nn.Sequential(
            nn.Linear(d + self.n_opt + 1, d), nn.ReLU(), nn.Linear(d, 1),
        )
        # 可用模态重建（MNAR 表征稳定）
        self.decoder = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) for k in self.keys
        })
        # 后处理 rectifier g(δ)（stage2 拟合，推理时加性偏置校正）
        self.rectifier = nn.Sequential(
            nn.Linear(self.n_opt, d // 2), nn.ReLU(), nn.Linear(d // 2, 1),
        )
        self.use_rectifier = False

    def _encode_full(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """编码一次（不按 mask 置零）；缺失性在 `_fuse` 内按 mask 处理。"""
        H = {"structured": self.encoders["structured"](inputs["structured"])}
        for j, k in enumerate(self.optional):
            H[k] = self.encoders[k](inputs[k])
        return H

    def _fuse(self, H: dict[str, torch.Tensor], mask: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (h, stats)：h = MNAR 感知融合；stats = (a, |S_t|/M) (B, M+1) 供 C 头。"""
        parts = [H["structured"]]
        for j, k in enumerate(self.optional):
            parts.append(H[k] * mask[:, j:j + 1])               # 缺失槽置 0
        s = torch.cat(parts, dim=-1)                            # (B, S·d)
        if self.use_delta:
            d_emb = self.mask_emb(mask)                         # (B, d)
            h = self.fusion(torch.cat([s, d_emb], dim=-1))
        else:
            h = self.fusion(s)
        count = mask.sum(dim=1, keepdim=True) / self.n_opt      # (B, 1) 归一化数量
        stats = torch.cat([mask, count], dim=1)                 # (B, M+1)
        return h, stats

    def _derived_voi(self, H: dict[str, torch.Tensor], mask: torch.Tensor
                     ) -> torch.Tensor:
        """模态增量价值（派生）：Δ_m = C(S∪{m}) − C(S)，仅对当前缺失样本求。返回 (B, M)。"""
        B = mask.shape[0]
        h, stats = self._fuse(H, mask)
        C_cur = torch.sigmoid(self.evidence_head(torch.cat([h, stats], dim=-1)).squeeze(-1))
        out = torch.zeros(B, self.n_opt, device=mask.device)
        for j in range(self.n_opt):
            absent = mask[:, j] <= 0.5
            if not absent.any():
                continue
            m_add = mask.clone()
            m_add[:, j] = 1.0
            h_add, stats_add = self._fuse(H, m_add)
            C_add = torch.sigmoid(
                self.evidence_head(torch.cat([h_add, stats_add], dim=-1)).squeeze(-1))
            out[:, j] = torch.where(absent, C_add - C_cur, out[:, j])
        return out

    def forward_outputs(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (p_logit, C_logit, Δ(B,M))。评估/诊断用。"""
        H = self._encode_full(inputs)
        h, stats = self._fuse(H, mask)
        p = self.risk_head(h).squeeze(-1)                       # (B,)
        C = self.evidence_head(torch.cat([h, stats], dim=-1)).squeeze(-1)  # (B,)
        D = self._derived_voi(H, mask)                          # (B, M)
        return p, C, D

    def forward(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """评估用单次前向（与基线同协议，返回风险 logits）。"""
        H = self._encode_full(inputs)
        h, _ = self._fuse(H, mask)
        logit = self.risk_head(h).squeeze(-1)
        if self.use_rectifier:
            logit = logit + self.rectifier(mask).squeeze(-1)
        return logit

    def forward_train(self, inputs: dict[str, torch.Tensor], mask: torch.Tensor,
                      n_views: int = 4, p_max: float = 0.75
                      ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor],
                                 list[torch.Tensor], dict[str, torch.Tensor],
                                 dict[str, torch.Tensor]]:
        """返回 (p_full, C_full, [p_sub], [C_sub], recon_full, H_full)。

        子集共享同一次编码；recon_full = {k: decoder[k](h_full)} 供 L_recon 重建可用模态。
        """
        H = self._encode_full(inputs)
        B = mask.shape[0]
        h_full, stats_full = self._fuse(H, mask)
        p_full = self.risk_head(h_full).squeeze(-1)
        C_full = self.evidence_head(torch.cat([h_full, stats_full], dim=-1)).squeeze(-1)
        recon_full = {k: self.decoder[k](h_full) for k in self.keys}
        p_subs: list[torch.Tensor] = []
        C_subs: list[torch.Tensor] = []
        for _ in range(n_views):
            p = float(torch.rand(1).item()) * p_max
            m_j = _random_drop(mask, p)
            h_j, stats_j = self._fuse(H, m_j)
            p_subs.append(self.risk_head(h_j).squeeze(-1))
            C_subs.append(self.evidence_head(torch.cat([h_j, stats_j], dim=-1)).squeeze(-1))
        return p_full, C_full, p_subs, C_subs, recon_full, H


def _smoke() -> None:
    optional = ["ecg", "echo", "vitals", "cxr"]
    d = 128
    encoders = make_baseline_encoders(optional, d, 0.1)
    B, L = 8, 500
    inputs = {
        "structured": torch.randn(B, 6),
        "ecg": torch.randn(B, 12, L),
        "echo": torch.randn(B, 3),
        "vitals": torch.randn(B, 6),
        "cxr": torch.randn(B, 3),
    }
    mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    for use_delta in (True, False):
        m = ERUFModel(encoders, optional, d=d, dropout=0.1, use_delta=use_delta)
        skip_fwd = ("rectifier", "decoder") if use_delta else ("rectifier", "decoder", "mask_emb")
        p_logit, C_logit, D = m.forward_outputs(inputs, mask)
        assert p_logit.shape == (B,) and C_logit.shape == (B,)
        assert D.shape == (B, len(optional))
        assert torch.isfinite(p_logit).all() and torch.isfinite(C_logit).all()
        assert torch.isfinite(D).all()
        (p_logit.sum() + C_logit.sum() + D.sum()).backward()
        assert all(p.grad is not None for n, p in m.named_parameters()
                   if p.requires_grad and not n.startswith(skip_fwd))

        # 训练路径 + 三损失
        m.zero_grad()
        out = m.forward_train(inputs, mask, n_views=3, p_max=0.75)
        p_full, C_full, p_subs, C_subs, recon_full, H = out
        assert p_full.shape == (B,) and C_full.shape == (B,)
        assert all(s.shape == (B,) for s in p_subs) and all(s.shape == (B,) for s in C_subs)
        assert set(recon_full.keys()) == set(m.keys)
        y = torch.randint(0, 2, (B,), dtype=torch.float32)
        crit = F.binary_cross_entropy_with_logits
        loss = crit(p_full, y)                                     # L_pred
        for ps in p_subs:
            loss = loss + 0.5 * crit(ps, y)
        y_ev = (torch.sigmoid(p_full.detach()) > 0.5).float()      # L_evidence
        loss = loss + 0.5 * crit(C_full, (y_ev == y).float())
        for ps, Cs in zip(p_subs, C_subs):
            y_ev_s = (torch.sigmoid(ps.detach()) > 0.5).float()
            loss = loss + 0.5 * crit(Cs, (y_ev_s == y).float())
        for k in m.keys:                                           # L_recon
            loss = loss + 0.1 * F.mse_loss(recon_full[k], H[k])
        assert torch.isfinite(loss)
        loss.backward()
        skip_tr = ("rectifier",) if use_delta else ("rectifier", "mask_emb")
        assert all(p.grad is not None for n, p in m.named_parameters()
                   if p.requires_grad and not n.startswith(skip_tr))

        # rectifier 单独可微
        m.zero_grad()
        m.rectifier(mask).sum().backward()
        assert all(p.grad is not None for p in m.rectifier.parameters() if p.requires_grad)
        print(f"[smoke] ERUFModel(use_delta={use_delta}) OK  "
              f"params={sum(p.numel() for p in m.parameters()):,}  loss={loss.item():.4f}")


if __name__ == "__main__":
    _smoke()
