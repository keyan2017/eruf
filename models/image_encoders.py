"""波形编码器（ECG 12 导联，250Hz 原始波形 → latent）。

两种 ECG 编码器，输出统一 latent_dim（默认 128）：

  MimicECGWaveformEncoder —— 基线用的浅层 4-block CNN + 全局平均池化。
  ResECGWaveformEncoder  —— ERUF 用的 ResNet-1D + SE（更深，残差 + 通道注意力）。

输入形状：ECG (B, 12, L)（L≈2500）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MimicECGWaveformEncoder(nn.Module):
    """1D CNN：MIMIC 原始 12 导联波形 (B, 12, L) -> latent（250Hz，L≈2500）。

    比浅层 CNN 更深（4 个 conv block），直接吃抽稀后的原始波形，
    学习 PH 相关的心电形态特征（右偏轴 / R-S 比 / 右室肥厚 / P 肺型等）。
    输入可为 float16（内部转 float32 以适配 Conv1d）。
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 12, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, 1, 7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, 1, 4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, 1, 3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, 1, 2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.head(self.features(x).flatten(1))


class _SEResBlock1D(nn.Module):
    """1D 残差块 + SE 通道注意力：conv-bn-relu-conv-bn → SE 重标定 → shortcut 相加。

    支持 stride 下采样（stride 落在第一层卷积）；SE 用全局平均池化 + 两层 1×1 conv 学习
    通道重要性（reduction=16），参数几乎为零，但对 ECG 各导联形态的通道重标定通常 +0.5~1% AUROC。
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1, kernel: int = 7,
                 reduction: int = 16):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_c, max(out_c // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(out_c // reduction, 4), out_c, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm1d(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)          # 通道重标定
        out = out + self.shortcut(x)
        return self.relu(out)


class ResECGWaveformEncoder(nn.Module):
    """ResNet-1D + SE 波形编码器：比 MimicECGWaveformEncoder（浅层 4-block CNN + 全局平均池化，
    无残差/注意力）更深（4 阶段 × 2 SE 残差块，64→128→128 通道），残差连接缓解深层退化、
    SE 通道注意力重标定各导联形态、浅层大核（k=15/7）捕捉 QRS/右室肥厚等宽形态、深层小核（k=5）
    捕捉细粒度变化，最后自适应池化（长度无关）。

    输入 (B, 12, L)（250Hz，L≈2500），输出 latent_dim。长度无关（AdaptiveAvgPool1d）。
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 12, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, 2, 7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._make_layer(64, 64, 2, 1, 7)
        self.layer2 = self._make_layer(64, 128, 2, 2, 7)
        self.layer3 = self._make_layer(128, 128, 2, 2, 5)
        self.layer4 = self._make_layer(128, 128, 2, 1, 5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, latent_dim))

    @staticmethod
    def _make_layer(in_c: int, out_c: int, blocks: int, stride: int, kernel: int) -> nn.Sequential:
        layers = [_SEResBlock1D(in_c, out_c, stride, kernel)]
        for _ in range(1, blocks):
            layers.append(_SEResBlock1D(out_c, out_c, 1, kernel))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))
