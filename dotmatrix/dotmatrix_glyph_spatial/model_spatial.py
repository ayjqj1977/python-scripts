# -*- coding: utf-8 -*-
"""
model_spatial.py — 点阵字形编码 Transformer (带空间感知编码器)
=================================================================
基于 model_pointnet.py 改造：
  - 将 PointMatrixEncoder (两层 Linear) 替换为 PointMatrixSpatialEncoder
  - 新增 Conv2d 层捕捉 32×32 网格中的笔画连续性、邻域结构
  - 1024位 → reshape(1,32,32) → Conv2d(8, k3) → Conv2d(4, k3) → Linear(256) → Linear(64)
  - 参数: ~1.07M (原版 279K, +790K)
  - 其余架构同 PointMatrixModel

用法:
    model = PointMatrixSpatialModel(vocab_size=6622)
    model.load_glyph_matrix(numpy_array)   # (6622, 1024) 原始点阵
    logits = model(token_ids)
"""
import numpy as np
import torch
import torch.nn as nn


class PointMatrixSpatialEncoder(nn.Module):
    """带空间感知的点阵编码器: 1024 维原始点阵 → 64 维字形嵌入

    原版 PointMatrixEncoder: 1024 → Linear(256) → GELU → Linear(64)
    本版加入:
      - reshape 1024 → (1, 32, 32) 恢复 2D 网格结构
      - 2 层 Conv2d (k=3, pad=1) 捕捉 8 邻域空间关系
      - 3×3 卷积核天然识别笔画转角、交叉、连续性
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        # 空间特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # → (B, 8, 32, 32)
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),   # → (B, 4, 32, 32)
            nn.GELU(),
        )
        # 展平 + 投影
        self.proj = nn.Sequential(
            nn.Linear(4 * 32 * 32, hidden_dim),          # 4096 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),            # 256 → 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, 1024) 原始二值点阵向量 (来自 register_buffer)
        Returns:
            (*, 64) 可训练的字形嵌入
        """
        batch_shape = x.shape[:-1]  # (*)
        x = x.reshape(-1, 1, 32, 32)   # (N, 1, 32, 32)
        x = self.conv(x)                # (N, 4, 32, 32)
        x = x.reshape(x.shape[0], -1)   # (N, 4096)
        x = self.proj(x)                # (N, 64)
        return x.reshape(*batch_shape, 64)


class PointMatrixSpatialModel(nn.Module):
    """点阵字形编码 Transformer 模型 (空间感知版)"""

    def __init__(
        self,
        vocab_size: int = 6726,
        d_glyph: int = 1024,
        d_semantic: int = 64,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 语义嵌入 (可学习)
        self.semantic_embed = nn.Embedding(vocab_size, d_semantic)

        # 原始点阵矩阵 (frozen buffer)
        self.register_buffer("glyph_matrix", torch.zeros(vocab_size, d_glyph))

        # 可训练的空间感知点阵编码器
        self.point_encoder = PointMatrixSpatialEncoder(
            input_dim=d_glyph, hidden_dim=256, output_dim=64
        )

        # 输入投影: glyph(64) + semantic(64) = 128 → 512
        self.input_proj = nn.Sequential(
            nn.Linear(64 + d_semantic, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 可学习的位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 最终层归一化 + 输出头
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_glyph_matrix(self, embedding_matrix: np.ndarray):
        """加载 HZK32 原始点阵矩阵 (frozen buffer)"""
        self.glyph_matrix.copy_(torch.from_numpy(embedding_matrix).float())

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        B, T = token_ids.shape
        T_clamped = min(T, self.max_seq_len)

        # 点阵 → 空间感知编码器 → 64 维
        # 优化: Conv2d 对全体 6726 字批量执行一次，再按 token 索引，消除逐步 reshape/kernel 启动开销
        glyph_all = self.glyph_matrix.reshape(self.vocab_size, 1, 32, 32)  # (V, 1, 32, 32)
        glyph_conv = self.point_encoder.conv(glyph_all)                     # (V, 4, 32, 32)
        glyph_conv = glyph_conv.reshape(self.vocab_size, -1)                # (V, 4096)
        glyph_feat = self.point_encoder.proj(glyph_conv[token_ids[:, :T_clamped]])  # (B, T, 64)

        # 语义嵌入
        sem_feat = self.semantic_embed(token_ids[:, :T_clamped])

        # 融合 → 投影
        combined = torch.cat([glyph_feat, sem_feat], dim=-1)
        hidden = self.input_proj(combined)

        # 位置编码
        hidden = hidden + self.pos_encoding[:, :T_clamped, :]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T_clamped, T_clamped, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.ln_final(hidden)

        return self.lm_head(hidden)

    def param_count(self) -> tuple:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    model = PointMatrixSpatialModel(vocab_size=6726)
    total, trainable = model.param_count()
    print(f"PointMatrixSpatialModel: total={total:,} trainable={trainable:,}")

    # 模拟点阵
    dummy_glyph = np.random.randint(0, 2, (6726, 1024)).astype(np.float32)
    dummy_glyph[6623:] = 0.0
    model.load_glyph_matrix(dummy_glyph)

    batch = torch.randint(0, 6726, (4, 64))
    logits = model(batch)
    print(f"Input: {batch.shape} → Output: {logits.shape}")
    print(f"vocab_size={model.vocab_size}, d_model={model.d_model}")
