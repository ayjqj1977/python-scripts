# -*- coding: utf-8 -*-
"""
model_pointnet.py — 点阵字形编码 Transformer (~42.3M 参数)
=========================================================
基于 model_v10.py 改造：
  - 将冻死的 34 维 glyph_embedding 替换为可训练的点阵编码器
  - PointMatrixEncoder: Linear(1024→256) → LayerNorm → GELU → Linear(256→64)
  - 字形向量(64) + 语义向量(64) = 128 → Linear → 512
  - vocab_size = 6622 (裁剪词表,仅保留 GB2312 可读字)
  - glyph_matrix 为 register_buffer,存储原始 32×32 二值点阵(1024 维)
  - 主干: 12 层 Transformer, 8 头, d_ff=2048

用法:
    model = PointMatrixModel(vocab_size=6622)
    model.load_glyph_matrix(numpy_array)  # (6622, 1024) 原始点阵
    logits = model(token_ids)
"""
import numpy as np
import torch
import torch.nn as nn


class PointMatrixEncoder(nn.Module):
    """可训练的点阵编码器: 1024 维原始点阵 → 64 维字形嵌入"""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, 1024) 原始二值点阵向量 (来自 register_buffer,不含梯度)
        Returns:
            (*, 64) 可训练的字形嵌入
        """
        return self.encoder(x)


class PointMatrixModel(nn.Module):
    """点阵字形编码 Transformer 模型"""

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

        # 原始点阵矩阵 (register_buffer: 冻结,不参与梯度,仅用作编码器输入)
        self.register_buffer("glyph_matrix", torch.zeros(vocab_size, d_glyph))

        # 可训练的点阵编码器
        self.point_encoder = PointMatrixEncoder(
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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_glyph_matrix(self, embedding_matrix: np.ndarray):
        """
        加载预提取的原始点阵矩阵 (frozen buffer)。
        Args:
            embedding_matrix: (vocab_size, 1024) 二值点阵,来自 HZK32Reader
        """
        self.glyph_matrix.copy_(torch.from_numpy(embedding_matrix).float())

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) 字符索引
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = token_ids.shape
        T_clamped = min(T, self.max_seq_len)

        # 原始点阵 (frozen buffer) → 可训练编码器 → 64 维字形向量
        glyph_raw = self.glyph_matrix[token_ids[:, :T_clamped]]  # (B, T, 1024)
        glyph_feat = self.point_encoder(glyph_raw)                # (B, T, 64)

        # 语义嵌入 (可学习)
        sem_feat = self.semantic_embed(token_ids[:, :T_clamped])  # (B, T, 64)

        # 融合 → 投影
        combined = torch.cat([glyph_feat, sem_feat], dim=-1)     # (B, T, 128)
        hidden = self.input_proj(combined)                        # (B, T, 512)

        # 位置编码
        hidden = hidden + self.pos_encoding[:, :T_clamped, :]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T_clamped, T_clamped, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        # Transformer
        hidden = self.transformer(hidden, mask=causal_mask)
        hidden = self.ln_final(hidden)

        return self.lm_head(hidden)

    def param_count(self) -> tuple:
        """返回 (总参数, 可训练参数)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    model = PointMatrixModel(vocab_size=6726)
    total, trainable = model.param_count()
    print(f"PointMatrixModel: total={total:,} trainable={trainable:,}")

    # 模拟加载点阵: 前 6623 有 HZK32 点阵, 后 103 补回字符为零向量
    dummy_glyph = np.random.randint(0, 2, (6726, 1024)).astype(np.float32)
    dummy_glyph[6623:] = 0.0  # 补回字符无点阵
    model.load_glyph_matrix(dummy_glyph)

    # 测试前向
    batch = torch.randint(0, 6726, (4, 64))
    logits = model(batch)
    print(f"Input: {batch.shape} → Output: {logits.shape}")
    print(f"vocab_size={model.vocab_size}, d_model={model.d_model}")
