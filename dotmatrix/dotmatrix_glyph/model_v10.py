# -*- coding: utf-8 -*-
"""
NeuroGlyph34V10 — 4300万参数 | 字形+语义双通道 Transformer
=========================================================
V10 架构：
  - 输入：34维字形嵌入 (预计算, frozen) + 64维语义嵌入 (可学习)
  - 融合：concat → 98维 → Linear → 512维
  - 主干：12层 Transformer, 8头, d_ff=2048
  - 输出头：vocab_size (7987) 下一字符预测
  - 总参数量：约 4275 万
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuroGlyph34V10(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7987,
        d_glyph: int = 34,
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

        # 可学习的语义嵌入
        self.semantic_embed = nn.Embedding(vocab_size, d_semantic)

        # 字形嵌入矩阵 (注册为 buffer, 不参与梯度)
        self.register_buffer("glyph_matrix", torch.zeros(vocab_size, d_glyph))

        # 输入投影: 34+64=98 → 512
        self.input_proj = nn.Sequential(
            nn.Linear(d_glyph + d_semantic, d_model),
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
        """加载预训练的字形嵌入矩阵 (frozen)"""
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

        # 字形嵌入 (frozen, 从预加载矩阵查表)
        glyph_feat = self.glyph_matrix[token_ids[:, :T_clamped]]  # (B, T, 34)

        # 语义嵌入 (可学习)
        sem_feat = self.semantic_embed(token_ids[:, :T_clamped])  # (B, T, 64)

        # 融合
        combined = torch.cat([glyph_feat, sem_feat], dim=-1)  # (B, T, 98)
        hidden = self.input_proj(combined)  # (B, T, 512)

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
