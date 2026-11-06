#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NanoText-18: 轻量级文本生成模型
基于18维向量嵌入层，专为16GB内存笔记本优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class NanoConfig:
    """模型配置"""
    vocab_size: int = 6170      # 词汇量
    n_embd: int = 18           # 嵌入维度 (必须为18!)
    n_head: int = 3             # 注意力头数 (18/3=6)
    n_layer: int = 6            # Transformer层数
    block_size: int = 128       # 上下文长度
    dropout: float = 0.1        # Dropout率
    bias: bool = False           # 是否使用bias


class LearnablePositionalEmbedding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.pe = nn.Embedding(config.block_size, config.n_embd)
    
    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        return self.pe(positions)


class CausalSelfAttention(nn.Module):
    """因果自注意力"""
    def __init__(self, config: NanoConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        
        # QKV投影
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # QKV投影
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 分割多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 注意力计算 (使用scaled_dot_product_attention优化)
        # 注意: is_causal=True 时不能同时使用 attn_mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, config: NanoConfig):
        super().__init__()
        hidden_dim = config.n_embd * 4  # 4倍扩展
        hidden_dim = (hidden_dim // 16) * 16  # 对齐
        
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class NanoBlock(nn.Module):
    """单个Transformer块 (Pre-LayerNorm)"""
    def __init__(self, config: NanoConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        # Pre-LayerNorm: 先归一化，再计算
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """NanoGPT-18 主模型"""
    def __init__(self, config: NanoConfig, embedding_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.config = config
        
        # 嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 位置编码
        self.wpe = LearnablePositionalEmbedding(config)
        
        # Transformer块
        self.blocks = nn.ModuleList([NanoBlock(config) for _ in range(config.n_layer)])
        
        # 输出归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # 输出头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 加载预训练嵌入权重
        if embedding_weights is not None:
            self.wte.weight.data = embedding_weights.clone()
            # 绑定输入输出嵌入 (可选)
            # self.lm_head.weight = self.wte.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 输出层使用较小的初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"
        
        # 嵌入 + 位置编码
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(idx)
        x = tok_emb + pos_emb
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 输出归一化
        x = self.ln_f(x)
        
        # 投影到词汇表
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 截断上下文
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 追加
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 遇到EOS停止 (这里简单处理)
            if idx_next.item() == 3:  # <EOS>
                break
        
        return idx
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_model(
    vocab_size: int = 6170,
    n_embd: int = 18,
    n_head: int = 3,
    n_layer: int = 6,
    block_size: int = 128,
    embedding_weights: Optional[torch.Tensor] = None
) -> NanoGPT:
    """创建模型的便捷函数"""
    config = NanoConfig(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    )
    
    model = NanoGPT(config, embedding_weights)
    return model


# ============== 使用示例 ==============
if __name__ == "__main__":
    print("=" * 50)
    print("NanoText-18 轻量级文本生成模型")
    print("=" * 50)
    
    # 创建模型
    model = create_model(
        vocab_size=6170,
        n_embd=18,
        n_head=3,
        n_layer=6,
        block_size=128
    )
    
    # 统计参数
    num_params = model.get_num_params()
    print(f"\n模型参数数量: {num_params:,}")
    print(f"模型大小: {num_params * 4 / 1024:.2f} KB (FP32)")
    
    # 测试前向传播
    print("\n测试前向传播...")
    idx = torch.randint(0, 6170, (2, 10))  # batch=2, seq=10
    logits, loss = model(idx, idx)  # 自回归训练
    print(f"输入形状: {idx.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试生成
    print("\n测试文本生成...")
    idx = torch.tensor([[4, 5, 6]], dtype=torch.long)  # 从汉字开始
    generated = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"生成序列: {generated[0].tolist()}")
    
    print("\n模型创建成功!")
