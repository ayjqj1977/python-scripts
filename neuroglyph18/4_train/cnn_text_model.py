#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN语言模型 - 支持序列预测
适合18维结构化向量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTextModel(nn.Module):
    """
    CNN序列模型 - 用于下一字符预测
    """
    def __init__(self, vocab_size, embed_dim=18, num_filters=128, 
                 filter_sizes=[2,3,4,5]):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN卷积层 - 多尺度
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # 输出层 - 每个位置预测
        self.fc = nn.Linear(num_filters * len(filter_sizes), vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # CNN需要 (batch, channels, seq)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # 多尺度卷积 - 先池化到相同长度再拼接
        conv_outputs = []
        for conv in self.convs:
            c = conv(embedded)  # (batch, num_filters, seq_len - kernel_size + 1)
            c = F.adaptive_max_pool1d(c, seq_len)  # 池化到原始长度
            conv_outputs.append(c)
        
        # 拼接所有卷积输出
        cat = torch.cat(conv_outputs, dim=1)  # (batch, num_filters*4, seq_len)
        
        # 投影到词汇表
        out = self.fc(cat.permute(0, 2, 1))  # (batch, seq_len, vocab_size)
        
        return out
    
    def generate(self, start_text, char_to_idx, idx_to_char, max_len=50, temperature=0.8, top_k=10, repeat_penalty=1.5):
        self.eval()
        
        # 编码
        context = [char_to_idx.get(c, 0) for c in start_text]
        context = context[-30:] if len(context) > 30 else context
        
        generated = context.copy()
        
        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([generated[-30:]], dtype=torch.long)
                if x.shape[1] < 30:
                    pad = torch.zeros(1, 30 - x.shape[1], dtype=torch.long)
                    x = torch.cat([pad, x], dim=1)
                
                logits = self.forward(x)
                next_logits = logits[0, -1, :].clone()
                
                # 重复惩罚
                for idx in generated[-5:]:
                    if idx < len(next_logits):
                        next_logits[idx] /= repeat_penalty
                
                # 温度采样
                if temperature > 0:
                    next_logits = next_logits / temperature
                
                # Top-K过滤
                if top_k > 0:
                    top_k_vals, top_k_idx = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                    filtered_logits = torch.full_like(next_logits, float('-inf'))
                    filtered_logits[top_k_idx] = top_k_vals
                    next_logits = filtered_logits
                
                probs = torch.softmax(next_logits, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                
                if next_idx == 0:
                    break
                    
                generated.append(next_idx)
        
        return ''.join([idx_to_char.get(i, '') for i in generated[len(context):]])

def create_cnn_model(vocab_size, embed_dim=18, **kwargs):
    return CNNTextModel(vocab_size, embed_dim, **kwargs)
