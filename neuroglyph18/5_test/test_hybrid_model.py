#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CNN+Transformer 混合模型生成效果
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# 设置路径
WORK_DIR = r"D:\desktop\学习结果小文件"
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_extended.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载词汇表
print("📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

# 加载嵌入矩阵
embedding_weights = np.load(EMBED_FILE)

# 编码/解码
def encode(text):
    return [char_to_idx.get(c, 0) for c in text]

def decode(ids):
    return ''.join([idx_to_char.get(i, '一') for i in ids])

# 模型定义
class CNNTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, kernel_size, block_size, embedding_weights=None):
        super().__init__()
        
        self.d_model = d_model
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if embedding_weights is not None:
            with torch.no_grad():
                self.token_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        self.cnn = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20, repetition_penalty=1.2):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    for j in idx[i].tolist():
                        logits[i, j] /= repetition_penalty
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == 0 and idx.size(1) > 20:
                break
        
        return idx

# 加载模型
print("📥 加载模型...")
model = CNNTransformerLM(
    vocab_size=10114,
    d_model=18,
    n_head=3,
    n_layer=4,
    kernel_size=3,
    block_size=128,
    embedding_weights=embedding_weights
).to(DEVICE)

checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "cnn_transformer_hybrid.pt"), map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ 模型加载完成!")
print("\n" + "="*60)
print("🧪 输入汉字测试生成效果（输入quit退出）")
print("="*60)

while True:
    prompt = input("\n请输入起始汉字: ").strip()
    
    if not prompt:
        continue
    
    if prompt.lower() in ['quit', 'q', '退出']:
        print("再见!")
        break
    
    # 生成
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)
    generated = model.generate(context, max_new_tokens=30, temperature=0.8, top_k=20)
    result = decode(generated[0].tolist())
    
    print(f"'{prompt}' → '{result}'")
