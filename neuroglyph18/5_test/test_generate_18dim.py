# -*- coding: utf-8 -*-
"""
18维字符模型快速测试脚本
=======================
测试模型是否能生成汉字
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置
WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
VOCAB_SIZE = 10114
EMBED_DIM = 18
HIDDEN_DIM = 128
BLOCK_SIZE = 32

# 加载向量数据
print("📚 加载向量...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

char_to_idx = {}
idx_to_char = {}
for i, char_info in enumerate(vector_data['characters']):
    char = char_info['char']
    char_to_idx[char] = i
    idx_to_char[i] = char

# 加载嵌入
embedding_weights = np.load(EMBED_FILE)

# 定义简化模型（与训练脚本一致）
class MultiScaleCharCNN(nn.Module):
    def __init__(self, embed_dim=18, hidden_dim=128):
        super().__init__()
        
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.tone_embedding = nn.Embedding(5, 16)
        self.type_embedding = nn.Embedding(4, 16)
        
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        sijiao = x[:, :, 0:5].permute(0, 2, 1)
        wubi = x[:, :, 5:9].permute(0, 2, 1)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)
        tone = x[:, :, 15].long()
        char_type = x[:, :, 16].long()
        
        sj_out = self.sijiao_conv(sijiao)
        wb_out = self.wubi_conv(wubi)
        py_out = self.pinyin_conv(pinyin)
        
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len]
        wb_out = wb_out[:, :, :min_len]
        py_out = py_out[:, :, :min_len]
        
        sj_out = sj_out.permute(0, 2, 1)
        wb_out = wb_out.permute(0, 2, 1)
        py_out = py_out.permute(0, 2, 1)
        
        tone_emb = self.tone_embedding(tone[:, :min_len])
        type_emb = self.type_embedding(char_type[:, :min_len])
        
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        output = self.fusion(combined)
        
        return output


class NeuroGlyph18(nn.Module):
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=128, n_head=4, n_layer=2, block_size=32):
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        self.cnn = MultiScaleCharCNN(embed_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(block_size, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.block_size = block_size
    
    def forward(self, idx, targets=None):
        x = self.char_embedding(idx)
        x = self.cnn(x)
        
        seq_len_prime = x.size(1)
        positions = torch.arange(seq_len_prime, device=idx.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        x = self.transformer(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            min_len = min(logits.size(1), targets.size(1))
            loss = F.cross_entropy(
                logits[:, :min_len, :].reshape(-1, logits.size(-1)),
                targets[:, :min_len].reshape(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=1.0, top_k=10):
        """生成新字符"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            if idx_next.item() == 0:  # 遇到结束符
                break
        
        return idx


# 创建模型
print("🏗️ 创建模型...")
model = NeuroGlyph18(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model.eval()

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"   参数量: {total_params:,}")

# 测试生成 - 用"床前明月光"开头
print("\n📝 测试生成...")
test_prompts = ["床前明月光", "春眠不觉晓", "白日依山尽", "千山鸟飞绝"]

for prompt in test_prompts:
    indices = [char_to_idx.get(c, 0) for c in prompt]
    input_tensor = torch.tensor([indices], dtype=torch.long)
    
    output = model.generate(input_tensor, max_new_tokens=10, temperature=0.8, top_k=5)
    generated = output[0].tolist()
    
    result_chars = [idx_to_char.get(i, '?') for i in generated]
    print(f"   输入: {prompt}")
    print(f"   生成: {''.join(result_chars[len(prompt):])}")
    print()

print("✅ 测试完成!")
