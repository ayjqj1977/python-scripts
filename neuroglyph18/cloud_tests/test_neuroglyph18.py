# -*- coding: utf-8 -*-
"""
NeuroGlyph-18 测试脚本 - 加载模型并测试生成
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== 配置 ==============
WORK_DIR = r"D:\desktop\学习结果小文件"
VOCAB_FILE = os.path.join(WORK_DIR, "1_18维向量表", "vocab_large.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_large.npy")
MODEL_FILE = os.path.join(WORK_DIR, "checkpoints", "neuroglyph18.pt")

# ============== 模型定义 ==============
class MultiScale18CNN(nn.Module):
    def __init__(self, embed_dim=18, hidden_dim=128):
        super().__init__()
        
        # Stream 1: 四角号码 (5维)
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 2: 五笔编码 (4维)
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 3: 拼音编码 (6维)
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 声调
        self.tone_embedding = nn.Embedding(5, 32)
        
        # 类型
        self.type_embedding = nn.Embedding(4, 32)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 32 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        sijiao = x[:, :, 0:5].permute(0, 2, 1)
        wubi = x[:, :, 5:9].permute(0, 2, 1)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)
        tone = torch.clamp(x[:, :, 15].long(), min=0, max=4)
        char_type = torch.clamp(x[:, :, 16].long(), min=0, max=3)
        
        sj_out = self.sijiao_conv(sijiao)
        wb_out = self.wubi_conv(wubi)
        py_out = self.pinyin_conv(pinyin)
        
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len].permute(0, 2, 1)
        wb_out = wb_out[:, :, :min_len].permute(0, 2, 1)
        py_out = py_out[:, :, :min_len].permute(0, 2, 1)
        
        tone_emb = self.tone_embedding(tone[:, :min_len])
        type_emb = self.type_embedding(char_type[:, :min_len])
        
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        return self.fusion(combined)


class NeuroGlyph18(nn.Module):
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=128, n_head=4, n_layer=4):
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        
        BLOCK_SIZE = 128
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        self.hidden_dim = hidden_dim
        self.block_size = BLOCK_SIZE
    
    def forward(self, idx, targets=None):
        x = self.char_embedding(idx)
        x = self.cnn(x)
        
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        x = self.transformer(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=30, temperature=0.8, top_k=15):
        BLOCK_SIZE = self.block_size
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            # 遇到结束符停止
            if idx_next.item() in [0, 1, 2, 3]:
                break
        
        return idx


# ============== 主程序 ==============
print("\n" + "="*60)
print("🧠 NeuroGlyph-18 模型测试")
print("="*60)

# 加载词表
print("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
VOCAB_SIZE = len(char_to_idx)

print(f"   词表大小: {VOCAB_SIZE}")

#加载嵌入
print("📥 加载嵌入矩阵...")
embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

# 加载模型
print("🔄 加载模型...")
device = "cpu"
model = NeuroGlyph18(
    vocab_size=VOCAB_SIZE,
    embed_dim=18,
    hidden_dim=128,
    n_head=4,
    n_layer=4
)

checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   模型已加载: {MODEL_FILE}")

# 测试生成
print("\n" + "="*60)
print("🎯 生成测试")
print("="*60)

test_prompts = [
    "天",
    "春江",
    "明月",
    "古诗词",
    "山高水长",
    "风花雪月",
    "人",
    "天地",
    "日照香炉生紫烟",
    "床前明月光",
]

for i, prompt in enumerate(test_prompts, 1):
    # 转换提示词为索引
    indices = [char_to_idx.get(c, 0) for c in prompt if c in char_to_idx or c in '，。、！？']
    
    if not indices:
        print(f"\n[{i}] \"{prompt}\" → 无法识别")
        continue
    
    # 生成
    input_idx = torch.tensor([indices], dtype=torch.long, device=device)
    output_idx = model.generate(input_idx, max_new_tokens=20, temperature=0.8)
    
    # 转换回文字
    generated_chars = ''.join([idx_to_char.get(idx, '') for idx in output_idx[0].tolist()])
    input_chars = ''.join([c for c in prompt if c in char_to_idx])
    
    print(f"\n[{i}] 输入: \"{input_chars}\"")
    print(f"    生成: \"{generated_chars}\"")

print("\n" + "="*60)
print("✅ 测试完成!")
print("="*60)
