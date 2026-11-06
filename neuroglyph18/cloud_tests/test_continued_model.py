# -*- coding: utf-8 -*-
"""
NeuroGlyph-18 知识图谱续训模型测试
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK_DIR = r"D:\desktop\学习结果小文件"
CLOUD_DIR = os.path.join(WORK_DIR, "云端完整训练_含知识图谱")

VOCAB_FILE = os.path.join(CLOUD_DIR, "vocab_large.json")
EMBED_FILE = os.path.join(CLOUD_DIR, "embedding_large.npy")
MODEL_FILE = os.path.join(WORK_DIR, "neuroglyph18_continued.pth")


class MultiScale18CNN(nn.Module):
    def __init__(self, embed_dim=18, hidden_dim=256):
        super().__init__()
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1), nn.ReLU(),
        )
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1), nn.ReLU(),
        )
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.tone_embedding = nn.Embedding(5, 32)
        self.type_embedding = nn.Embedding(4, 32)
        self.fusion = nn.Sequential(
            nn.Linear(64+64+64+32+32, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
        )
    
    def forward(self, x):
        sijiao = x[:,:,0:5].permute(0,2,1)
        wubi = x[:,:,5:9].permute(0,2,1)
        pinyin = x[:,:,9:15].permute(0,2,1)
        tone = torch.clamp(x[:,:,15].long(), 0, 4)
        char_type = torch.clamp(x[:,:,16].long(), 0, 3)
        
        sj = self.sijiao_conv(sijiao)[:,:,:min(sj.size(2) for sj in [self.sijiao_conv(sijiao)])].permute(0,2,1) if False else self.sijiao_conv(sijiao).permute(0,2,1)
        wb = self.wubi_conv(wubi).permute(0,2,1)
        py = self.pinyin_conv(pinyin).permute(0,2,1)
        
        min_len = min(sj.size(1), wb.size(1), py.size(1))
        sj, wb, py = sj[:,:min_len,:], wb[:,:min_len,:], py[:,:min_len,:]
        
        return self.fusion(torch.cat([
            sj, wb, py,
            self.tone_embedding(tone[:,:min_len]),
            self.type_embedding(char_type[:,:min_len])
        ], dim=2))


class NeuroGlyph18(nn.Module):
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=256, n_head=8, n_layer=6):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)
        encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*4, dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.block_size = 128
    
    def forward(self, idx, targets=None):
        x = self.char_embedding(idx)
        x = self.cnn(x)
        x = x + self.pos_embedding(torch.arange(x.size(1), device=idx.device).unsqueeze(0))
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1) if targets is not None else None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new=30, temp=0.8, top_k=15):
        for _ in range(max_new):
            idx_cond = idx[:,-self.block_size:] if idx.size(1) > self.block_size else idx
            logits, _ = self.forward(idx_cond)
            logits = logits[:,-1,:] / temp
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:,[-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
            if idx_next.item() in [0,1,2,3]: break
        return idx


print("\n" + "="*60)
print("🧠 知识图谱续训模型测试")
print("="*60)

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx, idx_to_char = vocab['char_to_idx'], vocab['idx_to_char']
print(f"📚 词表: {len(char_to_idx)}")

embedding_weights = np.load(EMBED_FILE)
print(f"📥 嵌入: {embedding_weights.shape}")

print("🔄 加载模型...")
model = NeuroGlyph18(len(char_to_idx), 18, 256, 8, 6)
checkpoint = torch.load(MODEL_FILE, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ 模型已加载")

# 测试
print("\n" + "="*60)
print("🎯 生成测试")
print("="*60)

tests = ["天", "春", "明月", "日照香炉", "床前明月光", "春江花月夜", 
         "古诗词", "山高水长", "风花雪月", "人之初", "天地玄黄", "宇宙洪荒",
         "知识图谱", "人工智能", "深度学习"]

for i, p in enumerate(tests, 1):
    idx = [char_to_idx.get(c,0) for c in p if c in char_to_idx or c in '，。！？']
    if not idx: continue
    out = model.generate(torch.tensor([idx]), 25, 0.8)
    gen = ''.join(idx_to_char.get(str(i),'') for i in out[0].tolist())
    print(f"[{i:2d}] {p} → {gen}")

print("\n✅ 测试完成")
