# -*- coding: utf-8 -*-
"""
NeuroGlyph-18 轻量版 - CPU可跑
==============================
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============== 配置 ==============
WORK_DIR = r"D:\desktop\学习结果小文件"
VOCAB_FILE = os.path.join(WORK_DIR, "1_18维向量表", "vocab_large.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_large.npy")

CORPUS_DICT = os.path.join(WORK_DIR, "7_训练语料", "汉字.json")
CORPUS_WORD = os.path.join(WORK_DIR, "7_训练语料", "词语.json")
CORPUS_IDIOM = os.path.join(WORK_DIR, "7_训练语料", "成语.json")
CORPUS_POETRY = os.path.join(WORK_DIR, "corpus_training.txt")

# 轻量参数 - CPU也能跑
HIDDEN_DIM = 64   # 128→64
N_HEAD = 2        # 4→2
N_LAYER = 2       # 3→2
BLOCK_SIZE = 32   # 64→32
BATCH_SIZE = 16   # 64→16 (小batch省内存)
LEARNING_RATE = 1e-3
EPOCHS = 3        # 每阶段减少轮数

DEVICE = "cpu"    # 强制CPU
print(f"📱 设备: {DEVICE}")

# ============== 加载词表 ==============
print("\n📚 加载词表和嵌入...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
VOCAB_SIZE = len(char_to_idx)
EMBED_DIM = 18

print(f"   词表大小: {VOCAB_SIZE}")

embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

# ============== 加载语料 ==============
print("\n📖 加载语料...")

char_data = []
try:
    with open(CORPUS_DICT, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, dict):
            for char in raw.keys():
                if char in char_to_idx and len(char) == 1:
                    char_data.append(char)
    char_data = list(set(char_data))[:3000]
    print(f"   单字: {len(char_data)} 个")
except:
    pass

word_data = []
try:
    with open(CORPUS_WORD, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, list):
            for item in raw[:10000]:
                if isinstance(item, dict) and 'ci' in item:
                    word = item['ci']
                    if 2 <= len(word) <= 4 and all(c in char_to_idx for c in word):
                        word_data.append(word)
    print(f"   词语: {len(word_data)} 条")
except:
    pass

idiom_data = []
try:
    with open(CORPUS_IDIOM, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and 'chengyu' in item:
                    cy = item['chengyu']
                    if 4 <= len(cy) <= 6 and all(c in char_to_idx for c in cy):
                        idiom_data.append(cy)
    print(f"   成语: {len(idiom_data)} 条")
except:
    pass

poetry_data = []
try:
    with open(CORPUS_POETRY, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 4 <= len(line) <= 20 and all(c in char_to_idx or c in '，。、' for c in line):
                poetry_data.append(line)
    poetry_data = poetry_data[:5000]
    print(f"   诗词: {len(poetry_data)} 句")
except:
    pass

# ============== 模型 ==============
class MultiScale18CNN(nn.Module):
    def __init__(self, embed_dim=18, hidden_dim=64):
        super().__init__()
        
        # 四角号码 (5维) → kernel=2
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # 五笔 (4维) → kernel=2
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # 拼音 (6维) → kernel=3
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 声调
        self.tone_embedding = nn.Embedding(5, 8)
        
        # 类型
        self.type_embedding = nn.Embedding(4, 8)
        
        # 融合层: 16*3 + 8 + 8 = 64
        self.fusion = nn.Sequential(
            nn.Linear(16 + 16 + 16 + 8 + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        sijiao = x[:, :, 0:5].permute(0, 2, 1)
        wubi = x[:, :, 5:9].permute(0, 2, 1)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)
        tone = torch.clamp(x[:, :, 15].long(), min=0, max=4)
        char_type = torch.clamp(x[:, :, 16].long(), min=0, max=3)
        
        sj_out = self.sijiao_conv(sijiao)
        wb_out = self.wubi_conv(wubi)
        py_out = self.pinyin_conv(pinyin)
        
        # 维度对齐
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len]
        wb_out = wb_out[:, :, :min_len]
        py_out = py_out[:, :, :min_len]
        
        # 调整维度
        sj_out = sj_out.permute(0, 2, 1)
        wb_out = wb_out.permute(0, 2, 1)
        py_out = py_out.permute(0, 2, 1)
        
        # 嵌入
        tone_emb = self.tone_embedding(tone[:, :min_len])
        type_emb = self.type_embedding(char_type[:, :min_len])
        
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        return self.fusion(combined)


class NeuroGlyph18Lite(nn.Module):
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=64, n_head=2, n_layer=2):
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
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
    def generate(self, idx, max_new_tokens=20, temperature=1.0, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            if idx_next.item() in [0, 1, 2]:
                break
        
        return idx


# ============== 数据集 ==============
class TextDataset(Dataset):
    def __init__(self, texts, char_to_idx, block_size=32):
        self.data = []
        for text in texts:
            indices = [char_to_idx.get(c, 0) for c in text if c in char_to_idx or c in '，。、！？']
            if len(indices) >= 2:
                for i in range(len(indices) - 1):
                    self.data.append(indices[i:i+block_size+1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), -1, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    
    return x_pad, y_pad


# ============== 训练 ==============
def train_stage(stage_name, model, data, epochs=3):
    if len(data) < 10:
        return model
    
    dataset = TextDataset(data, char_to_idx, BLOCK_SIZE)
    if len(dataset) < 10:
        return model
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n   {stage_name}: {len(dataset)} 样本")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============== 主程序 ==============
print("\n" + "="*60)
print("🧠 NeuroGlyph-18 Lite (CPU版)")
print("="*60)

model = NeuroGlyph18Lite(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    n_head=N_HEAD,
    n_layer=N_LAYER
)

total_params = sum(p.numel() for p in model.parameters())
print(f"🏗️ 模型参数量: {total_params:,}")

# 课程学习
print("\n📚 课程学习训练")

if char_data:
    model = train_stage("阶段1: 单字", model, char_data[:2000], epochs=2)

if word_data:
    model = train_stage("阶段2: 词语", model, word_data[:5000], epochs=2)

all_poetry = idiom_data + poetry_data
if all_poetry:
    model = train_stage("阶段3: 诗词", model, all_poetry[:3000], epochs=2)

# 保存模型
MODEL_PATH = os.path.join(WORK_DIR, "neuroglyph18_lite.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': VOCAB_SIZE,
    'hidden_dim': HIDDEN_DIM,
    'n_head': N_HEAD,
    'n_layer': N_LAYER,
}, MODEL_PATH)

print(f"\n✅ 模型已保存: {MODEL_PATH}")
print("\n训练完成！")
