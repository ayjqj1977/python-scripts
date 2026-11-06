#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 快速测试版 - 100次迭代
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import time
import glob
import sys

print("="*50, flush=True)
print("🚀 NanoText-18 快速训练 (100次)", flush=True)
print("="*50, flush=True)

# 配置
WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 32   # 减小
MAX_ITERS = 100    # 快速测试
LEARNING_RATE = 0.002

# 加载词表
print("📚 加载词表...", flush=True)
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}", flush=True)

embed_matrix = np.load(EMBED_FILE)

# 读取JSON - 简化版
print("📖 读取诗词...", flush=True)
def load_poems(filepath):
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            content = item.get('content') or item.get('paragraphs') or []
            if isinstance(content, list):
                texts.extend([c.strip() for c in content if c.strip()])
    except: pass
    return texts

all_texts = []
for f in glob.glob(os.path.join(CORPUS_DIR, '*.json'))[:3]:  # 只读前3个
    texts = load_poems(f)
    if texts:
        print(f"   {os.path.basename(f)}: {len(texts)}", flush=True)
        all_texts.extend(texts[:500])  # 每文件最多500条

full_text = ''.join(all_texts[:10000])  # 限制1万字符
print(f"   总文本: {len(full_text)} 字符", flush=True)

# 编码
unk_idx = char_to_idx.get('<UNK>', 1)
indices = [char_to_idx.get(c, unk_idx) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)

# 数据集
class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"   批次: {len(dl)}", flush=True)

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  {device}", flush=True)

# 模型
print("🏗️ 创建模型...", flush=True)
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练
print("\n🎯 训练中...", flush=True)
start = time.time()
model.train()

for it, (x, y) in enumerate(dl):
    if it >= MAX_ITERS: break
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (it+1) % 20 == 0:
        print(f"   {it+1}/{MAX_ITERS} loss={loss.item():.4f}", flush=True)

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'quick_test.pt')
torch.save({'model_state_dict': model.state_dict(), 'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, save_path)

print(f"\n✅ 完成! 损失: {loss.item():.4f}", flush=True)
print(f"💾 {save_path}", flush=True)
print(f"⏱️ {time.time()-start:.1f}秒", flush=True)
