#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
快速测试 - 用全部语料训练500次
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
print("🚀 快速训练 (500次迭代)", flush=True)
print("="*50, flush=True)

# 配置
WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 500

# 加载
print("\n📚 加载...", flush=True)
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
embed_matrix = np.load(EMBED_FILE)
print(f"   词表: {vocab_size}", flush=True)

# 读取语料
print("📖 读取语料...", flush=True)
all_texts = []
for json_file in glob.glob(os.path.join(CORPUS_DIR, '*.json')):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                for key in ['content', 'paragraphs', 'text']:
                    if key in item:
                        content = item[key]
                        if isinstance(content, list):
                            for c in content:
                                if c and isinstance(c, str) and len(c.strip()) > 1:
                                    all_texts.append(c.strip())
                        break
    except: pass

print(f"   文本: {len(all_texts)}", flush=True)

full_text = ''.join(all_texts)[:300000]
print(f"   字符: {len(full_text):,}", flush=True)

# 编码
unk_idx = char_to_idx.get('<UNK>', 1)
indices = [char_to_idx.get(c, unk_idx) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)

class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

device = 'cpu'
print(f"\n🖥️  {device}", flush=True)

# 模型
print("🏗️ 创建模型...", flush=True)
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

# 训练
print("\n🎯 训练中...", flush=True)
start = time.time()
model.train()

data_iter = iter(dl)
for it in range(MAX_ITERS):
    try:
        x, y = next(data_iter)
    except:
        data_iter = iter(dl)
        x, y = next(data_iter)
    
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (it+1) % 50 == 0:
        print(f"   {it+1:4d}/500 | 损失: {loss.item():.4f} | {time.time()-start:.0f}秒", flush=True)

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'full_500.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print(f"\n✅ 完成! 损失: {loss.item():.4f}", flush=True)
print(f"💾 {save_path}", flush=True)
