#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
后台训练脚本 - 3000次迭代
保存损失历史，训练完可查看
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

# 重定向输出到文件
log_file = open(r'D:\desktop\学习结果小文件\checkpoints\train_log.txt', 'w', encoding='utf-8')

def log(msg):
    print(msg, flush=True)
    print(msg, flush=True, file=log_file)

log("="*60)
log("🚀 NanoText-18 训练 (3000次迭代)")
log("="*60)

# 配置
WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 0.002

# 加载词表
log("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
log(f"   词表: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)

# 读取JSON
log("\n📖 读取诗词...")
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
for json_file in glob.glob(os.path.join(CORPUS_DIR, '*.json')):
    texts = load_poems(json_file)
    if texts:
        all_texts.extend(texts)

log(f"   总: {len(all_texts)} 条")

full_text = ''.join(all_texts)[:300000]
log(f"   字符: {len(full_text):,}")

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"\n🖥️  {device}")

# 模型
log("\n🏗️ 创建模型...")
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
log(f"   参数: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练
log("\n" + "="*60)
log("🎯 开始训练!")
log("="*60)

model.train()
start_time = time.time()
losses = []
loss_history = []

data_iter = iter(dl)
iter_idx = 0

while iter_idx < MAX_ITERS:
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(dl)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    iter_idx += 1
    
    if iter_idx % 10 == 0:
        avg_loss = sum(losses[-10:]) / 10
        elapsed = time.time() - start_time
        time_per = elapsed / iter_idx
        remaining = time_per * (MAX_ITERS - iter_idx)
        
        log(f"   {iter_idx:4d}/3000 | 损失: {avg_loss:.4f} | {elapsed/60:.1f}分钟 | 剩: {remaining/60:.1f}分钟")
        loss_history.append(avg_loss)

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'train_3000.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'loss_history': loss_history,
}, save_path)

log("\n" + "="*60)
log("✅ 完成!")
log("="*60)
log(f"   模型: {save_path}")
log(f"   损失: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
log(f"   时间: {(time.time()-start_time)/60:.1f}分钟")

log_file.close()
