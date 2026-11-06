#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 快速训练脚本 (小白版)
简化版本，500次迭代，大约5-10分钟
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import time

print("="*50)
print("🚀 开始训练 NanoText-18 模型")
print("="*50)

# ===================== 配置 =====================
WORK_DIR = r'D:\desktop\学习结果小文件'
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
DATA_FILE = os.path.join(WORK_DIR, 'corpus_ancient_combined.txt')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')
BLOCK_SIZE = 64       # 上下文长度
BATCH_SIZE = 64       # 批量大小
MAX_ITERS = 500       # 训练迭代次数（减少以加快速度）
LEARNING_RATE = 0.001 # 学习率
# =====================

# 检查文件
if not os.path.exists(VOCAB_FILE):
    print(f"❌ 错误: 找不到词表文件 {VOCAB_FILE}")
    exit()
if not os.path.exists(DATA_FILE):
    print(f"❌ 错误: 找不到训练数据 {DATA_FILE}")
    exit()

print(f"\n📁 使用的数据: {os.path.basename(DATA_FILE)}")

# 加载词表
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"📚 词表大小: {vocab_size}")

# 加载嵌入
embed_matrix = np.load(EMBED_FILE)
print(f"📐 嵌入维度: {embed_matrix.shape}")

# 加载训练数据
print("\n📖 加载训练数据...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()
text = text[:500000]  # 限制长度加快速度
print(f"📝 训练文本长度: {len(text):,} 字符")

# 编码
indices = [char_to_idx.get(c, char_to_idx.get('<UNK>', 1)) for c in text]
data = torch.tensor(indices, dtype=torch.long)
print(f"✅ 数据编码完成: {len(data):,} tokens")

# 数据集
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

dataset = TextDataset(data, BLOCK_SIZE)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"📦 数据批次: {len(train_loader)}")

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  使用设备: {device}")

# 创建模型
print("\n🏗️ 创建模型...")
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(
    vocab_size=vocab_size,
    n_embd=18,
    n_head=3,
    n_layer=6,
    block_size=BLOCK_SIZE,
    embedding_weights=embedding_weights
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"🔧 模型参数: {total_params:,}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练
print("\n" + "="*50)
print("🎯 开始训练!")
print("="*50)
model.train()
start_time = time.time()
losses = []

for iter_idx, (x, y) in enumerate(train_loader):
    if iter_idx >= MAX_ITERS:
        break
    
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    
    if (iter_idx + 1) % 50 == 0:
        avg_loss = sum(losses[-50:]) / 50
        elapsed = time.time() - start_time
        print(f"  迭代 {iter_idx+1:4d}/{MAX_ITERS} | 损失: {avg_loss:.4f} | 已用时: {elapsed:.1f}秒")

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'quick_train.pt')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print("\n" + "="*50)
print("✅ 训练完成!")
print("="*50)
print(f"💾 模型已保存到: {save_path}")
print(f"📉 最终损失: {losses[-1]:.4f}")
print(f"⏱️ 总用时: {time.time() - start_time:.1f}秒")
