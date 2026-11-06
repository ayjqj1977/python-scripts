#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 训练脚本 - 3000次迭代版
每10次显示损失变化，让你看到训练过程
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import time
import glob

print("="*60)
print("🚀 NanoText-18 训练 (3000次迭代)")
print("   每10次显示损失变化")
print("="*60)

# ===================== 配置 =====================
WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')

BLOCK_SIZE = 512       # 上下文长度
BATCH_SIZE = 32       # 批量大小  
MAX_ITERS = 3000      # 3000次迭代
LEARNING_RATE = 0.005 # 学习率
# =====================

# 加载词表
print("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表大小: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)
print(f"   嵌入维度: {embed_matrix.shape}")

# 读取JSON诗词
print("\n📖 读取JSON诗词文件...")

def load_poems(filepath):
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            content = item.get('content') or item.get('paragraphs') or []
            if isinstance(content, list):
                texts.extend([c.strip() for c in content if c.strip()])
    except Exception as e:
        print(f"   警告: {filepath} - {e}")
    return texts

all_texts = []
json_files = glob.glob(os.path.join(CORPUS_DIR, '*.json'))

for json_file in json_files:
    filename = os.path.basename(json_file)
    texts = load_poems(json_file)
    if texts:
        print(f"   ✓ {filename}: {len(texts)} 条")
        all_texts.extend(texts)

print(f"\n   总共: {len(all_texts)} 条诗句")

# 合并为连续文本
full_text = ''.join(all_texts)
# 限制长度
max_chars = 300000
if len(full_text) > max_chars:
    full_text = full_text[:max_chars]
print(f"   训练文本长度: {len(full_text):,} 字符")

# 编码
print("\n🔢 编码文本...")
unk_idx = char_to_idx.get('<UNK>', 1)
indices = [char_to_idx.get(c, unk_idx) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)
print(f"   编码完成: {len(data):,} tokens")

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
print(f"   数据批次: {len(train_loader)}")

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  使用设备: {device}")

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
print(f"   模型参数: {total_params:,}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练
print("\n" + "="*60)
print("🎯 开始训练! (每10次显示损失)")
print("="*60)

model.train()
start_time = time.time()
losses = []
loss_history = []  # 记录每10次的损失

# 训练循环
iter_idx = 0
data_iter = iter(train_loader)

while iter_idx < MAX_ITERS:
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    iter_idx += 1
    
    # 每10次显示损失
    if iter_idx % 10 == 0:
        current_loss = loss.item()
        avg_loss_10 = sum(losses[-10:]) / 10
        elapsed = time.time() - start_time
        
        # 预计剩余时间
        if iter_idx > 0:
            time_per_iter = elapsed / iter_idx
            remaining = time_per_iter * (MAX_ITERS - iter_idx)
            remaining_str = f"{remaining/60:.1f}分钟"
        else:
            remaining_str = "计算中..."
        
        print(f"   迭代 {iter_idx:4d}/3000 | 损失: {avg_loss_10:.4f} | 已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining_str}")
        
        # 记录用于绘图
        loss_history.append(avg_loss_10)

# 保存
save_dir = os.path.join(WORK_DIR, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'train_3000.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'loss_history': loss_history,
}, save_path)

# 打印损失变化
print("\n" + "="*60)
print("📉 损失变化总结")
print("="*60)
print(f"   第  100次: {loss_history[9]:.4f}")
print(f"   第  500次: {loss_history[49]:.4f}")
print(f"   第 1000次: {loss_history[99]:.4f}")
print(f"   第 1500次: {loss_history[149]:.4f}")
print(f"   第 2000次: {loss_history[199]:.4f}")
print(f"   第 2500次: {loss_history[249]:.4f}")
print(f"   第 3000次: {loss_history[-1]:.4f}")

total_time = time.time() - start_time
print(f"\n✅ 训练完成!")
print(f"   💾 模型: {save_path}")
print(f"   ⏱️  总用时: {total_time/60:.1f}分钟")
print(f"   📉 损失从 {loss_history[0]:.4f} 降到 {loss_history[-1]:.4f}")
