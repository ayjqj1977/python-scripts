#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 训练脚本 - 直接读取JSON诗词文件
简单版本，直接用corpus文件夹里的JSON文件训练
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import time
import glob

print("="*50)
print("🚀 NanoText-18 训练 (直接读取JSON)")
print("="*50)

# ===================== 配置 =====================
WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')

BLOCK_SIZE = 64       # 上下文长度
BATCH_SIZE = 64       # 批量大小  
MAX_ITERS = 1000      # 训练迭代次数
LEARNING_RATE = 0.001 # 学习率
# =====================

# 加载词表
print("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表大小: {vocab_size}")

# 加载嵌入
embed_matrix = np.load(EMBED_FILE)
print(f"   嵌入维度: {embed_matrix.shape}")

# 从JSON文件读取诗词
print("\n📖 读取JSON诗词文件...")

def load_poems_from_json(filepath):
    """从JSON文件读取诗句"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 尝试不同字段
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
    texts = load_poems_from_json(json_file)
    if texts:
        print(f"   ✓ {filename}: {len(texts)} 条")
        all_texts.extend(texts)

print(f"\n   总共: {len(all_texts)} 条诗句")

# 合并为连续文本
full_text = ''.join(all_texts)
# 限制长度（防止内存不足）
max_chars = 200000
if len(full_text) > max_chars:
    full_text = full_text[:max_chars]
    print(f"   已截断到 {max_chars} 字符")

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
    
    if (iter_idx + 1) % 100 == 0:
        avg_loss = sum(losses[-100:]) / 100
        elapsed = time.time() - start_time
        remaining = (elapsed / (iter_idx + 1)) * (MAX_ITERS - iter_idx - 1)
        print(f"   迭代 {iter_idx+1:4d}/{MAX_ITERS} | 损失: {avg_loss:.4f} | 预计剩余: {remaining:.0f}秒")

# 保存
save_dir = os.path.join(WORK_DIR, 'checkpoints')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'json_train_{MAX_ITERS}.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print("\n" + "="*50)
print("✅ 训练完成!")
print("="*50)
print(f"💾 模型: {save_path}")
print(f"📉 最终损失: {losses[-1]:.4f}")
print(f"⏱️ 总用时: {time.time() - start_time:.1f}秒")
