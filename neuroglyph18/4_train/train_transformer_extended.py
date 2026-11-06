#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
Transformer模型训练 - 扩展词汇表+诗词数据
使用新的10114词汇表
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import glob
import time
import sys

print("="*60)
print("🚀 Transformer模型训练 - 扩展词汇表+诗词")
print("="*60)

WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')

VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_extended.json')
EMBED_FILE = os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 2000

# 加载扩展词汇表
print("\n📚 加载扩展词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表大小: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embed_matrix.shape}")

# 读取诗词
print("\n📖 读取诗词语料...")
all_texts = []

# 诗词文件
poetry_files = [
    os.path.join(CORPUS_DIR, 'quantangshi_2000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_3000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_4000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_5000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_6000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_7000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_8000.json'),
    os.path.join(CORPUS_DIR, 'quantangshi_9000.json'),
    os.path.join(CORPUS_DIR, 'songci_2000.json'),
    os.path.join(CORPUS_DIR, 'songci_3000.json'),
    os.path.join(CORPUS_DIR, 'songci_4000.json'),
    os.path.join(CORPUS_DIR, 'songci_5000.json'),
    os.path.join(CORPUS_DIR, 'songci_6000.json'),
    os.path.join(CORPUS_DIR, 'songci_7000.json'),
    os.path.join(CORPUS_DIR, 'songci_8000.json'),
    os.path.join(CORPUS_DIR, 'songci_9000.json'),
    os.path.join(CORPUS_DIR, 'songci_10000.json'),
    os.path.join(CORPUS_DIR, '诗经_shijing.json'),
    os.path.join(CORPUS_DIR, '楚辞_chuci.json'),
    os.path.join(CORPUS_DIR, '蒙学_sanzijing-new.json'),
    os.path.join(CORPUS_DIR, '蒙学_dizigui.json'),
    os.path.join(CORPUS_DIR, '蒙学_baijiaxing.json'),
]

for pf in poetry_files:
    if os.path.exists(pf):
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for poem in data:
                if 'paragraphs' in poem:
                    for para in poem['paragraphs']:
                        text = ''.join([c for c in para if c in char_to_idx])
                        if len(text) >= 4:
                            all_texts.append(text)
        except Exception as e:
            pass

# 合并文本
full_text = '\n'.join(all_texts)
print(f"   总文本: {len(full_text):,}字符")

# 编码
data = [char_to_idx.get(c, 0) for c in full_text if c in char_to_idx]
print(f"   Token: {len(data):,}")

# 数据集
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, i):
        x = torch.tensor(self.data[i:i+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[i+1:i+self.block_size+1], dtype=torch.long)
        return x, y

ds = TextDataset(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"   批次: {len(dl):,}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  设备: {device}")

# 创建Transformer模型
print("\n🏗️ 创建Transformer模型...")
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from nanotext18_model import create_model

# 使用扩展词汇表的嵌入
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)

model = create_model(
    vocab_size=vocab_size,
    n_embd=18,
    n_head=3,  # 18能被3整除
    n_layer=6,
    block_size=BLOCK_SIZE,
    embedding_weights=embedding_weights
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   参数: {total_params:,}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=0.00005)
loss_fn = nn.CrossEntropyLoss()

# 训练
print("\n🎯 训练...")
start = time.time()
model.train()

best_loss = float('inf')
data_iter = iter(dl)

for it in range(MAX_ITERS):
    try:
        x, y = next(data_iter)
    except:
        data_iter = iter(dl)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    
    logits, loss = model(x, y)
    loss_val = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if loss_val < best_loss:
        best_loss = loss_val
    
    if (it+1) % 200 == 0:
        print(f"   {it+1:4d}/{MAX_ITERS} | 损失: {loss_val:.4f} | 最佳: {best_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.6f} | {(time.time()-start)/60:.1f}分钟")

# 保存模型
print("\n💾 保存模型...")
save_path = os.path.join(WORK_DIR, 'checkpoints', 'transformer_poetry_extended.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'config': {
        'vocab_size': vocab_size,
        'n_embd': 18,
        'n_head': 4,
        'n_layer': 6,
        'block_size': BLOCK_SIZE
    }
}, save_path)

print(f"\n✅ 完成!")
print(f"   保存路径: {save_path}")
print(f"   最终损失: {loss_val:.4f}")
print(f"   最佳损失: {best_loss:.4f}")
print(f"   总耗时: {(time.time()-start)/60:.1f}分钟")
