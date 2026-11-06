# -*- coding: utf-8 -*-
"""
CNN模型训练 - 快速版
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import glob
import sys
import time

print("="*60)
print("🚀 CNN模型训练 - 18维向量优化版")
print("="*60)

WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')

VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_extended.json')
EMBED_FILE = os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy')

BLOCK_SIZE = 32  # CNN用较短序列
BATCH_SIZE = 64
MAX_ITERS = 3000  # 增加迭代到3000

# 加载词汇表
print("\n📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)
print(f"   嵌入: {embed_matrix.shape}")

# 读取纯诗词文本
print("\n📖 读取诗词...")
all_texts = []

for json_file in glob.glob(os.path.join(CORPUS_DIR, '*.json')):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                for key in ['paragraphs', 'content']:
                    if key in item:
                        content = item[key]
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, str) and len(c.strip()) > 3:
                                    text = ''.join(ch for ch in c.strip() if ch in char_to_idx)
                                    if len(text) >= 4:
                                        all_texts.append(text)
    except: pass

print(f"   文本: {len(all_texts):,}")

# 合并
full_text = ''.join(all_texts)[:400000]
print(f"   字符: {len(full_text):,}")

# 编码
indices = [char_to_idx.get(c, 0) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)
print(f"   编码: {len(data):,}")

# 数据集
class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  {device}")

# CNN模型
print("\n🏗️ 创建CNN模型...")
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from cnn_text_model import create_cnn_model

# 减少参数的轻量模型
model = create_cnn_model(
    vocab_size=vocab_size,
    embed_dim=18,
    num_filters=64,  # 减少
    filter_sizes=[2,3,4],  # 减少
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   参数: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=0.0001)

# 训练
print("\n🎯 训练...")
start = time.time()
model.train()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

data_iter = iter(dl)
for it in range(MAX_ITERS):
    try:
        x, y = next(data_iter)
    except:
        data_iter = iter(dl)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    
    # 输入前N-1个，预测第N个
    x_input = x[:, :-1]  # (batch, seq_len-1)
    y_target = x[:, 1:]  # (batch, seq_len-1)
    
    logits = model(x_input)  # (batch, seq_len-1, vocab_size)
    
    # 计算损失
    loss = loss_fn(logits.reshape(-1, vocab_size), y_target.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (it+1) % 200 == 0:
        print(f"   {it+1:4d}/{MAX_ITERS} | 损失: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.5f} | {(time.time()-start)/60:.1f}分钟")

# 保存
print("\n💾 保存...")
save_path = os.path.join(WORK_DIR, 'checkpoints', 'cnn_model.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print(f"\n✅ 完成! 损失: {loss.item():.4f}")
print(f"   路径: {save_path}")
print(f"   耗时: {(time.time()-start)/60:.1f}分钟")
