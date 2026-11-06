#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
使用全部语料重新训练 - 包含四书五经、古文观止等
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
print("🚀 使用全部语料重新训练")
print("   包含: 诗经、论语、孟子、四书五经、古文观止等")
print("="*60)

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
print("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)

# 读取所有JSON语料
print("\n📖 读取所有语料...")

def load_all_poems():
    """读取corpus目录下所有JSON文件"""
    all_texts = []
    json_files = glob.glob(os.path.join(CORPUS_DIR, '*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    # 获取文本内容
                    content = None
                    for key in ['content', 'paragraphs', 'text', 'paragraph']:
                        if key in item:
                            content = item[key]
                            break
                    
                    if content and isinstance(content, list):
                        for c in content:
                            if c and isinstance(c, str) and len(c.strip()) > 0:
                                all_texts.append(c.strip())
                    elif content and isinstance(content, str):
                        if len(content.strip()) > 0:
                            all_texts.append(content.strip())
        except Exception as e:
            print(f"   警告: {os.path.basename(json_file)} - {e}")
    
    return all_texts

all_texts = load_all_poems()
print(f"   总文本数: {len(all_texts)}")

# 合并为训练文本
full_text = ''.join(all_texts)
print(f"   总字符数: {len(full_text):,}")

# 限制长度防止内存问题
max_chars = 500000
if len(full_text) > max_chars:
    full_text = full_text[:max_chars]
print(f"   训练字符: {len(full_text):,}")

# 编码
print("\n🔢 编码...")
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
print(f"   批次: {len(dl)}")

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  {device}")

# 模型
print("\n🏗️ 创建模型...")
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练
print("\n" + "="*60)
print("🎯 开始训练!")
print("="*60)

model.train()
start_time = time.time()
losses = []

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
        
        print(f"   {iter_idx:4d}/3000 | 损失: {avg_loss:.4f} | {elapsed/60:.1f}分钟 | 剩: {remaining/60:.1f}分钟")

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'full_corpus_3000.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print("\n" + "="*60)
print("✅ 完成!")
print("="*60)
print(f"   模型: {save_path}")
print(f"   损失: {losses[0]:.4f} → {losses[-1]:.4f}")
print(f"   时间: {(time.time()-start_time)/60:.1f}分钟")
