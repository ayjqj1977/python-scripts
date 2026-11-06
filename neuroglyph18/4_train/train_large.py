# -*- coding: utf-8 -*-
"""
用大词表训练模型 - 避免UNK问题
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import glob

print("="*60)
print("🚀 使用大词表训练 (解决UNK问题)")
print("="*60)

WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')

# 使用大词表
VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_large.json')
EMBED_FILE = os.path.join(WORK_DIR, 'embedding_large.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 1000  # 先1000次测试

# 加载词表
print("\n📚 加载大词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)
print(f"   嵌入: {embed_matrix.shape}")

# 读取语料
print("\n📖 读取语料...")
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
                                if isinstance(c, str) and len(c.strip()) > 1:
                                    all_texts.append(c.strip())
    except: pass

print(f"   文本: {len(all_texts)}")

# 合并
full_text = ''.join(all_texts)[:400000]
print(f"   字符: {len(full_text):,}")

# 编码（使用新的词表）
unk_idx = char_to_idx.get('<UNK>', 1)
indices = [char_to_idx.get(c, unk_idx) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)

# 检查UNK比例
unk_count = sum(1 for i in indices if i == unk_idx)
print(f"   UNK比例: {100*unk_count/len(indices):.1f}%")

class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

device = 'cpu'
print(f"\n🖥️  {device}")

# 模型
print("\n🏗️ 创建模型...")
from nanotext18_model import create_model
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

# 训练
print("\n🎯 训练中...")
import time
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
    
    if (it+1) % 100 == 0:
        print(f"   {it+1:4d}/{MAX_ITERS} | 损失: {loss.item():.4f} | {time.time()-start:.0f}秒")

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'large_vocab_1000.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print(f"\n✅ 完成! 损失: {loss.item():.4f}")
print(f"💾 {save_path}")
