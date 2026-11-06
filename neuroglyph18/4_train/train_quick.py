# -*- coding: utf-8 -*-
"""
快速训练脚本 - 确保完成
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import sys
import time

print("="*60)
print("🚀 快速训练 - 确保完成")
print("="*60)

WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')
KG_DIR = os.path.join(WORK_DIR, 'onegraph_temp', 'OneGraphV2-中文')

VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_extended.json')
EMBED_FILE = os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy')

BLOCK_SIZE = 64
BATCH_SIZE = 64
MAX_ITERS = 3000

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

# 读取语料 - 精简
print("\n📖 读取语料...")
all_texts = []

# 诗词
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

print(f"   诗词: {len(all_texts)}")

# 知识图谱 - 限制
kg_count = 0
for csv_file in list(Path(KG_DIR).rglob('*.csv'))[:3]:
    if kg_count >= 300000:
        break
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', nrows=50000)
        if 'Head' in df.columns:
            for _, row in df.iterrows():
                head = str(row['Head'])[:10]
                tail = str(row.get('Tail', ''))[:10]
                if head and tail:
                    text = head + '是' + tail
                    filtered = ''.join(c for c in text if c in char_to_idx)
                    if len(filtered) >= 2:
                        all_texts.append(filtered)
                        kg_count += 1
    except: pass

print(f"   知识图谱: {kg_count}")

# 合并
full_text = ''.join(all_texts)[:800000]
print(f"   总字符: {len(full_text):,}")

# 编码
indices = [char_to_idx.get(c, 0) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)

# 数据集
class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  {device}")

# 模型
print("\n🏗️ 模型...")
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from nanotext18_model import create_model

embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
model = create_model(vocab_size, 18, 3, 6, BLOCK_SIZE, embedding_weights).to(device)
print(f"   参数: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

# 训练
print("\n🎯 训练...")
start = time.time()
model.train()

data_iter = iter(dl)
for it in range(MAX_ITERS):
    try:
        x, y = next(data_iter)
    except:
        data_iter = iter(dl)
        x, y = next(data_iter)
    
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (it+1) % 200 == 0:
        print(f"   {it+1:4d}/{MAX_ITERS} | 损失: {loss.item():.4f} | {(time.time()-start)/60:.1f}分钟")

# 保存
save_path = os.path.join(WORK_DIR, 'checkpoints', 'extended_vocab_3000.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print(f"\n✅ 完成! 损失: {loss.item():.4f}")
print(f"   耗时: {(time.time()-start)/60:.1f}分钟")
print(f"   路径: {save_path}")
