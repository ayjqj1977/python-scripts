# -*- coding: utf-8 -*-
"""
优化版训练 - 针对损失下降
核心改进：
1. 只用纯文本（诗词）训练，不混知识图谱
2. 增加模型容量
3. 优化学习率
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import glob
import sys
import time

print("="*60)
print("🚀 优化版训练 - 专注纯文本")
print("="*60)

WORK_DIR = r'D:\desktop\学习结果小文件'
CORPUS_DIR = os.path.join(WORK_DIR, 'corpus')

VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_extended.json')
EMBED_FILE = os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy')

BLOCK_SIZE = 128  # 增加上下文
BATCH_SIZE = 64
MAX_ITERS = 2000

# 加载词汇表
print("\n📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}")

embed_matrix = np.load(EMBED_FILE)

# 只读取诗词纯文本
print("\n📖 读取纯诗词文本...")
all_texts = []

# 诗词 - 全部
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
                                    # 只保留诗词中的汉字，去除杂质
                                    text = ''.join(ch for ch in c.strip() if ch in char_to_idx)
                                    if len(text) >= 4:
                                        all_texts.append(text)
    except: pass

print(f"   纯文本条数: {len(all_texts):,}")

# 合并 - 限制长度
full_text = ''.join(all_texts)
max_chars = 500000  # 50万字符足够
if len(full_text) > max_chars:
    full_text = full_text[:max_chars]
print(f"   总字符: {len(full_text):,}")

# 编码
indices = [char_to_idx.get(c, 0) for c in full_text]
data = torch.tensor(indices, dtype=torch.long)
print(f"   编码后: {len(data):,}")

# 数据集
class DS(Dataset):
    def __init__(self, d, bs): self.d, self.bs = d, bs
    def __len__(self): return len(self.d) - self.bs
    def __getitem__(self, i): return self.d[i:i+self.bs], self.d[i+1:i+self.bs+1]

ds = DS(data, BLOCK_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  {device}")

# 模型 - 增加容量
print("\n🏗️ 创建更大模型...")
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from nanotext18_model import create_model

# 使用更大的配置
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
# 增加层数和头数
model = create_model(
    vocab_size=vocab_size,
    n_embd=18,
    n_head=6,  # 增加到6个头
    n_layer=12,  # 增加层数到12
    block_size=BLOCK_SIZE,
    embedding_weights=embedding_weights
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   参数: {total_params:,} (原来是38万)")

# 使用更好的优化器设置
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if (it+1) % 100 == 0:
        elapsed = time.time() - start
        print(f"   {it+1:4d}/{MAX_ITERS} | 损失: {loss.item():.4f} | {elapsed/60:.1f}分钟")

# 保存
print("\n💾 保存...")
save_path = os.path.join(WORK_DIR, 'checkpoints', 'optimized_model.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, save_path)

print(f"\n✅ 完成!")
print(f"   损失: {loss.item():.4f}")
print(f"   路径: {save_path}")
print(f"   耗时: {(time.time()-start)/60:.1f}分钟")
