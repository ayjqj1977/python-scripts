#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用18维向量字符编码表训练CNN模型
训练语料：诗词 + 知识图谱
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# 设置路径
WORK_DIR = r"D:\desktop\学习结果小文件"
POETRY_FILE = os.path.join(WORK_DIR, "corpus_training.txt")
KG_FILE = os.path.join(WORK_DIR, "corpus_knowledge_graph.txt")
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_extended.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

# 模型参数
VOCAB_SIZE = 10114
N_EMBD = 18
HIDDEN = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRAIN_ITERS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32

print(f"📱 使用设备: {DEVICE}")

# 加载词汇表
print("📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}
print(f"   词汇表大小: {len(char_to_idx)}")

# 加载18维嵌入矩阵
print("📊 加载18维嵌入矩阵...")
embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵形状: {embedding_weights.shape}")

# 加载语料
print("📖 加载语料...")

# 诗词语料
poetry_lines = []
with open(POETRY_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line) >= 2:
            poetry_lines.append(line)
print(f"   诗词: {len(poetry_lines)} 条")

# 知识图谱（限制数量）
kg_lines = []
with open(KG_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 300000:
            break
        line = line.strip()
        if line and len(line) >= 2:
            kg_lines.append(line)
print(f"   知识图谱: {len(kg_lines)} 条")

# 合并
corpus = poetry_lines + kg_lines
random.shuffle(corpus)
print(f"   总语料: {len(corpus)} 条")

# 编码
def encode_text(text):
    indices = []
    for char in text:
        if char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            indices.append(0)
    return indices

def decode_text(indices):
    result = []
    for idx in indices:
        if idx in idx_to_char:
            result.append(idx_to_char[idx])
        elif idx == 0:
            result.append('一')
    return ''.join(result)

# 简化版LM：基于字符的下一个字符预测
class CharLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_weights=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if embedding_weights is not None:
            with torch.no_grad():
                self.embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        # 简单MLP: 18*32 -> 256 -> 10114
        self.fc1 = nn.Linear(embed_dim * SEQ_LEN, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, targets=None):
        # x: (batch, seq_len)
        batch_size = x.size(0)
        
        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.view(batch_size, -1)  # (batch, seq_len * embed_dim)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        loss = None
        if targets is not None:
            # targets是(batch,) 即每个序列的下一个字符
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

# 训练
print("\n" + "="*60)
print("🚀 开始训练18维向量模型")
print("="*60)

# 创建模型
print("\n🏗️ 创建模型...")
model = CharLM(
    vocab_size=VOCAB_SIZE,
    embed_dim=N_EMBD,
    hidden_dim=HIDDEN,
    embedding_weights=embedding_weights
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"   模型参数量: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 准备训练数据：下一个字符预测
print("\n📊 准备训练数据...")
train_data = []
for line in corpus:
    tokens = encode_text(line)
    if len(tokens) >= 2:
        # 存储 (输入序列, 下一个字符)
        for i in range(len(tokens) - 1):
            if i + SEQ_LEN < len(tokens):
                input_seq = tokens[i:i+SEQ_LEN]
                next_char = tokens[i+SEQ_LEN]
                train_data.append((input_seq, next_char))

random.shuffle(train_data)
print(f"   有效训练样本: {len(train_data)}")

# 训练
model.train()
start_time = time.time()
best_loss = float('inf')

print(f"\n📈 开始训练 ({TRAIN_ITERS} 轮)...")

for iter_num in range(TRAIN_ITERS):
    # 采样批次
    batch = random.sample(train_data, min(BATCH_SIZE, len(train_data)))
    
    # 填充
    x_batch = torch.zeros(len(batch), SEQ_LEN, dtype=torch.long)
    y_batch = torch.zeros(len(batch), dtype=torch.long)
    
    for i, (x, y) in enumerate(batch):
        x_batch[i] = torch.tensor(x)
        y_batch[i] = y
    
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)
    
    optimizer.zero_grad()
    logits, loss = model(x_batch, y_batch)
    loss.backward()
    optimizer.step()
    
    if iter_num % 50 == 0:
        elapsed = time.time() - start_time
        print(f"   {iter_num}/{TRAIN_ITERS} | 损失: {loss.item():.4f} | {elapsed/60:.1f}分钟")
    
    if loss.item() < best_loss:
        best_loss = loss.item()

# 保存模型
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "char_lm_18dim.pt")
torch.save({
    'model_state_dict': model.state_dict(),
}, checkpoint_path)

print(f"\n✅ 训练完成!")
print(f"   最终损失: {best_loss:.4f}")
print(f"   模型保存: {checkpoint_path}")

# 测试生成
print("\n🧪 测试生成...")
model.eval()

def generate_text(prompt, max_len=30):
    context = encode_text(prompt)
    generated = context.copy()
    
    for _ in range(max_len):
        if len(context) < SEQ_LEN:
            context_padded = context + [0] * (SEQ_LEN - len(context))
        else:
            context_padded = context[-SEQ_LEN:]
        
        x = torch.tensor([context_padded], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[0] / 0.8  # temperature
            
            # top-k
            top_k = 20
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float('-inf')
            probs = F.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
        
        if next_token == 0:
            break
        
        generated.append(next_token)
        context.append(next_token)
    
    return decode_text(generated)

test_prompts = ['春', '江', '花', '月', '夜', '山', '水', '相思', '知识', '计算机']
for prompt in test_prompts:
    result = generate_text(prompt)
    print(f"   '{prompt}' → '{result}'")
