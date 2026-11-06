#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用18维向量字符编码表训练CNN模型（更轻量）
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
from torch.utils.data import Dataset, DataLoader

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
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4, 5]
BATCH_SIZE = 128
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

# 诗词语料（保留）
poetry_lines = []
with open(POETRY_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line) >= 2:
            poetry_lines.append(line)
print(f"   诗词: {len(poetry_lines)} 条")

# 知识图谱（只取前50万条）
kg_lines = []
with open(KG_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 500000:  # 限制知识图谱数量
            break
        line = line.strip()
        if line and len(line) >= 2:
            kg_lines.append(line)
print(f"   知识图谱: {len(kg_lines)} 条")

# 合并语料
corpus = poetry_lines + kg_lines
random.shuffle(corpus)
print(f"   总语料: {len(corpus)} 条")

# 编码函数
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

# CNN文本分类模型（用于语言建模）
class CNNTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, embedding_weights=None):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 加载预训练嵌入
        if embedding_weights is not None:
            with torch.no_grad():
                self.embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vocab_size)
        )
    
    def forward(self, x, targets=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # 卷积 + 池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len - k + 1)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        # 合并所有卷积核的输出
        x = torch.cat(conv_outputs, dim=1)  # (batch, len(kernel_sizes) * num_filters)
        
        # 全连接
        logits = self.fc(x)
        
        loss = None
        if targets is not None:
            # 简单交叉熵
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成（简化版：每次预测一个token）"""
        model.eval()
        generated = idx.tolist()
        
        for _ in range(max_new_tokens):
            # 取最后SEQ_LEN个字符
            context = generated[-SEQ_LEN:] if len(generated) >= SEQ_LEN else generated
            context_tensor = torch.tensor([context + [0] * (SEQ_LEN - len(context))], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                logits, _ = model(context_tensor)
                logits = logits[0] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(0)))
                    logits[logits < v[-1]] = float('-inf')
                
                probs = F.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            if next_token == 0 and len(generated) > 10:  # 避免生成过多未知字符
                break
        
        return generated

# 训练
print("\n" + "="*60)
print("🚀 开始训练18维向量CNN模型")
print("="*60)

# 创建模型
print("\n🏗️ 创建模型...")
model = CNNTextModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=N_EMBD,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    embedding_weights=embedding_weights
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"   模型参数量: {total_params:,}")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 准备训练数据
print("\n📊 准备训练数据...")
random.shuffle(corpus)
train_data = []
for line in corpus:
    tokens = encode_text(line)
    if len(tokens) >= 2:
        train_data.append(tokens)

print(f"   有效训练样本: {len(train_data)}")

# 训练循环
model.train()
start_time = time.time()
best_loss = float('inf')

print(f"\n📈 开始训练 ({TRAIN_ITERS} 轮)...")

for iter_num in range(TRAIN_ITERS):
    # 随机采样批次
    batch = random.sample(train_data, min(BATCH_SIZE, len(train_data)))
    
    # 填充到相同长度
    max_len = max(len(x) for x in batch)
    max_len = min(max_len, SEQ_LEN)
    
    x_batch = []
    y_batch = []
    for seq in batch:
        seq = seq[:max_len]
        x_batch.append(seq[:-1])
        y_batch.append(seq[1:])
    
    # 填充
    x_padded = torch.zeros(len(x_batch), max_len - 1, dtype=torch.long)
    y_padded = torch.zeros(len(y_batch), max_len - 1, dtype=torch.long)
    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        x_padded[i, :len(x)] = torch.tensor(x)
        y_padded[i, :len(y)] = torch.tensor(y)
    
    x_padded = x_padded.to(DEVICE)
    y_padded = y_padded.to(DEVICE)
    
    optimizer.zero_grad()
    
    # 计算每个样本的损失
    total_loss = 0
    for i in range(len(x_padded)):
        logits_i, loss_i = model(x_padded[i:i+1], y_padded[i:i+1])
        total_loss += loss_i
    
    total_loss = total_loss / len(x_padded)
    total_loss.backward()
    optimizer.step()
    
    if iter_num % 100 == 0:
        elapsed = time.time() - start_time
        print(f"   {iter_num}/{TRAIN_ITERS} | 损失: {loss.item():.4f} | {elapsed/60:.1f}分钟")
    
    if loss.item() < best_loss:
        best_loss = loss.item()

# 保存模型
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "cnn_18dim_knowledge.pt")
torch.save({
    'model_state_dict': model.state_dict(),
}, checkpoint_path)

print(f"\n✅ 训练完成!")
print(f"   最终损失: {best_loss:.4f}")
print(f"   模型保存: {checkpoint_path}")

# 测试生成
print("\n🧪 测试生成...")
model.eval()
test_prompts = ['春', '江', '花', '月', '夜', '山', '水', '相思', '知识']

for prompt in test_prompts:
    context = encode_text(prompt)
    generated = model.generate(context, max_new_tokens=30, temperature=0.8, top_k=20)
    result = decode_text(generated)
    print(f"   '{prompt}' → '{result}'")
