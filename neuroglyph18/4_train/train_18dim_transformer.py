#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用18维向量字符编码表训练Transformer模型
训练语料：诗词 + 知识图谱 + 汉字成语歇后语
"""

import os
import sys
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# 设置路径
WORK_DIR = r"D:\desktop\学习结果小文件"
CORPUS_FILE = os.path.join(WORK_DIR, "corpus_combined.txt")
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_extended.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

# 模型参数
VOCAB_SIZE = 10114  # 扩展词汇表大小
N_EMBD = 18         # 18维向量
N_HEAD = 3          # 注意力头数（18能被3整除）
N_LAYER = 6         # Transformer层数
BLOCK_SIZE = 128     # 上下文长度
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
TRAIN_ITERS = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]
print(f"   语料条数: {len(corpus)}")

# 语料预处理
def encode_text(text):
    """将文本编码为索引序列"""
    indices = []
    for char in text:
        if char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            # 未知字符用0代替
            indices.append(0)
    return indices

def decode_text(indices):
    """将索引序列解码为文本"""
    result = []
    for idx in indices:
        if idx in idx_to_char:
            result.append(idx_to_char[idx])
        elif idx == 0:
            result.append('一')  # 索引0对应'一'
    return ''.join(result)

# 创建训练数据
class TextDataset(Dataset):
    def __init__(self, corpus, block_size):
        self.data = []
        self.block_size = block_size
        
        # 将所有语料连接成一个大序列
        print("   构建训练序列...")
        for i, line in enumerate(corpus):
            if i % 1000000 == 0:
                print(f"      处理 {i}/{len(corpus)}...")
            tokens = encode_text(line)
            if tokens:
                self.data.extend(tokens)
        
        print(f"   总token数: {len(self.data)}")
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# 初始化嵌入层（使用18维编码）
class Embedding18(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 不使用预训练嵌入，而是直接使用18维编码
        # 18维编码是固定的：前17维是字符属性，第18维用于学习
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

# Transformer模型
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, embedding_weights=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 加载预训练的18维嵌入（如果提供）
        if embedding_weights is not None:
            print(f"   加载预训练嵌入矩阵，形状: {embedding_weights.shape}")
            with torch.no_grad():
                # embedding_weights 应该是 (vocab_size, 18)
                self.token_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (b, t, n_embd)
        
        # Position embeddings
        pos = torch.arange(t, device=device).unsqueeze(0)  # (1, t)
        pos_emb = self.position_embedding(pos)  # (1, t, n_embd)
        
        # 合并
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# 训练函数
def train():
    print("\n" + "="*60)
    print("🚀 开始训练18维向量Transformer模型")
    print("="*60)
    
    # 创建数据集
    print("\n📊 创建数据集...")
    dataset = TextDataset(corpus, BLOCK_SIZE)
    print(f"   数据集大小: {len(dataset)}")
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        embedding_weights=embedding_weights
    ).to(DEVICE)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    print(f"\n📈 开始训练 ({TRAIN_ITERS} 轮)...")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    for iter_num, (x, y) in enumerate(dataloader):
        if iter_num >= TRAIN_ITERS:
            break
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter_num % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   {iter_num}/{TRAIN_ITERS} | 损失: {loss.item():.4f} | lr: {current_lr:.6f} | {elapsed/60:.1f}分钟")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # 保存模型
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "transformer_18dim_knowledge.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'n_embd': N_EMBD,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
    }, checkpoint_path)
    
    print(f"\n✅ 训练完成!")
    print(f"   最终损失: {best_loss:.4f}")
    print(f"   模型保存: {checkpoint_path}")
    
    # 测试生成
    print("\n🧪 测试生成...")
    model.eval()
    test_prompts = ['春', '江', '花', '月', '夜', '山水', '相思', '知识', '计算机', '哲学']
    
    for prompt in test_prompts:
        context = torch.tensor([encode_text(prompt)], dtype=torch.long).to(DEVICE)
        generated = model.generate(context, max_new_tokens=30, temperature=0.8, top_k=20)
        result = decode_text(generated[0].cpu().tolist())
        print(f"   '{prompt}' → '{result}'")
    
    return model

if __name__ == "__main__":
    train()
