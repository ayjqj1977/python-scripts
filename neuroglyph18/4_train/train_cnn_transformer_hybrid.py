#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN + Transformer 混合模型
- CNN: 提取局部特征 (n-gram)
- Transformer: 捕捉长距离依赖
- 使用18维字符编码作为嵌入层
- 词汇表: 10114字
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ============= 配置 =============
WORK_DIR = r"D:\desktop\学习结果小文件"
POETRY_FILE = os.path.join(WORK_DIR, "corpus_training.txt")
KG_FILE = os.path.join(WORK_DIR, "corpus_knowledge_graph.txt")
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_extended.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

# 模型参数
VOCAB_SIZE = 10114
N_EMBD = 18
N_HEAD = 3          # 18能被3整除
N_LAYER = 4         # Transformer层数
CNN_KERNEL = 3      # CNN核大小
HIDDEN = 72         # FFN隐藏层 (18*4)
BLOCK_SIZE = 128     # 序列长度

# 训练参数
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TRAIN_ITERS = 3000  # 减少迭代次数加快训练
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📱 使用设备: {DEVICE}")
print(f"📊 模型配置: vocab={VOCAB_SIZE}, embed={N_EMBD}, head={N_HEAD}, layer={N_LAYER}")

# ============= 加载数据 =============
print("\n📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}
print(f"   词汇表: {len(char_to_idx)} 字")

print("📊 加载18维嵌入矩阵...")
embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

print("📖 加载语料...")

# 诗词
poetry_lines = []
with open(POETRY_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line) >= 4:
            poetry_lines.append(line)
print(f"   诗词: {len(poetry_lines)} 条")

# 知识图谱 (取50万条)
kg_lines = []
with open(KG_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 500000:  # 50万条
            break
        line = line.strip()
        if line and len(line) >= 2:
            kg_lines.append(line)
print(f"   知识图谱: {len(kg_lines)} 条")

# 合并
corpus = poetry_lines + kg_lines
random.shuffle(corpus)
print(f"   总语料: {len(corpus)} 条")

# 编码/解码
def encode(text):
    return [char_to_idx.get(c, 0) for c in text]

def decode(ids):
    return ''.join([idx_to_char.get(i, '一') for i in ids])

# ============= 数据集 =============
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = []
        self.block_size = block_size
        
        print("   构建训练序列...")
        total_tokens = 0
        for line in data:
            tokens = encode(line)
            if tokens:
                self.data.extend(tokens)
                total_tokens += len(tokens)
        
        print(f"   总token数: {total_tokens:,}")
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# ============= CNN + Transformer 混合模型 =============
class CNNTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, kernel_size, block_size, embedding_weights=None):
        super().__init__()
        
        self.d_model = d_model
        self.block_size = block_size
        
        # 嵌入层 (使用18维编码)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 加载预训练嵌入
        if embedding_weights is not None:
            with torch.no_grad():
                self.token_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        # 位置编码
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        # CNN层 (因果卷积)
        self.cnn = nn.Conv1d(
            d_model, d_model, 
            kernel_size, 
            padding=kernel_size//2,
            groups=d_model  # 深度可分离卷积，减少参数
        )
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 输出
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device
        
        # Token嵌入
        x = self.token_embedding(idx)  # (b, t, d_model)
        
        # CNN局部特征
        x_cnn = x.permute(0, 2, 1)  # (b, d_model, t)
        x_cnn = self.cnn(x_cnn)  # (b, d_model, t)
        x_cnn = x_cnn.permute(0, 2, 1)  # (b, t, d_model)
        x = x + 0.5 * x_cnn  # 残差连接
        
        # 位置编码
        pos = torch.arange(t, device=device).unsqueeze(0)
        x = x + self.position_embedding(pos)
        
        # Transformer
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20, repetition_penalty=1.2):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # 重复惩罚
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    for j in idx[i].tolist():
                        logits[i, j] /= repetition_penalty
            
            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 避免全是UNK
            if idx_next.item() == 0 and idx.size(1) > 20:
                break
        
        return idx

# ============= 训练 =============
def train():
    print("\n" + "="*60)
    print("🚀 训练 CNN + Transformer 混合模型")
    print("="*60)
    
    # 数据集
    print("\n📊 创建数据集...")
    dataset = TextDataset(corpus, BLOCK_SIZE)
    print(f"   数据集大小: {len(dataset):,}")
    
    # 模型
    print("\n🏗️ 创建模型...")
    model = CNNTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        kernel_size=CNN_KERNEL,
        block_size=BLOCK_SIZE,
        embedding_weights=embedding_weights
    ).to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_ITERS, eta_min=LEARNING_RATE/10)
    
    # 训练
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"\n📈 开始训练 ({TRAIN_ITERS} 轮)...")
    
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
        scheduler.step()
        
        if iter_num % 200 == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"   {iter_num}/{TRAIN_ITERS} | 损失: {loss.item():.4f} | lr: {lr:.6f} | {elapsed/60:.1f}分钟")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
    
    # 保存
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "cnn_transformer_hybrid.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    
    print(f"\n✅ 训练完成!")
    print(f"   最终损失: {best_loss:.4f}")
    print(f"   模型保存: {checkpoint_path}")
    
    # 测试
    print("\n🧪 测试生成...")
    model.eval()
    
    test_prompts = ['春', '江', '花', '月', '夜', '床', '明', '山水', '相思', '知识', '计算机', '哲学']
    
    for prompt in test_prompts:
        context = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)
        generated = model.generate(context, max_new_tokens=30, temperature=0.8, top_k=20)
        result = decode(generated[0].tolist())
        print(f"   '{prompt}' → '{result}'")

if __name__ == "__main__":
    train()
