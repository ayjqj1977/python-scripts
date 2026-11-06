#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18维多尺度CNN+Transformer中文认知模型
==============================================
架构设计理念：适配18维编码的物理结构

18维编码分组：
- 维度 0-4 (5维): 四角号码   → 视觉/结构特征
- 维度 5-8 (4维): 五笔码     → 字形特征
- 维度 9-14 (6维): 拼音码    → 语音特征
- 维度 15 (1维): 声调        → 韵律特征
- 维度 16-17 (2维): 类型     → 语义类型

多尺度CNN设计：
- Stream1: 四角卷积 (kernel=2,3)   → 捕捉字形结构
- Stream2: 五笔卷积 (kernel=2,3)   → 捕捉字根组合
- Stream3: 拼音卷积 (kernel=3,4)   → 捕捉韵律
- Stream4: 声调嵌入              → 韵律辅助
- Stream5: 类型嵌入              → 语义门控

学习阶段：
- Stage1: 字典（单字）→ 自编码重构
- Stage2: 词典（词语）→ 词语预测
- Stage3: 诗词 → 诗句生成
- Stage4: 长文 → 篇章生成
"""

import os
import json
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ============== 配置 ==============
WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_extended.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_extended.npy")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

# 语料文件
CORPUS_DICT = os.path.join(WORK_DIR, "7_训练语料", "汉字.json")       # 字典
CORPUS_WORD = os.path.join(WORK_DIR, "7_训练语料", "词语.json")       # 词典
CORPUS_IDIOM = os.path.join(WORK_DIR, "7_训练语料", "成语.json")      # 成语
CORPUS_XHY = os.path.join(WORK_DIR, "7_训练语料", "歇后语.json")      # 歇后语
CORPUS_POETRY = os.path.join(WORK_DIR, "corpus_training.txt")         # 诗词

# 模型参数
VOCAB_SIZE = 10114
EMBED_DIM = 18
HIDDEN_DIM = 128
N_HEAD = 4
N_LAYER = 3
BLOCK_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# 训练阶段
STAGE_DICT = 1    # 字典阶段
STAGE_WORD = 2    # 词典阶段  
STAGE_POETRY = 3  # 诗词阶段
STAGE_LONG = 4    # 长文阶段

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 设备: {DEVICE}")

# ============== 加载数据 ==============
print("\n📚 加载18维向量...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

# 构建字符索引
char_vectors = {}
char_to_idx = {}
idx_to_char = {}

for i, char_info in enumerate(vector_data['characters']):
    char = char_info['char']
    vector = char_info['vector']
    char_vectors[char] = np.array(vector, dtype=np.float32)
    char_to_idx[char] = i
    idx_to_char[i] = char

# 添加未知字符 (索引为VOCAB_SIZE-1，即10113)
UNK_IDX = VOCAB_SIZE - 1
char_to_idx['<UNK>'] = UNK_IDX
idx_to_char[UNK_IDX] = '<UNK>'

print(f"   字符数: {len(char_to_idx)}")

# 加载嵌入矩阵
print("📊 加载嵌入矩阵...")
embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入形状: {embedding_weights.shape}")

# ============== 加载语料 ==============
print("\n📖 加载训练语料...")

# 阶段1: 字典数据
dict_data = []
try:
    with open(CORPUS_DICT, 'r', encoding='utf-8') as f:
        dict_raw = json.load(f)
        if isinstance(dict_raw, dict):
            for char, info in dict_raw.items():
                if char in char_to_idx:
                    dict_data.append(char)
        elif isinstance(dict_raw, list):
            for item in dict_raw:
                if isinstance(item, str) and len(item) == 1 and item in char_to_idx:
                    dict_data.append(item)
        print(f"   字典: {len(set(dict_data))} 个独体字")
except Exception as e:
    print(f"   字典加载失败: {e}")

# 阶段2: 词语数据
word_data = []
try:
    with open(CORPUS_WORD, 'r', encoding='utf-8') as f:
        word_raw = json.load(f)
        if isinstance(word_raw, list):
            for item in word_raw[:50000]:  # 取前5万条
                if isinstance(item, dict) and 'ci' in item:
                    word = item['ci']
                    if 2 <= len(word) <= 6:  # 2-6字词语
                        # 检查是否都在词表中
                        if all(c in char_to_idx for c in word):
                            word_data.append(word)
        print(f"   词语: {len(word_data)} 条")
except Exception as e:
    print(f"   词语加载失败: {e}")

# 阶段3: 诗词数据
poetry_data = []
try:
    with open(CORPUS_POETRY, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and 4 <= len(line) <= 128:
                # 检查是否都在词表中
                if all(c in char_to_idx or c == '，' or c == '。' or c == '、' for c in line):
                    poetry_data.append(line)
    print(f"   诗词: {len(poetry_data)} 首")
except Exception as e:
    print(f"   诗词加载失败: {e}")

# 阶段4: 长文数据
long_text_data = []
try:
    # 从诗词扩展
    long_text_data = poetry_data * 3  # 复制3次模拟长文
    print(f"   长文: {len(long_text_data)} 段")
except Exception as e:
    print(f"   长文加载失败: {e}")

# 合并所有数据用于训练
all_corpus = dict_data + word_data + poetry_data + long_text_data
random.shuffle(all_corpus)
print(f"   总语料: {len(all_corpus)} 条")

# ============== 编码函数 ==============
def text_to_indices(text):
    """文本转索引序列"""
    indices = []
    for c in text:
        if c in char_to_idx:
            indices.append(char_to_idx[c])
        elif c == '，' or c == '。' or c == '、':
            indices.append(UNK_IDX)  # 标点用UNK
    return indices

def indices_to_text(indices):
    """索引序列转文本"""
    result = []
    for idx in indices:
        if idx in idx_to_char and idx < VOCAB_SIZE:
            result.append(idx_to_char[idx])
    return ''.join(result)

# ============== 多尺度CNN+Transformer模型 ==============
class MultiScaleCharCNN(nn.Module):
    """
    多尺度分组CNN，适配18维编码结构
    
    输入: (batch, seq_len, 18)
    输出: (batch, seq_len, hidden_dim)
    """
    
    def __init__(self, embed_dim=18, hidden_dim=128):
        super().__init__()
        
        # Stream 1: 四角号码特征 (维度 0-4, 5维) → 结构特征
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=2, padding=0),  # 2-gram
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        
        # Stream 2: 五笔特征 (维度 5-8, 4维) → 字根特征
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        
        # Stream 3: 拼音特征 (维度 9-14, 6维) → 语音特征
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stream 4: 声调嵌入 (维度 15, 1维) → 韵律特征
        self.tone_embedding = nn.Embedding(5, 16)  # 5种声调
        
        # Stream 5: 类型嵌入 (维度 16-17, 2维) → 语义门控
        self.type_embedding = nn.Embedding(4, 16)  # 4种类型
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, 18)
        """
        batch, seq_len, _ = x.shape
        
        # 分割各维度组
        sijiao = x[:, :, 0:5].permute(0, 2, 1)  # (batch, 5, seq_len)
        wubi = x[:, :, 5:9].permute(0, 2, 1)     # (batch, 4, seq_len)
        pinyin = x[:, :, 9:15].permute(0, 2, 1) # (batch, 6, seq_len)
        tone = x[:, :, 15].long()                  # (batch, seq_len)
        char_type = x[:, :, 16].long()             # (batch, seq_len)
        
        # 各流卷积
        sj_out = self.sijiao_conv(sijiao)  # (batch, 32, seq_len-3)
        wb_out = self.wubi_conv(wubi)        # (batch, 32, seq_len-3)
        py_out = self.pinyin_conv(pinyin)   # (batch, 32, seq_len)
        
        # 调整维度对齐
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len]
        wb_out = wb_out[:, :, :min_len]
        py_out = py_out[:, :, :min_len]
        
        # 调整卷积输出维度: (batch, 32, min_len) -> (batch, min_len, 32)
        sj_out = sj_out.permute(0, 2, 1)
        wb_out = wb_out.permute(0, 2, 1)
        py_out = py_out.permute(0, 2, 1)
        
        # 声调和类型嵌入
        tone_emb = self.tone_embedding(tone[:, :min_len])  # (batch, min_len, 16)
        type_emb = self.type_embedding(char_type[:, :min_len])  # (batch, min_len, 16)
        
        # 融合: (batch, min_len, 32+32+32+16+16) = (batch, min_len, 132)
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        
        output = self.fusion(combined)  # (batch, min_len, hidden_dim)
        
        return output


class NeuroGlyph18(nn.Module):
    """
    完整模型: 多尺度CNN + Transformer
    """
    
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=128, n_head=4, n_layer=3, block_size=64):
        super().__init__()
        
        # 18维编码嵌入层 (不参与训练，直接查表)
        self.char_embedding = nn.Embedding(vocab_size + 1, embed_dim)
        
        # 加载预训练嵌入
        with torch.no_grad():
            # 扩展嵌入矩阵，添加UNK
            extended_embed = np.vstack([embedding_weights, np.zeros((1, embed_dim))])
            self.char_embedding.weight[:] = torch.from_numpy(extended_embed).float()
        
        # 多尺度CNN
        self.cnn = MultiScaleCharCNN(embed_dim, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Embedding(block_size, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # 输出层
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # 序列填充
        self.block_size = block_size
    
    def forward(self, idx, targets=None):
        """
        idx: (batch, seq_len) 字符索引
        """
        batch, seq_len = idx.shape
        device = idx.device
        
        # 查表获取18维编码
        x = self.char_embedding(idx)  # (batch, seq_len, 18)
        
        # CNN特征提取
        x = self.cnn(x)  # (batch, seq_len', hidden_dim)
        
        # 位置编码
        seq_len_prime = x.size(1)
        positions = torch.arange(seq_len_prime, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Transformer
        x = self.transformer(x)
        x = self.ln_f(x)
        
        # 预测
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # 对齐序列长度
            min_len = min(logits.size(1), targets.size(1))
            loss = F.cross_entropy(
                logits[:, :min_len, :].reshape(-1, logits.size(-1)),
                targets[:, :min_len].reshape(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# ============== 训练函数 ==============
def train_stage(stage_name, model, train_data, epochs=5):
    """训练单个阶段"""
    print(f"\n{'='*50}")
    print(f"📚 {stage_name}")
    print(f"{'='*50}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_data) // BATCH_SIZE)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_data) - BATCH_SIZE, BATCH_SIZE):
            batch_texts = train_data[i:i+BATCH_SIZE]
            
            # 准备数据
            max_len = min(max(len(t) for t in batch_texts) + 1, BLOCK_SIZE)
            
            x_list = []
            y_list = []
            for text in batch_texts:
                if len(text) < 2:
                    continue
                ids = text_to_indices(text)
                if len(ids) < 2:
                    continue
                ids = ids[:max_len-1]
                x_list.append(ids[:-1] + [0] * (max_len - 1 - len(ids[:-1])))
                y_list.append(ids[1:] + [0] * (max_len - 1 - len(ids[1:])))
            
            if not x_list:
                continue
            
            x = torch.tensor(x_list, dtype=torch.long).to(DEVICE)
            y = torch.tensor(y_list, dtype=torch.long).to(DEVICE)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            if loss:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def main():
    print("\n" + "="*60)
    print("🧠 NeuroGlyph-18: 多尺度CNN+Transformer认知模型")
    print("="*60)
    print(f"\n18维编码分组:")
    print("   维度 0-4: 四角号码 (结构)")
    print("   维度 5-8: 五笔字型 (字形)")
    print("   维度 9-14: 拼音编码 (语音)")
    print("   维度 15: 声调 (韵律)")
    print("   维度 16-17: 类型 (语义)")
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = NeuroGlyph18(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE
    ).to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params:,}")
    
    # 阶段1: 字典学习
    if dict_data:
        model = train_stage("阶段1: 字典（单字认知）", model, list(set(dict_data)), epochs=3)
    
    # 阶段2: 词语学习
    if word_data:
        model = train_stage("阶段2: 词典（词语理解）", model, word_data[:30000], epochs=5)
    
    # 阶段3: 诗词学习
    if poetry_data:
        model = train_stage("阶段3: 诗词（韵律掌握）", model, poetry_data, epochs=10)
    
    # 保存模型
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "neuroglyph18.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    
    print(f"\n✅ 模型已保存: {checkpoint_path}")
    
    # 测试生成
    print("\n🧪 测试生成...")
    model.eval()
    
    test_prompts = ['春', '江', '花', '月', '床前', '明月']
    
    for prompt in test_prompts:
        context = torch.tensor([text_to_indices(prompt)], dtype=torch.long).to(DEVICE)
        generated = model.generate(context, max_new_tokens=20, temperature=0.8)
        result = indices_to_text(generated[0].tolist())
        print(f"   '{prompt}' → '{result}'")


if __name__ == "__main__":
    main()
