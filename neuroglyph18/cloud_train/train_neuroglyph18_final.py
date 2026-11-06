# -*- coding: utf-8 -*-
"""
NeuroGlyph-18: 18维编码专用CNN+Transformer模型
==============================================
按用户18维编码结构设计:
- 四角号码(0-4): 5维 → 分组卷积 kernel=2
- 五笔编码(5-8): 4维 → 分组卷积 kernel=2
- 拼音编码(9-14): 6维 → 分组卷积 kernel=2
- 声调(15): 1维 → 嵌入层
- 类型(16-17): 2维 → 嵌入层

课程学习: 单字 → 词语 → 诗词 → 长文
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============== 配置 ==============
WORK_DIR = r"D:\desktop\学习结果小文件"
VOCAB_FILE = os.path.join(WORK_DIR, "1_18维向量表", "vocab_large.json")
EMBED_FILE = os.path.join(WORK_DIR, "2_嵌入矩阵", "embedding_large.npy")

# 语料文件
CORPUS_DICT = os.path.join(WORK_DIR, "7_训练语料", "汉字.json")       # 字典(单字)
CORPUS_WORD = os.path.join(WORK_DIR, "7_训练语料", "词语.json")       # 词语
CORPUS_IDIOM = os.path.join(WORK_DIR, "7_训练语料", "成语.json")      # 成语
CORPUS_POETRY = os.path.join(WORK_DIR, "corpus_training.txt")         # 诗词

# 模型参数
HIDDEN_DIM = 128
N_HEAD = 4
N_LAYER = 3
BLOCK_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 设备: {DEVICE}")

# ============== 加载词表和嵌入 ==============
print("\n📚 加载词表和嵌入...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
VOCAB_SIZE = len(char_to_idx)
EMBED_DIM = 18

print(f"   词表大小: {VOCAB_SIZE}")
print(f"   嵌入维度: {EMBED_DIM}")

# 加载预训练嵌入
embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

# ============== 加载语料 ==============
print("\n📖 加载语料...")

# 1. 单字数据
char_data = []
try:
    with open(CORPUS_DICT, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, dict):
            for char in raw.keys():
                if char in char_to_idx and len(char) == 1:
                    char_data.append(char)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and len(item) == 1 and item in char_to_idx:
                    char_data.append(item)
    char_data = list(set(char_data))
    print(f"   单字: {len(char_data)} 个")
except Exception as e:
    print(f"   单字加载失败: {e}")

# 2. 词语数据
word_data = []
try:
    with open(CORPUS_WORD, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, list):
            for item in raw[:50000]:
                if isinstance(item, dict) and 'ci' in item:
                    word = item['ci']
                    if 2 <= len(word) <= 6 and all(c in char_to_idx for c in word):
                        word_data.append(word)
    print(f"   词语: {len(word_data)} 条")
except Exception as e:
    print(f"   词语加载失败: {e}")

# 3. 成语数据
idiom_data = []
try:
    with open(CORPUS_IDIOM, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and len(item) == 4 and all(c in char_to_idx for c in item):
                    idiom_data.append(item)
    print(f"   成语: {len(idiom_data)} 条")
except Exception as e:
    print(f"   成语加载失败: {e}")

# 4. 诗词数据
poetry_data = []
try:
    with open(CORPUS_POETRY, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 4 <= len(line) <= 128:
                if all(c in char_to_idx or c in '，。、！？（）' for c in line):
                    poetry_data.append(line)
    print(f"   诗词: {len(poetry_data)} 首")
except Exception as e:
    print(f"   诗词加载失败: {e}")

# ============== 模型定义 ==============

class MultiScale18CNN(nn.Module):
    """
    针对18维编码的多尺度CNN
    输入: (batch, seq_len, 18)
    输出: (batch, seq_len', hidden_dim)
    """
    
    def __init__(self, embed_dim=18, hidden_dim=128):
        super().__init__()
        
        # Stream 1: 四角号码 (维度 0-4, 5维) → kernel=2 捕捉局部字形
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=2, padding=1),  # 保持序列
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 2: 五笔编码 (维度 5-8, 4维) → kernel=2 捕捉字根
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 3: 拼音编码 (维度 9-14, 6维) → kernel=3 捕捉韵律
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stream 4: 声调 (维度 15, 1维)
        self.tone_embedding = nn.Embedding(5, 16)  # 5种声调
        
        # Stream 5: 类型 (维度 16-17, 2维)
        # 2维 → 嵌入到16维
        self.type_embedding = nn.Embedding(4, 16)  # 4种类型
        
        # 融合层: 32*3 + 16 + 16 = 128维
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
        sijiao = x[:, :, 0:5].permute(0, 2, 1)   # (batch, 5, seq)
        wubi = x[:, :, 5:9].permute(0, 2, 1)      # (batch, 4, seq)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)   # (batch, 6, seq)
        tone = x[:, :, 15].long()                  # (batch, seq)
        char_type = x[:, :, 16].long()             # (batch, seq)
        
        # 钳制索引到有效范围
        tone = torch.clamp(tone, min=0, max=4)
        char_type = torch.clamp(char_type, min=0, max=3)
        
        # 各流卷积 (保持序列长度)
        sj_out = self.sijiao_conv(sijiao)  # (batch, 32, seq+1)
        wb_out = self.wubi_conv(wubi)        # (batch, 32, seq+1)
        py_out = self.pinyin_conv(pinyin)   # (batch, 32, seq)
        
        # 维度对齐：取最小长度
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len]
        wb_out = wb_out[:, :, :min_len]
        py_out = py_out[:, :, :min_len]
        
        # 调整维度: (batch, 32, min_len) → (batch, min_len, 32)
        sj_out = sj_out.permute(0, 2, 1)
        wb_out = wb_out.permute(0, 2, 1)
        py_out = py_out.permute(0, 2, 1)
        
        # 嵌入 (截断到min_len)
        tone_emb = self.tone_embedding(tone[:, :min_len])  # (batch, min_len, 16)
        type_emb = self.type_embedding(char_type[:, :min_len])  # (batch, min_len, 16)
        
        # 融合
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        output = self.fusion(combined)  # (batch, seq, hidden_dim)
        
        return output


class NeuroGlyph18(nn.Module):
    """
    完整模型: 18维专用CNN + Transformer语言模型
    """
    
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=128, n_head=4, n_layer=3, block_size=64):
        super().__init__()
        
        # 字符嵌入层 (18维)
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 加载预训练嵌入
        with torch.no_grad():
            if embed_dim == 18:
                self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        # 多尺度CNN
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        
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
        
        # 输出
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
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
            min_len = min(logits.size(1), targets.size(1))
            loss = F.cross_entropy(
                logits[:, :min_len, :].reshape(-1, logits.size(-1)),
                targets[:, :min_len].reshape(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=1.0, top_k=10):
        """生成字符"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            if idx_next.item() in [0, 1, 2]:  # 特殊token
                break
        
        return idx


# ============== 数据集 ==============
class TextDataset(Dataset):
    def __init__(self, texts, char_to_idx, block_size=64):
        self.data = []
        self.block_size = block_size
        
        for text in texts:
            # 转为索引
            indices = []
            for c in text:
                if c in char_to_idx:
                    indices.append(char_to_idx[c])
                elif c in '，。、！？（）【】':
                    indices.append(char_to_idx.get(c, 0))
            
            if len(indices) >= 2:
                # 滑动窗口
                for i in range(len(indices) - 1):
                    self.data.append(indices[i:i+block_size+1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """处理变长序列"""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), -1, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    
    return x_pad, y_pad


# ============== 训练 ==============
def train_stage(stage_name, model, data, epochs=5):
    """训练一个阶段"""
    if len(data) < 10:
        print(f"   数据不足，跳过")
        return model
    
    dataset = TextDataset(data, char_to_idx, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n   {stage_name}: {len(dataset)} 样本")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============== 主程序 ==============
print("\n" + "="*60)
print("🧠 NeuroGlyph-18: 18维编码专用CNN+Transformer")
print("="*60)

print("\n📐 模型架构:")
print(f"   四角号码(0-4): Conv1d kernel=2 × 2层")
print(f"   五笔编码(5-8): Conv1d kernel=2 × 2层")
print(f"   拼音编码(9-14): Conv1d kernel=3 × 2层")
print(f"   声调(15): Embedding(5, 16)")
print(f"   类型(16-17): Embedding(4, 16)")
print(f"   融合层: Linear(128, {HIDDEN_DIM})")
print(f"   Transformer: {N_LAYER}层, {N_HEAD}头")
print(f"   隐藏维度: {HIDDEN_DIM}")

# 创建模型
model = NeuroGlyph18(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️ 模型参数量: {total_params:,}")

# 课程学习训练
print("\n" + "="*60)
print("📚 课程学习训练")
print("="*60)

# 阶段1: 单字自编码
if char_data:
    model = train_stage("阶段1: 单字理解", model, char_data[:5000], epochs=3)

# 阶段2: 词语预测
if word_data:
    model = train_stage("阶段2: 词语学习", model, word_data[:30000], epochs=5)

# 阶段3: 成语+诗词
all_poetry = idiom_data + poetry_data
if all_poetry:
    model = train_stage("阶段3: 诗词韵律", model, all_poetry[:50000], epochs=8)

# 阶段4: 长文
if poetry_data and len(poetry_data) > 20:
    model = train_stage("阶段4: 长文理解", model, poetry_data[:30000], epochs=10)

# 保存模型
MODEL_PATH = os.path.join(WORK_DIR, "checkpoints", "neuroglyph18_final.pt")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, MODEL_PATH)
print(f"\n💾 模型已保存: {MODEL_PATH}")

# 测试生成
print("\n" + "="*60)
print("🧪 测试生成")
print("="*60)

test_prompts = ["床前明月光", "春眠不觉晓", "白日依山尽", "千山鸟飞绝"]

model.eval()
for prompt in test_prompts:
    indices = [char_to_idx.get(c, 0) for c in prompt]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(DEVICE)
    
    output = model.generate(input_tensor, max_new_tokens=10, temperature=0.8, top_k=5)
    generated = output[0].tolist()
    
    result_chars = [idx_to_char.get(i, '?') for i in generated]
    print(f"   输入: {prompt}")
    print(f"   生成: {''.join(result_chars[len(prompt):])}")
    print()

print("✅ 训练完成!")
