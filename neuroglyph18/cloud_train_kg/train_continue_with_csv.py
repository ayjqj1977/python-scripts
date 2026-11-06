# -*- coding: utf-8 -*-
"""
NeuroGlyph-18 续训版 - 在现有模型基础上继续训练
================================================
1. 加载已有模型 neuroglyph18_final.pth
2. 加入知识图谱CSV大语料
3. 增量训练不丢失之前成果
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
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_FILE = os.path.join(WORK_DIR, "vocab_large.json")
EMBED_FILE = os.path.join(WORK_DIR, "embedding_large.npy")

# 现有模型
MODEL_FILE = os.path.join(WORK_DIR, "neuroglyph18_final.pth")

# 续训参数 - 适度学习
HIDDEN_DIM = 256
N_HEAD = 8
N_LAYER = 6
BLOCK_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 5e-5  # 低学习率微调
EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 设备: {DEVICE}")

# ============== 加载词表 ==============
print("\n📚 加载词表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
VOCAB_SIZE = len(char_to_idx)
EMBED_DIM = 18

print(f"   词表大小: {VOCAB_SIZE}")

embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

# ============== 加载已有模型 ==============
print("\n🔄 加载已有模型...")
checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
print(f"   从: {MODEL_FILE}")
print(f"   词表: {checkpoint.get('vocab_size', 'N/A')}")


# ============== 数据加载函数 ==============
def load_json_lines(filepath, key=None, max_items=None):
    """加载JSON"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            if isinstance(raw, list):
                items = raw[:max_items] if max_items else raw
                for item in items:
                    if key and isinstance(item, dict):
                        text = item.get(key, '')
                    else:
                        text = item if isinstance(item, str) else str(item)
                    if text:
                        texts.append(text)
    except Exception as e:
        print(f"   加载失败 {filepath}: {e}")
    return texts


def load_txt_lines(filepath, max_lines=None):
    """加载纯文本"""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:max_lines] if max_lines else f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    texts.append(line)
    except Exception as e:
        print(f"   加载失败 {filepath}: {e}")
    return texts


def load_csv_triples(filepath, max_lines=None):
    """加载CSV三元组知识图谱 (entity1, relation, entity2)"""
    texts = []
    try:
        import csv
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if max_lines and i >= max_lines:
                    break
                # 跳过表头
                if i == 0 and any('entity' in str(x).lower() for x in row[:3]):
                    continue
                # 尝试提取三元组
                for cell in row:
                    cell = str(cell).strip()
                    # 提取中文文本片段
                    chinese = ''.join(c for c in cell if '\u4e00' <= c <= '\u9fff')
                    if 2 <= len(chinese) <= 50:
                        texts.append(chinese)
                    # 整行中文
                    if len(cell) >= 4 and sum(1 for c in cell if '\u4e00' <= c <= '\u9fff') >= 3:
                        texts.append(cell[:100])
    except Exception as e:
        print(f"   CSV加载失败 {filepath}: {e}")
    return texts


# ============== 加载全部语料 ==============
print("\n📖 加载语料...")

all_texts = []

# 原有语料
existing_files = [
    ("汉字", os.path.join(WORK_DIR, "汉字.json"), 'zi', 10000),
    ("词语", os.path.join(WORK_DIR, "词语.json"), 'ci', 30000),
    ("成语", os.path.join(WORK_DIR, "成语.json"), 'chengyu', 15000),
    ("歇后语", os.path.join(WORK_DIR, "歇后语.json"), None, 5000),
    ("唐诗", os.path.join(WORK_DIR, "corpus_training.txt"), None, 20000),
    ("论语", os.path.join(WORK_DIR, "lunyu_lunyu.json"), 'content', 5000),
]

for name, path, key, limit in existing_files:
    if os.path.exists(path):
        if path.endswith('.txt'):
            data = load_txt_lines(path, limit)
        else:
            data = load_json_lines(path, key, limit)
        all_texts.extend(data)
        print(f"   {name}: {len(data)} 条")

# 知识图谱CSV (新增！)
csv_files = [
    ("语言学词典", os.path.join(WORK_DIR, "linguistics_all.csv"), 100000),
    ("艺术", os.path.join(WORK_DIR, "arts_all.csv"), 80000),
    ("哲学", os.path.join(WORK_DIR, "philosophy_all.csv"), 80000),
    ("历史", os.path.join(WORK_DIR, "history_all.csv"), 60000),
    ("文学", os.path.join(WORK_DIR, "literature_all.csv"), 50000),
    ("其他", os.path.join(WORK_DIR, "other_all.csv"), 30000),
]

print("\n🆕 知识图谱语料:")
for name, path, limit in csv_files:
    if os.path.exists(path):
        data = load_csv_triples(path, limit)
        all_texts.extend(data)
        print(f"   {name}: {len(data)} 条")

# 去重
all_texts = list(set(all_texts))
random.shuffle(all_texts)

print(f"\n📊 总语料: {len(all_texts)} 条")


# ============== 模型定义 ==============
class MultiScale18CNN(nn.Module):
    def __init__(self, embed_dim=18, hidden_dim=256):
        super().__init__()
        
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.tone_embedding = nn.Embedding(5, 32)
        self.type_embedding = nn.Embedding(4, 32)
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 32 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        sijiao = x[:, :, 0:5].permute(0, 2, 1)
        wubi = x[:, :, 5:9].permute(0, 2, 1)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)
        tone = torch.clamp(x[:, :, 15].long(), min=0, max=4)
        char_type = torch.clamp(x[:, :, 16].long(), min=0, max=3)
        
        sj_out = self.sijiao_conv(sijiao)
        wb_out = self.wubi_conv(wubi)
        py_out = self.pinyin_conv(pinyin)
        
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len].permute(0, 2, 1)
        wb_out = wb_out[:, :, :min_len].permute(0, 2, 1)
        py_out = py_out[:, :, :min_len].permute(0, 2, 1)
        
        tone_emb = self.tone_embedding(tone[:, :min_len])
        type_emb = self.type_embedding(char_type[:, :min_len])
        
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        return self.fusion(combined)


class NeuroGlyph18Continue(nn.Module):
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=256, n_head=8, n_layer=6):
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        x = self.char_embedding(idx)
        x = self.cnn(x)
        
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        x = self.transformer(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            if idx_next.item() in [0, 1, 2, 3]:
                break
        
        return idx


# ============== 数据集 ==============
class TextDataset(Dataset):
    def __init__(self, texts, char_to_idx, block_size=128):
        self.data = []
        self.block_size = block_size
        
        for text in texts:
            indices = []
            for c in text:
                if c in char_to_idx:
                    indices.append(char_to_idx[c])
                elif c in '，。、！？；：""''（）【】':
                    indices.append(char_to_idx.get(c, 0))
            
            if len(indices) >= 2:
                for i in range(len(indices) - 1):
                    self.data.append(indices[i:i+block_size+1])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), -1, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    
    return x_pad, y_pad


# ============== 主程序 ==============
print("\n" + "="*60)
print("🧠 NeuroGlyph-18 续训版 (加入知识图谱)")
print("="*60)

# 创建模型
model = NeuroGlyph18Continue(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    n_head=N_HEAD,
    n_layer=N_LAYER
)

# 加载已有权重
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️ 模型参数量: {total_params:,}")
print(f"✅ 已加载原有训练成果")

# 创建数据集
print("\n📦 创建数据集...")
dataset = TextDataset(all_texts, char_to_idx, BLOCK_SIZE)
print(f"   样本数: {len(dataset)}")

if len(dataset) < 100:
    print("❌ 数据太少，退出")
    exit(1)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 优化器 - 低学习率微调
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loader))

# 训练
print("\n" + "="*60)
print(f"📚 开始续训 (Epochs: {EPOCHS}, LR: {LEARNING_RATE})")
print("="*60)

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_idx % 200 == 0 and batch_idx > 0:
            print(f"   Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / max(batch_count, 1)
    print(f"\n   Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 保存
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(WORK_DIR, "neuroglyph18_continued.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            'vocab_size': VOCAB_SIZE,
            'hidden_dim': HIDDEN_DIM,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
        }, save_path)
        print(f"   ✅ 已保存最佳模型")

# 最终保存
final_path = os.path.join(WORK_DIR, "neuroglyph18_final_v2.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': VOCAB_SIZE,
    'hidden_dim': HIDDEN_DIM,
    'n_head': N_HEAD,
    'n_layer': N_LAYER,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, final_path)

print(f"\n🎉 续训完成!")
print(f"📁 最终模型: {final_path}")
