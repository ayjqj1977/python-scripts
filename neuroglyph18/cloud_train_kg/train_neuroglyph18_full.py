# -*- coding: utf-8 -*-
"""
NeuroGlyph-18 完整版 - 云端GPU训练
================================
按用户18维编码结构设计:
- 四角号码(0-4): 5维 → Conv1d kernel=2
- 五笔编码(5-8): 4维 → Conv1d kernel=2
- 拼音编码(9-14): 6维 → Conv1d kernel=3
- 声调(15): 1维 → Embedding
- 类型(16-17): 2维 → Embedding

课程学习: 单字 → 词语 → 成语 → 诗词 → 古文 → 长文
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

# 完整版参数 - 发挥18维潜力
HIDDEN_DIM = 256      # 隐藏层
N_HEAD = 8            # 多头注意力
N_LAYER = 6           # Transformer层数
BLOCK_SIZE = 128      # 序列长度
BATCH_SIZE = 64       # batch大小
LEARNING_RATE = 1e-4  # 学习率
EPOCHS = 10           # 每阶段轮数

# 自动选择设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 设备: {DEVICE}")

# ============== 加载词表 ==============
print("\n📚 加载词表和嵌入...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
VOCAB_SIZE = len(char_to_idx)
EMBED_DIM = 18

print(f"   词表大小: {VOCAB_SIZE}")

embedding_weights = np.load(EMBED_FILE)
print(f"   嵌入矩阵: {embedding_weights.shape}")

# ============== 加载全部语料 ==============
print("\n📖 加载全部语料...")

def load_json_lines(filepath, key=None, max_items=None):
    """加载JSON数组或JSON Lines"""
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
            elif isinstance(raw, dict):
                if key:
                    text = raw.get(key, '')
                    if text:
                        texts.append(text)
                else:
                    for v in raw.values():
                        if isinstance(v, str):
                            texts.append(v)
    except Exception as e:
        print(f"   加载失败 {filepath}: {e}")
    return texts

def load_txt_lines(filepath, max_lines=None):
    """加载纯文本每行"""
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

# 阶段1: 单字 (字典)
corpus_files = {
    '汉字': os.path.join(WORK_DIR, "汉字.json"),
}

# 阶段2: 词语
word_files = {
    '词语': os.path.join(WORK_DIR, "词语.json"),
    '新华词语': os.path.join(WORK_DIR, "xinhua_word.json"),
}

# 阶段3: 成语/歇后语
idiom_files = {
    '成语': os.path.join(WORK_DIR, "成语.json"),
    '歇后语': os.path.join(WORK_DIR, "歇后语.json"),
}

# 阶段4: 诗词
poetry_files = {
    '唐诗': os.path.join(WORK_DIR, "corpus_training.txt"),
    '宋词': os.path.join(WORK_DIR, "corpus", "songci_10000.json"),
    '古诗': os.path.join(WORK_DIR, "corpus", "quantangshi_2000.json"),
}

# 阶段5: 古文
ancient_files = {
    '古籍': os.path.join(WORK_DIR, "corpus_ancient.txt"),
    '元曲': os.path.join(WORK_DIR, "corpus", "yuanqu_yuanqu.json"),
    '诗经': os.path.join(WORK_DIR, "corpus", "shijing_shijing.json"),
    '孟子': os.path.join(WORK_DIR, "corpus", "四书五经_mengzi.json"),
    '楚辞': os.path.join(WORK_DIR, "corpus", "楚辞_chuci.json"),
}

# 加载数据
stage1_data = []  # 单字
for name, path in corpus_files.items():
    if os.path.exists(path):
        data = load_json_lines(path, 'zi', max_items=10000)
        stage1_data.extend(data)
        print(f"   {name}: {len(data)} 条")

stage2_data = []  # 词语
for name, path in word_files.items():
    if os.path.exists(path):
        data = load_json_lines(path, 'ci', max_items=50000)
        stage2_data.extend(data)
        print(f"   {name}: {len(data)} 条")

stage3_data = []  # 成语
for name, path in idiom_files.items():
    if os.path.exists(path):
        data = load_json_lines(path, 'chengyu', max_items=20000)
        stage3_data.extend(data)
        print(f"   {name}: {len(data)} 条")

stage4_data = []  # 诗词
for name, path in poetry_files.items():
    if os.path.exists(path):
        if path.endswith('.txt'):
            data = load_txt_lines(path, max_lines=20000)
        else:
            data = load_json_lines(path, max_items=10000)
        stage4_data.extend(data)
        print(f"   {name}: {len(data)} 条")

stage5_data = []  # 古文
for name, path in ancient_files.items():
    if os.path.exists(path):
        if path.endswith('.txt'):
            data = load_txt_lines(path, max_lines=10000)
        else:
            data = load_json_lines(path, max_items=5000)
        stage5_data.extend(data)
        print(f"   {name}: {len(data)} 条")

# 去重
stage1_data = list(set(stage1_data))
stage2_data = list(set(stage2_data))
stage3_data = list(set(stage3_data))
stage4_data = list(set(stage4_data))
stage5_data = list(set(stage5_data))

print(f"\n📊 语料统计:")
print(f"   阶段1(单字): {len(stage1_data)}")
print(f"   阶段2(词语): {len(stage2_data)}")
print(f"   阶段3(成语): {len(stage3_data)}")
print(f"   阶段4(诗词): {len(stage4_data)}")
print(f"   阶段5(古文): {len(stage5_data)}")

# ============== 模型 ==============
class MultiScale18CNN(nn.Module):
    """多尺度CNN - 适配18维编码结构"""
    
    def __init__(self, embed_dim=18, hidden_dim=256):
        super().__init__()
        
        # Stream 1: 四角号码 (维度 0-4, 5维) → kernel=2 捕捉局部字形
        self.sijiao_conv = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 2: 五笔编码 (维度 5-8, 4维) → kernel=2 捕捉字根
        self.wubi_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        
        # Stream 3: 拼音编码 (维度 9-14, 6维) → kernel=3 捕捉韵律
        self.pinyin_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stream 4: 声调 (维度 15, 1维)
        self.tone_embedding = nn.Embedding(5, 32)
        
        # Stream 5: 类型 (维度 16-17, 2维)
        self.type_embedding = nn.Embedding(4, 32)
        
        # 融合层: 64*3 + 32 + 32 = 256
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 32 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 分割各维度组
        sijiao = x[:, :, 0:5].permute(0, 2, 1)
        wubi = x[:, :, 5:9].permute(0, 2, 1)
        pinyin = x[:, :, 9:15].permute(0, 2, 1)
        tone = torch.clamp(x[:, :, 15].long(), min=0, max=4)
        char_type = torch.clamp(x[:, :, 16].long(), min=0, max=3)
        
        # 卷积
        sj_out = self.sijiao_conv(sijiao)
        wb_out = self.wubi_conv(wubi)
        py_out = self.pinyin_conv(pinyin)
        
        # 维度对齐
        min_len = min(sj_out.size(2), wb_out.size(2), py_out.size(2))
        sj_out = sj_out[:, :, :min_len]
        wb_out = wb_out[:, :, :min_len]
        py_out = py_out[:, :, :min_len]
        
        # 调整维度
        sj_out = sj_out.permute(0, 2, 1)
        wb_out = wb_out.permute(0, 2, 1)
        py_out = py_out.permute(0, 2, 1)
        
        # 嵌入
        tone_emb = self.tone_embedding(tone[:, :min_len])
        type_emb = self.type_embedding(char_type[:, :min_len])
        
        # 融合
        combined = torch.cat([sj_out, wb_out, py_out, tone_emb, type_emb], dim=2)
        return self.fusion(combined)


class NeuroGlyph18Full(nn.Module):
    """完整版模型"""
    
    def __init__(self, vocab_size, embed_dim=18, hidden_dim=256, n_head=8, n_layer=6):
        super().__init__()
        
        # 字符嵌入
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 加载预训练嵌入
        with torch.no_grad():
            self.char_embedding.weight[:] = torch.from_numpy(embedding_weights).float()
        
        # 多尺度CNN
        self.cnn = MultiScale18CNN(embed_dim, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, hidden_dim)
        
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


# ============== 数据集 ==============
class TextDataset(Dataset):
    def __init__(self, texts, char_to_idx, block_size=128):
        self.data = []
        self.block_size = block_size
        
        for text in texts:
            # 转索引
            indices = []
            for c in text:
                if c in char_to_idx:
                    indices.append(char_to_idx[c])
                elif c in '，。、！？；：""''（）【】':
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
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), -1, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    
    return x_pad, y_pad


# ============== 训练 ==============
def train_stage(stage_name, model, data, epochs=10, save_name="checkpoint"):
    if len(data) < 10:
        print(f"   数据不足，跳过")
        return model
    
    dataset = TextDataset(data, char_to_idx, BLOCK_SIZE)
    if len(dataset) < 10:
        return model
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       collate_fn=collate_fn, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n   {stage_name}: {len(dataset)} 样本, {len(loader)} 批次")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(loader):
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
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"      Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(WORK_DIR, f"{save_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    return model


# ============== 主程序 ==============
print("\n" + "="*60)
print("🧠 NeuroGlyph-18 完整版 (云端GPU)")
print("="*60)

print(f"\n📐 模型架构:")
print(f"   隐藏维度: {HIDDEN_DIM}")
print(f"   注意力头: {N_HEAD}")
print(f"   Transformer层: {N_LAYER}")
print(f"   序列长度: {BLOCK_SIZE}")
print(f"   Batch大小: {BATCH_SIZE}")
print(f"   学习率: {LEARNING_RATE}")
print(f"   训练轮数: {EPOCHS}/阶段")

# 创建模型
model = NeuroGlyph18Full(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    n_head=N_HEAD,
    n_layer=N_LAYER
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️ 模型参数量: {total_params:,}")

# 课程学习训练
print("\n" + "="*60)
print("📚 课程学习训练")
print("="*60)

# 阶段1: 单字理解
if stage1_data:
    model = train_stage("阶段1: 单字理解", model, stage1_data[:50000], epochs=EPOCHS, save_name="stage1_char")

# 阶段2: 词语学习
if stage2_data:
    model = train_stage("阶段2: 词语学习", model, stage2_data[:100000], epochs=EPOCHS, save_name="stage2_word")

# 阶段3: 成语歇后语
if stage3_data:
    model = train_stage("阶段3: 成语学习", model, stage3_data[:50000], epochs=EPOCHS, save_name="stage3_idiom")

# 阶段4: 诗词
if stage4_data:
    model = train_stage("阶段4: 诗词韵律", model, stage4_data[:100000], epochs=EPOCHS, save_name="stage4_poetry")

# 阶段5: 古文
if stage5_data:
    model = train_stage("阶段5: 古文理解", model, stage5_data[:100000], epochs=EPOCHS, save_name="stage5_ancient")

# 保存最终模型
MODEL_PATH = os.path.join(WORK_DIR, "neuroglyph18_final.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': VOCAB_SIZE,
    'hidden_dim': HIDDEN_DIM,
    'n_head': N_HEAD,
    'n_layer': N_LAYER,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, MODEL_PATH)

print(f"\n✅ 训练完成!")
print(f"📁 模型保存: {MODEL_PATH}")
