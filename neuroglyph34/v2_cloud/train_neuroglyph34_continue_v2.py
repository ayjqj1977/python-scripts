#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGlyph-34 Cloud V2 — 知识图谱续训脚本
===========================================
加载主训练产出的模型，在大规模知识图谱CSV语料上低学习率微调。
借鉴18维项目的续训设计：用 knowledge_graph/ 下10个CSV补充领域知识。
"""
import os, sys, json, math, time, random, logging, argparse, warnings, glob, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = r"D:\Desktop\neuroglyph34_project"
kg_dir = os.path.join(data_dir, "knowledge_graph")
checkpoint_dir = os.path.join(project_dir, "checkpoints_continue")

# 主训练产出的模型
MODEL_CKPT = os.path.join(project_dir, "checkpoints", "final.pt")
# 如果final不存在，回退到latest
if not os.path.exists(MODEL_CKPT):
    MODEL_CKPT = os.path.join(project_dir, "checkpoints", "latest.pt")

embedding_file = os.path.join(data_dir, "embedding_matrix.npy")
vocab_file = os.path.join(data_dir, "char_to_idx.json")

# 续训参数
MICRO_BATCH = 16
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-5       # 低学习率微调
EPOCHS = 5
MAX_SEQ_LEN = 256
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
SAVE_INTERVAL = 2000
USE_AMP = True
MAX_KG_LINES_PER_FILE = 30000

logger.info(f"主模型路径: {MODEL_CKPT}")
logger.info(f"是否存在: {os.path.exists(MODEL_CKPT)}")


# ==================== 知识图谱数据加载 ====================

def load_kg_csv(filepath, max_lines=None):
    """从知识图谱CSV提取中文文本片段"""
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if max_lines and i >= max_lines:
                    break
                if i == 0 and any("entity" in str(x).lower() for x in row[:3]):
                    continue
                for cell in row:
                    cell = str(cell).strip()
                    chinese = "".join(c for c in cell if "\u4e00" <= c <= "\u9fff")
                    if 4 <= len(chinese) <= 80:
                        texts.append(chinese)
                    if len(cell) >= 6 and sum(1 for c in cell if "\u4e00" <= c <= "\u9fff") >= 4:
                        texts.append(cell[:120])
    except Exception as e:
        logger.warning(f"  加载CSV失败 {os.path.basename(filepath)}: {e}")
    return texts


def load_json_texts(filepath, key=None, max_items=None):
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items[:max_items] if max_items else items:
            if key and isinstance(item, dict):
                text = item.get(key, "")
            elif isinstance(item, str):
                text = item
            else:
                continue
            if text:
                texts.append(text)
    except:
        pass
    return texts


def load_txt_lines(filepath, max_lines=None):
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
    except:
        pass
    return texts


def load_all_continue_data():
    """加载知识图谱语料 + 部分原有语料混入"""
    all_texts = []

    # 知识图谱CSV
    kg_files = [
        ("linguistics_all.csv", 30000),
        ("arts_all.csv", 20000),
        ("philosophy_all.csv", 20000),
        ("history_all.csv", 20000),
        ("literature_all.csv", 15000),
        ("other_all.csv", 10000),
        ("agricultural_resources_environment_all.csv", 8000),
        ("agronomy_all.csv", 8000),
        ("horticulture_all.csv", 5000),
        ("veterinary_medicine_all.csv", 5000),
    ]
    logger.info("\n📊 知识图谱语料:")
    for fname, limit in kg_files:
        fpath = os.path.join(kg_dir, fname)
        if os.path.exists(fpath):
            t = load_kg_csv(fpath, limit)
            all_texts.extend(t)
            logger.info(f"  {fname}: {len(t)} 条")

    # 混入部分原有语料防止灾难性遗忘
    existing_sources = [
        ("7_训练语料/汉字.json", "zi", 10000),
        ("7_训练语料/词语.json", "ci", 20000),
        ("7_训练语料/成语.json", "chengyu", 15000),
        ("corpus_training.txt", None, 15000),
        ("corpus_ancient.txt", None, 10000),
    ]
    logger.info("\n📖 混入已有语料:")
    for path, key, limit in existing_sources:
        fpath = os.path.join(data_dir, path)
        if os.path.exists(fpath):
            if fpath.endswith(".txt"):
                t = load_txt_lines(fpath, limit)
            else:
                t = load_json_texts(fpath, key, limit)
            all_texts.extend(t)
            logger.info(f"  {os.path.basename(fpath)}: {len(t)} 条")

    all_texts = list(set(all_texts))
    random.shuffle(all_texts)
    logger.info(f"\n📊 续训总计: {len(all_texts)} 条")
    return all_texts


# ==================== 数据集（复用主脚本） ====================

class NeuroGlyphDataset(Dataset):
    def __init__(self, texts, char_to_idx, embedding_matrix, seq_len=256):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.embedding_matrix = torch.FloatTensor(embedding_matrix)

        all_tokens = []
        for text in texts:
            for ch in text:
                if ch in char_to_idx:
                    all_tokens.append(char_to_idx[ch])

        self.tokens = torch.LongTensor(all_tokens)
        stride = max(1, seq_len // 2)
        self.starts = list(range(0, max(1, len(self.tokens) - seq_len - 1), stride))
        if len(self.starts) == 0 and len(self.tokens) > seq_len + 1:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        token_seq = self.tokens[start:end]
        target_seq = self.tokens[start + 1:end + 1]

        if len(token_seq) < self.seq_len:
            token_seq = F.pad(token_seq, (0, self.seq_len - len(token_seq)), value=0)
            target_seq = F.pad(target_seq, (0, self.seq_len - len(target_seq)), value=0)

        feat_seq = self.embedding_matrix[token_seq]
        return feat_seq, target_seq


# ==================== 模型（精简版，与主训练一致） ====================
# 直接导入主脚本的模型类

class StreamCNN(nn.Module):
    def __init__(self, in_dim, ch1, ch2, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, ch1, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.GELU(),
            nn.Conv1d(ch1, ch2, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.GELU(),
        )

    def forward(self, x, T):
        out = self.conv(x)
        return out[:, :, :T]


class NeuroGlyph34XL(nn.Module):
    def __init__(self):
        super().__init__()
        # 5路CNN
        cnn_slices = [(0, 5), (5, 9), (10, 13), (16, 26), (26, 32)]
        cnn_channels = [64, 64, 64, 64, 64]
        cnn_kernels = [2, 2, 2, 3, 3]
        self.cnn_streams = nn.ModuleList()
        for (s, e), ch2, ks in zip(cnn_slices, cnn_channels, cnn_kernels):
            self.cnn_streams.append(StreamCNN(e - s, (e - s) * 2, ch2, ks))
        self.cnn_slices = cnn_slices

        # 5路Embedding
        self.emb_total_strokes = nn.Embedding(37, 16)
        self.emb_radical_extra = nn.Embedding(27, 16)
        self.emb_same_group = nn.Embedding(57, 16)
        self.emb_structure = nn.Embedding(15, 16)
        self.emb_symbol = nn.Embedding(256, 32)

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(416, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, 384))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384, nhead=12, dim_feedforward=1536,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=10)
        self.ln_final = nn.LayerNorm(384)
        self.lm_head = nn.Linear(384, 7987, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T)
            feat = feat.permute(0, 2, 1)
            cnn_outputs.append(feat)

        emb_outputs = [
            self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)),
        ]

        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged)
        fused = fused + self.pos_encoding[:, :T, :]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)


# ==================== 续训 ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    logger.info(f"设备: {device}")

    # 加载词表和嵌入
    embedding_matrix = np.load(embedding_file)
    with open(vocab_file, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    logger.info(f"词表: {len(char_to_idx)}")

    # 加载主训练模型
    logger.info(f"加载主模型: {MODEL_CKPT}")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = NeuroGlyph34XL().to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"  原 step={ckpt.get('step', 'N/A')}, best_loss={ckpt.get('best_loss', 'N/A'):.4f}")

    total, _ = sum(p.numel() for p in model.parameters()), 0
    logger.info(f"模型参数: {total:,}")

    # 加载续训数据
    texts = load_all_continue_data()
    if len(texts) < 100:
        logger.error("续训数据不足!")
        return

    dataset = NeuroGlyphDataset(texts, char_to_idx, embedding_matrix, MAX_SEQ_LEN)
    loader = DataLoader(
        dataset, batch_size=MICRO_BATCH, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    logger.info(f"数据集: {len(dataset):,} 序列, {len(loader):,} batch")

    # 优化器 - 低学习率
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = EPOCHS * len(loader) * GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=LEARNING_RATE * 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"续训开始: {EPOCHS} epochs, LR={LEARNING_RATE}, ~{total_steps} 步")
    logger.info(f"{'='*60}")

    step = 0
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        train_iter = iter(loader)

        while True:
            optimizer.zero_grad()
            micro_losses = []

            for _ in range(GRADIENT_ACCUMULATION):
                try:
                    feat_seq, target_seq = next(train_iter)
                except StopIteration:
                    break

                feat_seq = feat_seq.to(device)
                target_seq = target_seq.to(device)

                if scaler:
                    with autocast():
                        logits = model(feat_seq)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                        loss = loss / GRADIENT_ACCUMULATION
                    scaler.scale(loss).backward()
                else:
                    logits = model(feat_seq)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                    loss = loss / GRADIENT_ACCUMULATION
                    loss.backward()

                micro_losses.append(loss.item() * GRADIENT_ACCUMULATION)

            if not micro_losses:
                break

            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            if step < warmup_steps:
                lr_scale = float(step + 1) / max(1, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * (0.1 + 0.9 * lr_scale)
            else:
                scheduler.step()

            step += 1
            step_loss = sum(micro_losses) / len(micro_losses)
            epoch_loss += step_loss
            batch_count += 1

            if step % 100 == 0:
                logger.info(f"  Step {step:>6} | Loss: {step_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if step % SAVE_INTERVAL == 0:
                save_ckpt = os.path.join(checkpoint_dir, f"continue_step_{step}.pt")
                tmp = save_ckpt + ".tmp"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": step_loss,
                }, tmp)
                os.replace(tmp, save_ckpt)

        avg_loss = epoch_loss / max(batch_count, 1)
        logger.info(f"  Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(checkpoint_dir, "continue_best.pt")
            torch.save({
                "step": step,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": best_loss,
            }, best_ckpt)

    # 最终保存
    final_ckpt = os.path.join(checkpoint_dir, "continue_final.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "loss": avg_loss,
    }, final_ckpt)

    logger.info(f"\n✅ 续训完成! 最终Loss: {avg_loss:.4f}, 最佳Loss: {best_loss:.4f}")
    logger.info(f"📁 模型: {final_ckpt}")


if __name__ == "__main__":
    main()