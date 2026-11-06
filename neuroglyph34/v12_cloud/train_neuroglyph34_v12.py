# -*- coding: utf-8 -*-
"""
train_neuroglyph34_v12.py — V12 训练脚本（扩参版）
=====================================================
架构: 34维字形(frozen) + 64维语义(trainable) → Transformer(12层, 768维, 12头) → vocab_size

V12 改进 (相对于 V11):
  - d_model: 512 → 768
  - d_ff: 2048 → 3072
  - n_heads: 8 → 12
  - 参数量: 42.6M → ~110M
  - MAX_STEPS: 100000 → 200000
"""
import json
import os
import random
import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_v10 import NeuroGlyph34V10

# ──────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus_v10_article.txt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx.json")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embedding_matrix.npy")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v12")

# ──────────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────────
VOCAB_SIZE = 7988
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 768       # ← 512 → 768
N_LAYERS = 12
N_HEADS = 12        # ← 8 → 12
D_FF = 3072         # ← 2048 → 3072
DROPOUT = 0.1
MAX_SEQ_LEN = 256

WINDOW_SIZE = 256
WINDOW_STRIDE = 128

BATCH_SIZE = 16     # ← 32 → 16 (模型大了，减 batch 防 OOM)
GRAD_ACCUM_STEPS = 8  # ← 4 → 8 (保持有效 batch=128)
LEARNING_RATE = 1e-4
WARMUP_STEPS = 2000   # ← 1000 → 2000 (大模型需要更长 warmup)
MAX_STEPS = 200000    # ← 100000 → 200000

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────
# 窗口切片数据集
# ──────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, articles: List[List[int]], window_size: int = 256, stride: int = 128):
        self.windows = []
        for tokens in articles:
            if len(tokens) < 2:
                continue
            for start in range(0, max(1, len(tokens) - 1), stride):
                end = min(start + window_size + 1, len(tokens))
                window = tokens[start:end]
                if len(window) >= 4:
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in batch])
    max_len = lengths.max().item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, : len(s)] = s
    input_ids = padded[:, :-1]
    target_ids = padded[:, 1:]
    loss_mask = torch.zeros_like(target_ids, dtype=torch.float)
    for i, l in enumerate(lengths):
        loss_mask[i, : l - 1] = 1.0
    return input_ids, target_ids, loss_mask


def load_articles(corpus_path: str, char_to_idx: dict) -> List[List[int]]:
    print(f"[train] 加载语料: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        raw_text = f.read()
    articles_raw = raw_text.split("\n\n\n")
    print(f"[train] 原始文章数: {len(articles_raw):,}")
    articles = []
    total_chars = 0
    for art in articles_raw:
        art = art.strip()
        if not art:
            continue
        tokens = [char_to_idx.get(ch, 0) for ch in art]
        total_chars += len(tokens)
        articles.append(tokens)
    print(f"[train] 有效文章: {len(articles):,}, 总字符: {total_chars:,}")
    return articles


# ──────────────────────────────────────────────────
# 学习率调度
# ──────────────────────────────────────────────────
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = lr * (base_lr / self.base_lrs[0])

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lrs[0] * (self.current_step / max(1, self.warmup_steps))
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────────
# 训练主循环
# ──────────────────────────────────────────────────
def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    articles = load_articles(CORPUS_PATH, char_to_idx)
    random.shuffle(articles)

    print(f"[train] 文章数: {len(articles):,}")

    train_set = WindowDataset(articles, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"[train] 训练窗口: {len(train_set):,}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    model = NeuroGlyph34V10(
        vocab_size=VOCAB_SIZE, d_glyph=D_GLYPH, d_semantic=D_SEMANTIC,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        d_ff=D_FF, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    )
    embedding_matrix = np.load(EMBEDDING_PATH)
    aligned_matrix = np.zeros((VOCAB_SIZE, embedding_matrix.shape[1]), dtype=np.float32)
    aligned_matrix[1:] = embedding_matrix
    model.load_glyph_matrix(aligned_matrix)
    model.to(DEVICE)

    total, trainable = model.param_count()
    print(f"[train] 参数量: {total:,} / 可训练 {trainable:,} (~{total/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS)

    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
    use_amp = scaler is not None

    model.train()
    global_step = 0
    total_loss_accum = 0.0
    step_loss_accum = 0.0
    start_time = time.time()

    eff_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"\n{'='*60}")
    print(f"V12 扩参训练 | 设备: {DEVICE} | AMP: {use_amp}")
    print(f"D_MODEL={D_MODEL} N_HEADS={N_HEADS} D_FF={D_FF}")
    print(f"WINDOW={WINDOW_SIZE} STRIDE={WINDOW_STRIDE} MAX_SEQ={MAX_SEQ_LEN}")
    print(f"BATCH={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM_STEPS} = EFF_BATCH={eff_batch}")
    print(f"样本: {len(train_set):,} train")
    print(f"{'='*60}\n")

    train_iter = iter(train_loader)

    while global_step < MAX_STEPS:
        try:
            input_ids, target_ids, loss_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids, loss_mask = next(train_iter)

        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        loss_mask = loss_mask.to(DEVICE)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                target_flat = target_ids.reshape(-1)
                mask_flat = loss_mask.reshape(-1)
                loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_flat = target_ids.reshape(-1)
            mask_flat = loss_mask.reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            (loss / GRAD_ACCUM_STEPS).backward()

        step_loss_accum += loss.item()

        if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss_accum += step_loss_accum
            step_loss_accum = 0.0

        global_step += 1

        if global_step % 50 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_lr()
            avg_loss = total_loss_accum / max(1, global_step // GRAD_ACCUM_STEPS)
            print(f"[step {global_step:>6d}/{MAX_STEPS}] loss={avg_loss:.4f}  lr={lr:.2e}  elapsed={elapsed:.0f}s")

    # 保存最终
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "latest.pt"))
    print(f"\n训练完成!")
    print(f"checkpoint 目录: {CKPT_DIR}")


if __name__ == "__main__":
    main()
