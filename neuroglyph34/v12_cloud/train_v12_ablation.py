# -*- coding: utf-8 -*-
"""
train_v12_ablation.py — V12 消融实验：96维纯随机嵌入
=====================================================
与 V12 基线 (34维字形 frozen + 64维语义 trainable = 98维) 对比：
  - 完全移除手工字形特征（d_glyph=0）
  - 使用 96维纯随机可训练语义嵌入（d_semantic=96）
  - 总嵌入维度 96 ≈ 原 98，参数量可比

V12 架构: d_model=768, n_heads=12, d_ff=3072, ~110M 参数
训练参数：与 V11 训练脚本完全一致
输出目录: checkpoints_v12_ablation（不覆盖原 V12 结果）
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "corpus_v10_article.txt")
CHAR_TO_IDX_PATH = os.path.join(SCRIPT_DIR, "char_to_idx.json")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_v12_ablation")

# ──────────────────────────────────────────────────
# 超参数 — 架构: V12 (d_model=768) | 嵌入: 消融版 (d_glyph=0, d_semantic=96)
# ──────────────────────────────────────────────────
VOCAB_SIZE = 7988
D_GLYPH = 0       # 消融: 完全移除字形特征
D_SEMANTIC = 96    # 消融: 96维纯随机可训练 (≈ 原34+64=98)
D_MODEL = 768      # V12 架构
N_LAYERS = 12
N_HEADS = 12       # V12 架构
D_FF = 3072        # V12 架构
DROPOUT = 0.1
MAX_SEQ_LEN = 256

# ── 训练超参数 (与 V11 完全一致) ──
WINDOW_SIZE = 256
WINDOW_STRIDE = 128

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
MAX_STEPS = 100000
EVAL_STEPS = 1000
SAVE_STEPS = 5000

VAL_SPLIT = 0.05
EARLY_STOP_PATIENCE = 10
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
    """将文章切成 256字滑动窗口"""

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
# 评估
# ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, max_batches: int = 100) -> float:
    """计算验证困惑度（限制最多 max_batches 批）"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (input_ids, target_ids, loss_mask) in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        loss_mask = loss_mask.to(DEVICE)

        logits = model(input_ids)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = target_ids.reshape(-1)
        mask_flat = loss_mask.reshape(-1)

        loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
        loss = (loss * mask_flat).sum()
        total_loss += loss.item()
        total_tokens += mask_flat.sum().item()

    model.train()
    return math.exp(total_loss / max(1, total_tokens))


# ──────────────────────────────────────────────────
# 生成采样
# ──────────────────────────────────────────────────
@torch.no_grad()
def sample_generation(model: nn.Module, idx_to_char: dict, prompt: str, length: int = 50):
    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    prompt_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    for _ in range(length):
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, -MAX_SEQ_LEN:]
        logits = model(input_ids)
        probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

    return "".join(idx_to_char.get(t, "?") for t in input_ids[0].tolist())


# ──────────────────────────────────────────────────
# 训练主循环
# ──────────────────────────────────────────────────
def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 加载数据 ──
    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    articles = load_articles(CORPUS_PATH, char_to_idx)
    random.shuffle(articles)

    n_val = max(1, int(len(articles) * VAL_SPLIT))
    val_articles = articles[:n_val]
    train_articles = articles[n_val:]
    print(f"[train] 训练文章: {len(train_articles):,}, 验证文章: {len(val_articles):,}")

    # 创建窗口数据集
    train_set = WindowDataset(train_articles, WINDOW_SIZE, WINDOW_STRIDE)
    val_set = WindowDataset(val_articles, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"[train] 训练窗口: {len(train_set):,}, 验证窗口: {len(val_set):,}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    # ── 构建模型 (消融版: d_glyph=0, d_semantic=96) ──
    model = NeuroGlyph34V10(
        vocab_size=VOCAB_SIZE, d_glyph=D_GLYPH, d_semantic=D_SEMANTIC,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        d_ff=D_FF, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    )
    # 消融版不加载 embedding_matrix.npy，纯随机初始化语义嵌入
    model.to(DEVICE)

    total, trainable = model.param_count()
    print(f"[train] 参数量: {total:,} / 可训练 {trainable:,} (~{total/1e6:.1f}M)")
    print(f"[train] 嵌入配置: d_glyph={D_GLYPH}, d_semantic={D_SEMANTIC} (纯随机可训练)")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS)

    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    use_amp = scaler is not None

    model.train()
    global_step = 0
    best_val_ppl = float("inf")
    total_loss_accum = 0.0
    step_loss_accum = 0.0
    start_time = time.time()
    training_log = []
    no_improve_count = 0

    print(f"\n{'='*60}")
    print(f"V12 消融实验 (96维纯随机嵌入) | 设备: {DEVICE} | AMP: {use_amp}")
    print(f"架构: d_model={D_MODEL} n_heads={N_HEADS} d_ff={D_FF}")
    print(f"嵌入: d_glyph={D_GLYPH} (移除) d_semantic={D_SEMANTIC} (纯随机)")
    print(f"WINDOW={WINDOW_SIZE} STRIDE={WINDOW_STRIDE} MAX_SEQ={MAX_SEQ_LEN}")
    print(f"BATCH={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM_STEPS} = EFF_BATCH={BATCH_SIZE*GRAD_ACCUM_STEPS}")
    print(f"样本: {len(train_set):,} train, {len(val_set):,} val")
    print(f"输出: {CKPT_DIR}")
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
            with torch.cuda.amp.autocast():
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

        if global_step % EVAL_STEPS == 0:
            recent_steps = max(1, GRAD_ACCUM_STEPS)
            current_train_loss = step_loss_accum / recent_steps if step_loss_accum > 0 else total_loss_accum / max(1, global_step // GRAD_ACCUM_STEPS)

            val_ppl = evaluate(model, val_loader)
            val_loss = math.log(val_ppl)
            lr = scheduler.get_lr()

            training_log.append({
                "step": global_step,
                "train_loss": round(current_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_ppl": round(val_ppl, 2),
                "lr": lr,
            })

            print(f"[eval step {global_step:>6d}] train_loss={current_train_loss:.4f}  val_ppl={val_ppl:.2f}  lr={lr:.2e}")

            # 生成采样
            for prompt in ["今天天气", "人工智能"]:
                gen = sample_generation(model, idx_to_char, prompt, 50)
                print(f"  [{prompt}] → {gen}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
                print(f"  *** 最佳模型已保存 (ppl={best_val_ppl:.2f}) ***")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(f"\n  Early stopping: {EARLY_STOP_PATIENCE} 次无改善")
                    break

        if global_step % SAVE_STEPS == 0:
            torch.save({
                "step": global_step, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_ppl": best_val_ppl,
            }, os.path.join(CKPT_DIR, f"checkpoint_step{global_step}.pt"))
            print(f"  checkpoint 已保存: step {global_step}")

    # 保存最终
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "latest.pt"))
    with open(os.path.join(CKPT_DIR, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成! 最佳 val_ppl={best_val_ppl:.2f}")
    print(f"checkpoint 目录: {CKPT_DIR}")


if __name__ == "__main__":
    main()
