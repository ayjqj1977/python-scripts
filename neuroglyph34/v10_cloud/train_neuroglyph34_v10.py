# -*- coding: utf-8 -*-
"""
train_neuroglyph34_v10.py — V10 训练脚本
========================================
架构: 34维字形 + 64维语义 → Transformer(12层, 512维, 8头) → vocab_size
参数量: ~4275万

训练策略:
  - 文章级输入: corpus_v10_article.txt 每篇文章为一个 sample
  - 全局 shuffle: 所有文章随机打乱
  - 最大序列长度: 512 (超长文章截断)
  - batch padding + causal mask

Checkpoint: checkpoints_v10/ 子目录
"""
import json
import os
import random
import math
import time
from typing import List, Tuple, Optional

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
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v10")

# ──────────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────────
VOCAB_SIZE = 7988  # 含 <UNK>
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 512

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4  # 有效 batch = 64
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
MAX_STEPS = 50000
EVAL_STEPS = 500
SAVE_STEPS = 2000

VAL_SPLIT = 0.05
EARLY_STOP_PATIENCE = 8  # 8 次评估无改善即停止 (8 × 500 = 4000 步)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────
class ArticleDataset(Dataset):
    """文章级数据集: 每篇文章为一个 sample"""

    def __init__(self, articles: List[List[int]], max_seq_len: int = MAX_SEQ_LEN):
        self.articles = articles
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        tokens = self.articles[idx]
        # 截断超长文章
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]
        if len(tokens) < 2:
            # 退化为最小序列
            tokens = [0, 1]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """padding + 构建输入/目标"""
    lengths = torch.tensor([len(s) for s in batch])
    max_len = lengths.max().item()

    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, : len(s)] = s

    # input: tokens[:-1], target: tokens[1:]
    input_ids = padded[:, :-1]
    target_ids = padded[:, 1:]

    # mask for loss: 只计算有效位置
    loss_mask = torch.zeros_like(target_ids, dtype=torch.float)
    for i, l in enumerate(lengths):
        loss_mask[i, : l - 1] = 1.0

    return input_ids, target_ids, loss_mask


def load_articles(corpus_path: str, char_to_idx: dict) -> List[List[int]]:
    """加载文章级语料，转换为 token ID 序列"""
    print(f"[train] 加载语料: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        raw_text = f.read()

    # 按双空行分割文章
    articles_raw = raw_text.split("\n\n\n")
    print(f"[train] 原始文本分段: {len(articles_raw)}")

    articles = []
    skip_count = 0
    unk_count = 0

    for art in articles_raw:
        art = art.strip()
        if not art:
            skip_count += 1
            continue

        tokens = []
        for ch in art:
            if ch in char_to_idx:
                tokens.append(char_to_idx[ch])
            else:
                tokens.append(0)  # UNK → index 0
                unk_count += 1

        if len(tokens) >= 2:
            articles.append(tokens)
        else:
            skip_count += 1

    print(f"[train] 有效文章: {len(articles)}, 跳过: {skip_count}")
    print(f"[train] 总 token 数: {sum(len(a) for a in articles):,}")
    if unk_count > 0:
        print(f"[train] UNK 字符数: {unk_count:,} ({100*unk_count/sum(len(a) for a in articles):.2f}%)")

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
def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    """计算验证困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids, loss_mask in dataloader:
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
    """训练过程中的文本生成采样"""
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

    # 全局 shuffle
    random.shuffle(articles)

    # 验证集
    n_val = max(1, int(len(articles) * VAL_SPLIT))
    val_articles = articles[:n_val]
    train_articles = articles[n_val:]
    print(f"[train] 训练文章: {len(train_articles):,}, 验证文章: {len(val_articles):,}")

    train_set = ArticleDataset(train_articles)
    val_set = ArticleDataset(val_articles)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    # ── 构建模型 ──
    model = NeuroGlyph34V10(
        vocab_size=VOCAB_SIZE,
        d_glyph=D_GLYPH,
        d_semantic=D_SEMANTIC,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    )
    embedding_matrix = np.load(EMBEDDING_PATH)
    # embedding_matrix.npy 保持 (7987, 34) 不变，映射到 (7988, 34):
    #   row 0 = zero (UNK), rows 1.. = embedding_matrix rows 0..
    aligned_matrix = np.zeros((VOCAB_SIZE, embedding_matrix.shape[1]), dtype=np.float32)
    aligned_matrix[1:] = embedding_matrix
    model.load_glyph_matrix(aligned_matrix)
    model.to(DEVICE)

    total, trainable = model.param_count()
    print(f"[train] 模型参数量: {total:,} / 可训练 {trainable:,} (~{total/1e6:.1f}M)")

    # ── 优化器 & 调度器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    use_amp = scaler is not None

    # ── 训练 ──
    model.train()
    global_step = 0
    best_val_ppl = float("inf")
    best_val_loss = float("inf")
    total_loss_accum = 0.0
    step_loss_accum = 0.0
    start_time = time.time()

    # 训练日志 + early stopping
    training_log = []
    no_improve_count = 0

    print(f"\n{'='*60}")
    print(f"开始训练 | 设备: {DEVICE} | AMP: {use_amp}")
    print(f"batch_size={BATCH_SIZE} × grad_accum={GRAD_ACCUM_STEPS} = eff_batch={BATCH_SIZE*GRAD_ACCUM_STEPS}")
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

        # Forward
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

        # 梯度累积步
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

        # 日志
        if global_step % 50 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_lr()
            avg_loss = total_loss_accum / max(1, global_step // GRAD_ACCUM_STEPS)
            print(f"[step {global_step:>6d}/{MAX_STEPS}] loss={avg_loss:.4f}  lr={lr:.2e}  elapsed={elapsed:.0f}s")

        # 评估
        if global_step % EVAL_STEPS == 0:
            # 计算当前训练 loss
            recent_steps = max(1, GRAD_ACCUM_STEPS)
            current_train_loss = step_loss_accum / recent_steps if step_loss_accum > 0 else total_loss_accum / max(1, global_step // GRAD_ACCUM_STEPS)

            val_ppl = evaluate(model, val_loader)
            val_loss = math.log(val_ppl)
            lr = scheduler.get_lr()

            # 记录训练日志
            training_log.append({
                "step": global_step,
                "train_loss": round(current_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_ppl": round(val_ppl, 4),
                "lr": round(lr, 8),
            })

            print(f"[eval  step {global_step:>6d}] train_loss={current_train_loss:.4f}  val_ppl={val_ppl:.2f}  lr={lr:.2e}")

            # 生成采样
            prompts = ["今天天气", "人工智能"]
            for p in prompts:
                sample = sample_generation(model, idx_to_char, p, length=40)
                print(f"  [{p}] → {sample}")

            # 保存最佳
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save(
                    {
                        "step": global_step,
                        "epoch": global_step // len(train_loader),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_ppl": val_ppl,
                    },
                    os.path.join(CKPT_DIR, "best_model.pt"),
                )
                print(f"  *** 最佳模型已保存 (ppl={val_ppl:.2f}) ***")
            else:
                no_improve_count += 1
                print(f"  val_ppl 无改善 ({no_improve_count}/{EARLY_STOP_PATIENCE})")

            # early stopping
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] 连续 {EARLY_STOP_PATIENCE} 次评估无改善，停止训练于 step {global_step}")
                print(f"  最佳 val_ppl={best_val_ppl:.2f}  val_loss={best_val_loss:.4f}")
                break

        # 定期保存 checkpoint
        if global_step % SAVE_STEPS == 0:
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ppl": best_val_ppl,
                },
                os.path.join(CKPT_DIR, f"checkpoint_step{global_step}.pt"),
            )
            print(f"  checkpoint 已保存: step {global_step}")

    # ── 训练结束 ──
    # 保存训练日志
    log_path = os.path.join(CKPT_DIR, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"\n训练日志已保存: {log_path}")

    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ppl": best_val_ppl,
        },
        os.path.join(CKPT_DIR, "latest.pt"),
    )
    print(f"训练完成! 最佳 val_ppl={best_val_ppl:.2f}")
    print(f"checkpoint 目录: {CKPT_DIR}")


if __name__ == "__main__":
    main()
