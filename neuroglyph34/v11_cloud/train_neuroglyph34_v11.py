
# -*- coding: utf-8 -*-
"""
train_neuroglyph34_v11.py — V11 训练脚本（窗口切片版）
=====================================================
架构: 34维字形(frozen) + 64维语义(trainable) → Transformer(12层, 512维, 8头) → vocab_size

V11 改进:
  - 窗口切片: 文章切成 256字窗口，128字步长 (50%重叠)
  - 样本量: 从80万篇 → ~550万个窗口
  - MAX_SEQ_LEN=256, BATCH_SIZE=32, GRAD_ACCUM=4 (有效batch=128)
  - 目标: PPL < 10
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
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_v11")

# ──────────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────────
VOCAB_SIZE = 7988
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 256

WINDOW_SIZE = 256
WINDOW_STRIDE = 128

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4  # 有效 batch = 128
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
            # 滑动窗口
            for start in range(0, max(1, len(tokens) - 1), stride):
                end = min(start + window_size + 1, len(tokens))
                window = tokens[start:end]
                if len(window) >= 4:  # 最少4个token
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
# 字形准确率 (V11 新增)
# ──────────────────────────────────────────────────
@torch.no_grad()
def glyph_accuracy(model: nn.Module, glyph_matrix: torch.Tensor, dataloader: DataLoader) -> float:
    """计算字形级预测准确率: 预测token的字形向量与目标token的字形向量 cosine similarity"""
    model.eval()
    total_correct = 0
    total_tokens = 0
    threshold = 0.9  # cosine > 0.9 视为字形匹配

    for input_ids, target_ids, loss_mask in dataloader:
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        loss_mask = loss_mask.to(DEVICE)

        logits = model(input_ids)  # [B, L, V]
        pred_ids = logits.argmax(dim=-1)  # [B, L]

        # 获取预测和目标 token 的字形向量
        pred_glyph = glyph_matrix[pred_ids]    # [B, L, 34]
        target_glyph = glyph_matrix[target_ids] # [B, L, 34]

        # cosine similarity
        pred_norm = F.normalize(pred_glyph, dim=-1)
        target_norm = F.normalize(target_glyph, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B, L]

        correct = (cos_sim > threshold) & loss_mask.bool()
        total_correct += correct.sum().item()
        total_tokens += loss_mask.sum().item()

    model.train()
    return total_correct / max(1, total_tokens)


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
# 形近字判别 (V11 新增)
# ──────────────────────────────────────────────────
@torch.no_grad()
def similar_glyph_test(model: nn.Module, glyph_matrix: torch.Tensor, idx_to_char: dict):
    """测试模型对形近字的区分能力"""
    # 常见形近字对
    pairs = [
        ("未", "末"), ("已", "己"), ("干", "千"), ("土", "士"),
        ("人", "入"), ("大", "太"), ("本", "木"), ("困", "因"),
        ("侯", "候"), ("辨", "辩"), ("徒", "徙"), ("浙", "淅"),
        ("爪", "瓜"), ("券", "卷"), ("灸", "炙"), ("栗", "粟"),
        ("日", "曰"), ("刀", "力"), ("建", "健"), ("复", "覆"),
    ]

    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    glyph_norm = F.normalize(glyph_matrix, dim=-1)

    results = []
    for a, b in pairs:
        if a not in char_to_idx or b not in char_to_idx:
            continue
        ia, ib = char_to_idx[a], char_to_idx[b]
        cos = (glyph_norm[ia] * glyph_norm[ib]).sum().item()
        label = "✓ 易区分" if cos < 0.7 else ("△ 一般" if cos < 0.85 else "✗ 难区分")
        results.append((a, b, cos, label))

    results.sort(key=lambda x: x[2], reverse=True)
    print("\n  形近字判别力测试 (cosine越低=区分越好):")
    easy = sum(1 for _, _, c, _ in results if c < 0.7)
    mid = sum(1 for _, _, c, _ in results if 0.7 <= c < 0.85)
    hard = sum(1 for _, _, c, _ in results if c >= 0.85)
    for a, b, cos, label in results[:15]:
        print(f"  {a}-{b}: {cos:.4f} {label}")
    print(f"  汇总: 易区分({easy}), 一般({mid}), 难区分({hard})")


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

    # ── 构建模型 ──
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

    glyph_matrix = model.glyph_matrix.to(DEVICE)

    total, trainable = model.param_count()
    print(f"[train] 参数量: {total:,} / 可训练 {trainable:,} (~{total/1e6:.1f}M)")

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
    print(f"V11 窗口切片训练 | 设备: {DEVICE} | AMP: {use_amp}")
    print(f"WINDOW={WINDOW_SIZE} STRIDE={WINDOW_STRIDE} MAX_SEQ={MAX_SEQ_LEN}")
    print(f"BATCH={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM_STEPS} = EFF_BATCH={BATCH_SIZE*GRAD_ACCUM_STEPS}")
    print(f"样本: {len(train_set):,} train, {len(val_set):,} val")
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

                # 最佳时跑形近字测试
                similar_glyph_test(model, glyph_matrix, idx_to_char)
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
