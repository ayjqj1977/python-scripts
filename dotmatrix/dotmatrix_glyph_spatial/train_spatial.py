# -*- coding: utf-8 -*-
"""
train_spatial.py — 空间感知点阵编码模型云端训练脚本
========================================================
基于 train_pointnet.py，改用 PointMatrixSpatialModel。

训练超参数 (与 PointMatrix 对齐):
  - LR: 1e-4, warmup + 余弦衰减
  - batch_size: 16 × 4 = 64 (梯度累积)
  - max_seq_len: 512
  - MAX_STEPS: 50,000
  - checkpoint: 每 5000 步保存, 保留最近 3 个
  - 语料: corpus_v10_article.txt

用法:
    python train_spatial.py                          # 从头训练
    python train_spatial.py --resume checkpoint      # 断点续训
    python train_spatial.py --steps 100000 --lr 2e-4 # 自定义
"""
import argparse
import json
import math
import os
import sys
import time
import gc
from array import array
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hzk32_reader import HZK32Reader
from model_spatial import PointMatrixSpatialModel


def get_default_config():
    return {
        "corpus_path": "corpus_v10_article.txt",
        "char_to_idx_path": "char_to_idx_full.json",
        "hzk32_path": "HZK32",
        "output_dir": "checkpoints_spatial",

        "vocab_size": 6726,
        "d_glyph": 1024,
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 512,
        "batch_size": 16,
        "grad_accum_steps": 4,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 2000,
        "max_steps": 50000,
        "log_interval": 10,
        "eval_interval": 5000,
        "save_interval": 5000,
        "max_checkpoints": 3,
    }


class CorpusDataset(Dataset):
    CHUNK = 10_000_000

    def __init__(self, corpus_path: str, char_to_idx: dict, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.char_to_idx = char_to_idx
        self.unk_id = char_to_idx.get("<UNK>", 0)
        self.memmap_path = corpus_path + ".ids.mmap"

        if not os.path.exists(self.memmap_path):
            print(f"[Dataset] 加载语料: {corpus_path}")
            with open(corpus_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            self.total_chars = len(raw_text)
            print(f"[Dataset] 总字符数: {self.total_chars:,}")

            t0 = time.time()
            fp = np.memmap(self.memmap_path, dtype=np.int32, mode="w+",
                           shape=(self.total_chars,))
            for i in range(0, self.total_chars, self.CHUNK):
                end_i = min(i + self.CHUNK, self.total_chars)
                arr = array("i", (char_to_idx.get(ch, self.unk_id)
                                  for ch in raw_text[i:end_i]))
                fp[i:end_i] = np.frombuffer(arr, dtype=np.int32)
                pct = end_i / self.total_chars * 100
                print(f"  tokenize {pct:.0f}% ({end_i:,}/{self.total_chars:,})")
            fp.flush()
            del fp, raw_text, arr
            gc.collect()
            print(f"[Dataset] 预计算完成, 耗时 {time.time()-t0:.0f}s, 写入 {self.memmap_path}")
        else:
            print(f"[Dataset] 复用已有 token ids: {self.memmap_path}")
            fp = np.memmap(self.memmap_path, dtype=np.int32, mode="r")
            self.total_chars = fp.shape[0]
            del fp

        self.token_ids = np.memmap(self.memmap_path, dtype=np.int32, mode="r",
                                   shape=(self.total_chars,))
        self.num_samples = max(self.total_chars // (max_seq_len + 1), 1)
        print(f"[Dataset] 样本数: {self.num_samples:,}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.max_seq_len + 1)
        end = min(start + self.max_seq_len + 1, self.total_chars)
        ids = self.token_ids[start:end].tolist()
        if len(ids) < self.max_seq_len + 1:
            ids.extend([0] * (self.max_seq_len + 1 - len(ids)))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, loss, output_dir, max_keep=3):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ckpt_step{step:06d}.pt")
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, tmp_path)
    os.replace(tmp_path, path)
    ckpts = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("ckpt_step") and f.endswith(".pt") and not f.endswith(".tmp")
    ])
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(os.path.join(output_dir, old))
    return path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"], ckpt["loss"]


def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("ckpt_step") and f.endswith(".pt") and not f.endswith(".tmp")
    ])
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


def build_glyph_matrix(hzk32_path: str, char_to_idx: dict, vocab_size: int) -> np.ndarray:
    reader = HZK32Reader(hzk32_path)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    matrix = np.zeros((vocab_size, 1024), dtype=np.float32)
    loaded, no_glyph, failed = 0, 0, 0
    for idx in range(vocab_size):
        ch = idx_to_char.get(idx, None)
        if ch is None or ch == "<UNK>":
            continue
        try:
            matrix[idx] = reader.get_glyph_vector(ch).astype(np.float32)
            loaded += 1
        except ValueError:
            no_glyph += 1
        except IOError:
            failed += 1
    print(f"[Glyph] 加载完成: 成功={loaded}, 无点阵={no_glyph}, 失败={failed}")
    return matrix


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    model.train()
    torch.cuda.empty_cache()
    return avg_loss, ppl


def train(args):
    config = get_default_config()
    for key in ["lr", "max_steps", "batch_size", "grad_accum_steps",
                "max_seq_len", "output_dir", "corpus_path"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    print(f"[Vocab] 加载: {config['char_to_idx_path']}")
    with open(config["char_to_idx_path"], "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    print(f"[Vocab] 词表大小: {len(char_to_idx)}")

    print("[Model] 初始化 PointMatrixSpatialModel...")
    model = PointMatrixSpatialModel(
        vocab_size=config["vocab_size"],
        d_glyph=config["d_glyph"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )

    print("[Glyph] 从 HZK32 提取点阵...")
    glyph_matrix = build_glyph_matrix(
        config["hzk32_path"], char_to_idx, config["vocab_size"]
    )
    model.load_glyph_matrix(glyph_matrix)
    del glyph_matrix
    gc.collect()

    total, trainable = model.param_count()
    print(f"[Model] 参数: total={total:,} trainable={trainable:,}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_lr_scheduler(optimizer, config["warmup_steps"], config["max_steps"])

    start_step = 0
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = find_latest_checkpoint(config["output_dir"])
    if ckpt_path:
        print(f"[Resume] 从 {ckpt_path} 恢复...")
        start_step, _ = load_checkpoint(model, optimizer, scheduler, ckpt_path)
        print(f"[Resume] 从 step={start_step} 继续")

    print(f"[Data] 加载语料: {config['corpus_path']}")
    dataset = CorpusDataset(config["corpus_path"], char_to_idx, config["max_seq_len"])
    train_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=0, drop_last=True,
    )

    val_len = max(len(dataset) // 20, 1)
    val_indices = list(range(len(dataset) - val_len, len(dataset)))
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    gc.collect()

    log_path = os.path.join(config["output_dir"], "training_log_spatial.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    logs = []

    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    print(f"\n{'='*60}")
    print(f"[Train] max_steps={config['max_steps']} lr={config['lr']} "
          f"batch={config['batch_size']}×{config['grad_accum_steps']}")
    print(f"{'='*60}\n")

    sample_x, sample_y = next(iter(train_loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)
    with torch.no_grad():
        _ = model(sample_x)
    del sample_x, sample_y
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("[Warmup] CUDA JIT 完成\n")
    dataloader_iter = iter(train_loader)

    train_log_fp = open(os.path.join(config["output_dir"], "train.log"), "a", encoding="utf-8")

    def prlog(msg):
        print(msg)
        train_log_fp.write(msg + "\n")
        train_log_fp.flush()

    step = start_step
    accum_step = 0
    total_loss_accum = 0.0
    tokens_accum = 0
    t_start = time.time()

    while step < config["max_steps"]:
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            x, y = next(dataloader_iter)

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = loss / config["grad_accum_steps"]
        loss.backward()

        total_loss_accum += loss.item() * config["grad_accum_steps"] * y.numel()
        tokens_accum += y.numel()
        accum_step += 1

        if accum_step == config["grad_accum_steps"]:
            accum_step = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            del x, y, logits, loss

            if step % config["eval_interval"] == 0:
                avg_loss_val = total_loss_accum / max(tokens_accum, 1)
                val_loss, val_ppl = evaluate(model, val_loader, device)
                rec = {
                    "step": step,
                    "train_loss": round(avg_loss_val, 4),
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                    "lr": scheduler.get_last_lr()[0],
                }
                logs.append(rec)
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                prlog(f"  [Eval] step={step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

            if step % config["log_interval"] == 0:
                avg_loss = total_loss_accum / max(tokens_accum, 1)
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                prlog(f"step {step:6d}/{config['max_steps']} | "
                      f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                      f"lr {lr_now:.2e} | elapsed {elapsed:.0f}s")
                total_loss_accum = 0.0
                tokens_accum = 0

            if step % config["save_interval"] == 0:
                ckpt_path = save_checkpoint(
                    model, optimizer, scheduler, step,
                    total_loss_accum / max(tokens_accum, 1),
                    config["output_dir"], config["max_checkpoints"],
                )
                prlog(f"  [Save] {ckpt_path}")
                torch.cuda.empty_cache()

    train_log_fp.close()

    final_ckpt = save_checkpoint(
        model, optimizer, scheduler, config["max_steps"],
        total_loss_accum / max(tokens_accum, 1),
        config["output_dir"], config["max_checkpoints"],
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[Done] {config['max_steps']} steps, 耗时 {elapsed_total:.0f}s")
    print(f"[Done] Checkpoint: {final_ckpt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="空间感知点阵编码模型训练")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.steps:
        args.max_steps = args.steps
    else:
        args.max_steps = None
    if args.corpus:
        args.corpus_path = args.corpus
    else:
        args.corpus_path = None
    if args.output:
        args.output_dir = args.output
    else:
        args.output_dir = None

    train(args)
