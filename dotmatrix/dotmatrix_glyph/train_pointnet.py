# -*- coding: utf-8 -*-
"""
train_pointnet.py — 点阵字形编码模型云端训练脚本 (全词表版)
================================================================
基于 V10 训练逻辑，改用 PointMatrixModel + HZK32 点阵。
词表: char_to_idx_full.json (6726 条目: 6622 GB2312 + 103 补回标点/数字/字母)

训练超参数 (与 V10 对齐):
  - LR: 1e-4, warmup + 余弦衰减
  - batch_size: 16 × 4 = 64 (4 GPU 梯度累积)
  - max_seq_len: 512
  - MAX_STEPS: 50,000
  - checkpoint: 每 5000 步保存, 保留最近 3 个
  - 语料: corpus_v10_article.txt (~2GB, 云端路径)

用法:
    python train_pointnet.py                          # 从头训练
    python train_pointnet.py --resume checkpoint      # 断点续训
    python train_pointnet.py --steps 100000 --lr 2e-4 # 自定义参数
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

# 输出缓冲修复：nohup 后台运行时强制行缓冲，确保日志实时写入
sys.stdout.reconfigure(line_buffering=True)

# 添加项目目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hzk32_reader import HZK32Reader
from model_pointnet import PointMatrixModel


# ============================================================
# 配置
# ============================================================

def get_default_config():
    return {
        # 路径
        "corpus_path": "corpus_v10_article.txt",
        "char_to_idx_path": "char_to_idx_full.json",
        "hzk32_path": "HZK32",
        "output_dir": "checkpoints_pointnet",

        # 超参数
        "vocab_size": 6726,
        "d_glyph": 1024,
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 512,
        "batch_size": 16,
        "grad_accum_steps": 4,       # 等效 batch_size = 16×4 = 64
        "lr": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 2000,
        "max_steps": 50000,
        "log_interval": 10,
        "eval_interval": 5000,
        "save_interval": 5000,
        "max_checkpoints": 3,
    }


# ============================================================
# 语料数据集
# ============================================================

class CorpusDataset(Dataset):
    """大语料数据集：首次运行将语料预 tokenize 为 memmap 文件，后续直接 numpy 切片。

    预 tokenize 阶段：用 array('i') 分块写入 memmap，内存增量仅 ~40MB/chunk。
    __getitem__ 不再做 dict 查找，直接 memmap 切片 → torch.tensor。
    """

    CHUNK = 10_000_000  # 预 tokenize 时每块字符数（≈40MB memory）

    def __init__(self, corpus_path: str, char_to_idx: dict, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.char_to_idx = char_to_idx
        self.unk_id = char_to_idx.get("<UNK>", 0)
        self.memmap_path = corpus_path + ".ids.mmap"

        if not os.path.exists(self.memmap_path):
            # ── 首次运行：读文本 → 分块 tokenize → 写入 memmap ──
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
            elapsed = time.time() - t0
            print(f"[Dataset] 预计算完成, 耗时 {elapsed:.0f}s, 写入 {self.memmap_path}")
        else:
            print(f"[Dataset] 复用已有 token ids: {self.memmap_path}")
            fp = np.memmap(self.memmap_path, dtype=np.int32, mode="r")
            self.total_chars = fp.shape[0]
            del fp

        # 打开只读 memmap — getitem 直接切片，零开销
        self.token_ids = np.memmap(self.memmap_path, dtype=np.int32, mode="r",
                                   shape=(self.total_chars,))
        self.num_samples = max(self.total_chars // (max_seq_len + 1), 1)
        print(f"[Dataset] 样本数: {self.num_samples:,}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.max_seq_len + 1)
        end = min(start + self.max_seq_len + 1, self.total_chars)
        ids = self.token_ids[start:end].tolist()  # memmap 切片 (μs 级)

        if len(ids) < self.max_seq_len + 1:
            ids.extend([0] * (self.max_seq_len + 1 - len(ids)))

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


# ============================================================
# LR Scheduler: warmup + cosine decay
# ============================================================

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Checkpoint 管理
# ============================================================

def save_checkpoint(model, optimizer, scheduler, step, loss, output_dir, max_keep=3):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ckpt_step{step:06d}.pt")
    # 原子写入防止断电损坏
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, tmp_path)
    os.replace(tmp_path, path)

    # 清理旧 checkpoint
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
    """查找最新的 checkpoint"""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("ckpt_step") and f.endswith(".pt") and not f.endswith(".tmp")
    ])
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


# ============================================================
# 构建 glyph_matrix
# ============================================================

def build_glyph_matrix(hzk32_path: str, char_to_idx: dict, vocab_size: int) -> np.ndarray:
    """从 HZK32 提取每个字符的 1024 维点阵。
    GB2312 汉字 (idx 1~6622) 有 HZK32 点阵；
    补回的标点/数字/字母 (idx 6623~) 无点阵，保持零向量。
    """
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
            # GB2312 编码失败 → 无点阵（补回字符预期行为）
            no_glyph += 1
        except IOError:
            failed += 1

    print(f"[Glyph] 加载完成: 成功={loaded}, 无点阵={no_glyph}, 失败={failed}, 总计={vocab_size}")
    return matrix


# ============================================================
# 验证
# ============================================================

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
        logits = model(x)  # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    model.train()
    torch.cuda.empty_cache()
    return avg_loss, ppl


# ============================================================
# 训练主循环
# ============================================================

def train(args):
    config = get_default_config()

    # 命令行覆盖
    for key in ["lr", "max_steps", "batch_size", "grad_accum_steps",
                "max_seq_len", "output_dir", "corpus_path"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- 1. 加载词表 ----
    print(f"[Vocab] 加载: {config['char_to_idx_path']}")
    with open(config["char_to_idx_path"], "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    print(f"[Vocab] 词表大小: {len(char_to_idx)}")

    # ---- 2. 构建模型 ----
    print("[Model] 初始化 PointMatrixModel...")
    model = PointMatrixModel(
        vocab_size=config["vocab_size"],
        d_glyph=config["d_glyph"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )

    # ---- 3. 加载点阵矩阵 ----
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

    # ---- 4. 优化器 & 调度器 ----
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_lr_scheduler(optimizer, config["warmup_steps"], config["max_steps"])

    # ---- 5. 断点续训 ----
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = find_latest_checkpoint(config["output_dir"])
    if ckpt_path:
        print(f"[Resume] 从 {ckpt_path} 恢复...")
        start_step, _ = load_checkpoint(model, optimizer, scheduler, ckpt_path)
        print(f"[Resume] 从 step={start_step} 继续")

    # ---- 6. 数据集 ----
    print(f"[Data] 加载语料: {config['corpus_path']}")
    dataset = CorpusDataset(config["corpus_path"], char_to_idx, config["max_seq_len"])
    train_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=0, drop_last=True,
    )

    # 验证数据：取尾部 5%
    val_len = max(len(dataset) // 20, 1)
    val_indices = list(range(len(dataset) - val_len, len(dataset)))
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    gc.collect()

    # ---- 7. 训练日志 ----
    log_path = os.path.join(config["output_dir"], "training_log_pointnet.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    logs = []

    # ---- 8. 训练循环 ----
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    print(f"\n{'='*60}")
    print(f"[Train] 开始训练: max_steps={config['max_steps']}, "
          f"lr={config['lr']}, batch={config['batch_size']}×{config['grad_accum_steps']}={config['batch_size']*config['grad_accum_steps']}")
    print(f"{'='*60}\n")

    # ── CUDA JIT 预热：首次前向传播触发 kernel 编译，避免训练循环首步长时间静默 ──
    sample_x, sample_y = next(iter(train_loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)
    with torch.no_grad():
        _ = model(sample_x)
    del sample_x, sample_y
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("[Warmup] CUDA JIT 编译完成，进入训练循环\n")
    dataloader_iter = iter(train_loader)

    # 直接写日志文件，彻底绕过 stdout 管道缓冲
    train_log_fp = open(os.path.join(config["output_dir"], "train.log"), "a", encoding="utf-8")

    def prlog(msg):
        """双写：stdout + train.log，确保 nohup/nohup -u 都能看到"""
        print(msg)
        train_log_fp.write(msg + "\n")
        train_log_fp.flush()

    step = start_step
    accum_step = 0          # 梯度累积微批次计数
    total_loss_accum = 0.0
    tokens_accum = 0
    t_start = time.time()

    while step < config["max_steps"]:
        # 取下一批数据
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            x, y = next(dataloader_iter)

        x, y = x.to(device), y.to(device)

        # 梯度累积
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

            # 验证（必须在日志之前，因为日志会重置 accumulators）
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

            # 日志
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

            # 保存 checkpoint
            if step % config["save_interval"] == 0:
                ckpt_path = save_checkpoint(
                    model, optimizer, scheduler, step,
                    total_loss_accum / max(tokens_accum, 1),
                    config["output_dir"], config["max_checkpoints"],
                )
                prlog(f"  [Save] {ckpt_path}")
                torch.cuda.empty_cache()

    train_log_fp.close()

    # ---- 9. 最终保存 ----
    final_ckpt = save_checkpoint(
        model, optimizer, scheduler, config["max_steps"],
        total_loss_accum / max(tokens_accum, 1),
        config["output_dir"], config["max_checkpoints"],
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[Done] 训练完成: {config['max_steps']} steps, 耗时 {elapsed_total:.0f}s")
    print(f"[Done] Checkpoint: {final_ckpt}")
    print(f"[Done] Log: {log_path}")
    print(f"{'='*60}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点阵字形编码模型训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练")
    parser.add_argument("--steps", type=int, default=None,
                        help="最大训练步数")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="每 GPU batch size")
    parser.add_argument("--corpus", type=str, default=None,
                        help="语料路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    # 映射到 config key
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
