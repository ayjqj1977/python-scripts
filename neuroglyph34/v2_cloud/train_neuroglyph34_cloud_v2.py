#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGlyph-34 Cloud V2 — 融合18维最佳实践的云端训练脚本
===========================================================
借鉴18维项目:
  1. 5-Stream 并行CNN（在T维度做真正的跨字卷积，不用全局池化）
  2. 5-Stream 并行Embedding
  3. 课程学习（单字→词语→成语→诗词→古文全语料）
  4. 原子保存 + 自动断点续训
  5. 预训练嵌入矩阵作为特征输入

架构: 34维特征 → 5路CNN+5路Emb → 融合384维 → 10层12头Transformer → LM Head
参数量: ~21M（匹配云端GPU规模）
"""
import os, sys, json, math, time, random, logging, argparse, warnings, glob, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    # 路径 — 引用已有的34维项目数据
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = r"D:\Desktop\neuroglyph34_project"
    checkpoint_dir = os.path.join(project_dir, "checkpoints")
    log_dir = os.path.join(project_dir, "logs")

    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"

    # 模型 — 目标 ~21M 参数
    d_input = 34
    vocab_size = 7987
    special_tokens = 4

    # 5路CNN: 四角/五笔/部首/笔顺/拼音
    cnn_channels = [64, 64, 64, 64, 64]   # 每路输出通道数
    cnn_kernels = [2, 2, 2, 3, 3]

    # 5路Embedding: 总笔画/外笔画/同部同画/结构/符号
    embed_dims = [16, 16, 16, 16, 32]

    # 融合层
    total_cnn = sum(cnn_channels)  # 320
    total_emb = sum(embed_dims)    # 96
    fusion_input = total_cnn + total_emb  # 416

    d_fusion = 384       # 融合后维度
    n_layers = 10         # Transformer层数
    n_heads = 12          # 注意力头数
    d_ff = 1536           # FFN隐藏维度
    dropout = 0.15
    max_seq_len = 256

    # 训练
    micro_batch = 16
    gradient_accumulation = 8  # 等效 batch = 128
    lr = 1e-4
    warmup_steps = 2000
    weight_decay = 0.1
    grad_clip = 1.0
    save_interval = 5000
    checkpoint_keep = 3
    use_amp = True

    # 课程学习：每阶段 epochs
    stage_epochs = {
        "单字": 3,
        "词语": 4,
        "成语": 4,
        "诗词": 5,
        "古文全语料": 5,
    }

    # 预估总步数（用于余弦退火调度）
    total_steps = 300_000


config = Config()

# ==================== 数据加载 ====================

def load_json_texts(filepath, key=None, max_items=None):
    """从JSON文件提取文本行"""
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
            elif isinstance(item, dict):
                text = " ".join(str(v) for v in item.values() if isinstance(v, str))
            else:
                continue
            if text and len(text) > 0:
                texts.append(text)
    except Exception as e:
        logger.warning(f"  加载JSON失败 {filepath}: {e}")
    return texts


def load_txt_lines(filepath, max_lines=None):
    """从纯文本文件加载行"""
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
    except Exception as e:
        logger.warning(f"  加载TXT失败 {filepath}: {e}")
    return texts


def load_stage_data(stage_name):
    """按课程学习阶段加载数据"""
    data_dir = config.data_dir
    texts = []

    if stage_name == "单字":
        # 汉字字典
        for name, path, key, limit in [
            ("汉字", "7_训练语料/汉字.json", "zi", 20000),
        ]:
            full = os.path.join(data_dir, path)
            if os.path.exists(full):
                t = load_json_texts(full, key, limit)
                texts.extend(t)
                logger.info(f"  {name}: {len(t)} 条")

    elif stage_name == "词语":
        for name, path, key, limit in [
            ("词语", "7_训练语料/词语.json", "ci", 50000),
            ("新华词语", "7_训练语料/xinhua_word.json", "ci", 50000),
        ]:
            full = os.path.join(data_dir, path)
            if os.path.exists(full):
                t = load_json_texts(full, key, limit)
                texts.extend(t)
                logger.info(f"  {name}: {len(t)} 条")

    elif stage_name == "成语":
        for name, path, key, limit in [
            ("成语", "7_训练语料/成语.json", "chengyu", 30000),
            ("歇后语", "7_训练语料/歇后语.json", None, 10000),
        ]:
            full = os.path.join(data_dir, path)
            if os.path.exists(full):
                t = load_json_texts(full, key, limit)
                texts.extend(t)
                logger.info(f"  {name}: {len(t)} 条")

    elif stage_name == "诗词":
        # 唐诗宋词元曲诗经楚辞
        corpus_dir = os.path.join(data_dir, "corpus")
        poetry_files = [
            ("corpus_training.txt", None, 30000),
            ("corpus/quantangshi_2000.json", "paragraphs", 5000),
            ("corpus/quantangshi_3000.json", "paragraphs", 5000),
            ("corpus/songci_10000.json", "paragraphs", 5000),
            ("corpus/shijing_shijing.json", "content", 5000),
            ("corpus/楚辞_chuci.json", "content", 3000),
            ("corpus/yuanqu_yuanqu.json", "title", 5000),
        ]
        for path, key, limit in poetry_files:
            full = os.path.join(data_dir, path)
            if os.path.exists(full):
                if full.endswith(".txt"):
                    t = load_txt_lines(full, limit)
                else:
                    t = load_json_texts(full, key, limit)
                texts.extend(t)
                logger.info(f"  {os.path.basename(full)}: {len(t)} 条")

    elif stage_name == "古文全语料":
        # 加载全部可用语料
        # 古文
        for name, path, key, limit in [
            ("corpus_ancient.txt", "corpus_ancient.txt", None, 50000),
            ("corpus/四书五经_mengzi.json", "corpus/四书五经_mengzi.json", "content", 5000),
            ("corpus/幽梦影_youmengying.json", "corpus/幽梦影_youmengying.json", "content", 3000),
        ]:
            full = os.path.join(data_dir, path)
            if os.path.exists(full):
                if full.endswith(".txt"):
                    t = load_txt_lines(full, limit)
                else:
                    t = load_json_texts(full, key, limit)
                texts.extend(t)
                logger.info(f"  {name}: {len(t)} 条")

        # 蒙学
        mengxue_dir = os.path.join(data_dir, "corpus")
        for fname in sorted(os.listdir(mengxue_dir)):
            if fname.startswith("蒙学_") and fname.endswith(".json"):
                full = os.path.join(mengxue_dir, fname)
                t = load_json_texts(full, "content", 3000)
                texts.extend(t)
                logger.info(f"  蒙学/{fname}: {len(t)} 条")

        # 混合大语料
        for cf, limit in [
            ("corpus_combined.txt", 100000),
            ("corpus_training.txt", 50000),
        ]:
            full = os.path.join(data_dir, cf)
            if os.path.exists(full):
                t = load_txt_lines(full, limit)
                texts.extend(t)
                logger.info(f"  {cf}: {len(t)} 条")

    # 去重
    texts = list(set(texts))
    random.shuffle(texts)
    logger.info(f"  阶段 [{stage_name}] 合计: {len(texts)} 条")
    return texts


# ==================== 数据集 ====================

class NeuroGlyphDataset(Dataset):
    """将文本转为 34维特征序列 + 下一字符标签"""

    def __init__(self, texts, char_to_idx, embedding_matrix, seq_len=256):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.embedding_matrix = torch.FloatTensor(embedding_matrix)

        # 文本 → token序列
        all_tokens = []
        for text in texts:
            for ch in text:
                if ch in char_to_idx:
                    all_tokens.append(char_to_idx[ch])

        self.tokens = torch.LongTensor(all_tokens)

        # 滑动窗口切分
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


def create_loader(texts, char_to_idx, embedding_matrix, batch_size, seq_len=256):
    """创建 DataLoader"""
    if len(texts) < 10:
        return None
    dataset = NeuroGlyphDataset(texts, char_to_idx, embedding_matrix, seq_len)
    if len(dataset) < 2:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ==================== 模型 ====================

class StreamCNN(nn.Module):
    """单路CNN：在T维度做1D卷积，保持T维度不变"""

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
        # x: (B, in_dim, T)
        out = self.conv(x)
        return out[:, :, :T]  # trim to T


class NeuroGlyph34XL(nn.Module):
    """34维 → 5路CNN + 5路Embedding → 融合 → 10层Transformer → LM Head"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── 5路CNN ──
        cnn_slices = [
            (0, 5),    # 四角 (5维)
            (5, 9),    # 五笔 (4维)
            (10, 13),  # 部首索引 (3维)
            (16, 26),  # 笔顺 (10维)
            (26, 32),  # 拼音 (6维)
        ]
        self.cnn_streams = nn.ModuleList()
        for (s, e), ch2, ks in zip(cnn_slices, cfg.cnn_channels, cfg.cnn_kernels):
            in_dim = e - s
            self.cnn_streams.append(StreamCNN(in_dim, in_dim * 2, ch2, ks))
        self.cnn_slices = cnn_slices

        # ── 5路Embedding ──
        self.emb_total_strokes = nn.Embedding(37, cfg.embed_dims[0])     # 总笔画 0-36
        self.emb_radical_extra = nn.Embedding(27, cfg.embed_dims[1])     # 外笔画 -1~25 → +1 偏移
        self.emb_same_group = nn.Embedding(57, cfg.embed_dims[2])        # 同部同画 0-56
        self.emb_structure = nn.Embedding(15, cfg.embed_dims[3])         # 结构 0-14
        self.emb_symbol = nn.Embedding(256, cfg.embed_dims[4])           # 符号1*16+符号2

        # ── 融合 ──
        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_input, cfg.d_fusion),
            nn.LayerNorm(cfg.d_fusion),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # ── 位置编码 ──
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, cfg.max_seq_len, cfg.d_fusion)
        )
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # ── Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_fusion,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln_final = nn.LayerNorm(cfg.d_fusion)

        # ── LM Head ──
        self.lm_head = nn.Linear(cfg.d_fusion, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, T, 34) — 34维特征向量序列
        """
        B, T, _ = x.shape

        # ── CNN流 ──
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)  # (B, in_dim, T)
            feat = stream(feat, T)                 # (B, ch2, T)
            feat = feat.permute(0, 2, 1)           # (B, T, ch2)
            cnn_outputs.append(feat)

        # ── Embedding流 ──
        emb_outputs = []
        # 总笔画
        idx = x[:, :, 9].long().clamp(0, 36)
        emb_outputs.append(self.emb_total_strokes(idx))
        # 外笔画
        idx = (x[:, :, 13] + 1).long().clamp(0, 26)
        emb_outputs.append(self.emb_radical_extra(idx))
        # 同部同画
        idx = x[:, :, 14].long().clamp(0, 56)
        emb_outputs.append(self.emb_same_group(idx))
        # 结构
        idx = x[:, :, 15].long().clamp(0, 14)
        emb_outputs.append(self.emb_structure(idx))
        # 符号
        idx = (x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)
        emb_outputs.append(self.emb_symbol(idx))

        # ── 融合 ──
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)  # (B, T, 416)
        fused = self.fusion(merged)                            # (B, T, 384)
        fused = fused + self.pos_encoding[:, :T, :]

        # ── Transformer ──
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)

        # ── LM Head ──
        logits = self.lm_head(hidden)
        return logits

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==================== 训练引擎 ====================

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        self.best_loss = float("inf")
        self.scaler = GradScaler() if cfg.use_amp and self.device.type == "cuda" else None

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # 加载嵌入和词表
        emb_path = os.path.join(cfg.data_dir, cfg.embedding_file)
        self.embedding_matrix = np.load(emb_path)
        with open(os.path.join(cfg.data_dir, cfg.vocab_file), "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)

        logger.info(f"词表: {len(self.char_to_idx)}, 嵌入: {self.embedding_matrix.shape}")
        logger.info(f"设备: {self.device}, AMP: {'ON' if self.scaler else 'OFF'}")

    def build_model(self):
        self.model = NeuroGlyph34XL(self.cfg).to(self.device)
        total, trainable = self.model.param_count()
        logger.info(f"参数: {total:,} 总, {trainable:,} 可训练")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.cfg.warmup_steps)
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.total_steps - self.cfg.warmup_steps,
            eta_min=self.cfg.lr * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[self.cfg.warmup_steps]
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def save_checkpoint(self, name="latest"):
        ckpt = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
        }
        fpath = os.path.join(self.cfg.checkpoint_dir, f"{name}.pt")
        tmp = fpath + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, fpath)
        logger.info(f"Checkpoint: {fpath} (step={self.step})")

        # 清理旧checkpoint
        all_ckpts = sorted(
            glob.glob(os.path.join(self.cfg.checkpoint_dir, "step_*.pt")),
            key=os.path.getmtime,
            reverse=True,
        )
        for old in all_ckpts[self.cfg.checkpoint_keep :]:
            os.remove(old)

    def load_checkpoint(self, path=None):
        if path is None:
            ckpt_files = glob.glob(os.path.join(self.cfg.checkpoint_dir, "*.pt"))
            if not ckpt_files:
                return False
            path = max(ckpt_files, key=os.path.getmtime)

        logger.info(f"恢复: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        return True

    def train_stage(self, stage_name, texts, epochs):
        """训练一个课程学习阶段"""
        if len(texts) < 100:
            logger.info(f"  阶段 [{stage_name}] 数据不足，跳过")
            return

        loader = create_loader(
            texts, self.char_to_idx, self.embedding_matrix,
            self.cfg.micro_batch, self.cfg.max_seq_len
        )
        if loader is None:
            logger.info(f"  阶段 [{stage_name}] DataLoader创建失败，跳过")
            return

        stage_steps = epochs * len(loader) * self.cfg.gradient_accumulation
        logger.info(f"  阶段 [{stage_name}]: {len(texts)} 条, {len(loader)} batch, ~{stage_steps} 步")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            train_iter = iter(loader)

            while True:
                # 梯度累积
                self.optimizer.zero_grad()
                micro_losses = []

                for _ in range(self.cfg.gradient_accumulation):
                    try:
                        feat_seq, target_seq = next(train_iter)
                    except StopIteration:
                        # 本epoch结束
                        break

                    feat_seq = feat_seq.to(self.device)
                    target_seq = target_seq.to(self.device)

                    if self.scaler:
                        with autocast():
                            logits = self.model(feat_seq)
                            loss = self.criterion(
                                logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1)
                            )
                            loss = loss / self.cfg.gradient_accumulation
                        self.scaler.scale(loss).backward()
                    else:
                        logits = self.model(feat_seq)
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1)
                        )
                        loss = loss / self.cfg.gradient_accumulation
                        loss.backward()

                    micro_losses.append(loss.item() * self.cfg.gradient_accumulation)

                if not micro_losses:
                    break  # epoch结束

                # 梯度裁剪 + 更新
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.step += 1

                step_loss = sum(micro_losses) / len(micro_losses)
                epoch_loss += step_loss
                batch_count += 1

                if self.step % 100 == 0:
                    logger.info(
                        f"  Step {self.step:>6} | Loss: {step_loss:.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e} | [{stage_name} E{epoch+1}]"
                    )

            avg_loss = epoch_loss / max(batch_count, 1)
            logger.info(
                f"  阶段 [{stage_name}] Epoch {epoch+1}/{epochs} | "
                f"Avg Loss: {avg_loss:.4f} | Step: {self.step}"
            )

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best")

        self.save_checkpoint(f"stage_{stage_name}")
        self.save_checkpoint("latest")

    def train(self):
        """完整的课程学习训练流程"""
        self.build_model()

        # 尝试断点续训
        resumed = self.load_checkpoint()
        if resumed:
            logger.info(f"从 step={self.step} 断点续训，已有阶段已跳过")
            start_from = self._get_completed_stage()
        else:
            start_from = None
            logger.info("全新训练开始")

        stages = list(config.stage_epochs.keys())

        logger.info("=" * 60)
        logger.info("课程学习训练计划:")
        for name, ep in config.stage_epochs.items():
            marker = " ← 续训起点" if start_from and name == start_from else ""
            logger.info(f"  {name}: {ep} epochs{marker}")

        # 按顺序执行各阶段
        skip = True if start_from else False
        for stage_name in stages:
            if skip:
                if stage_name == start_from:
                    skip = False
                else:
                    logger.info(f"  阶段 [{stage_name}] 已完成，跳过")
                    continue

            logger.info(f"\n{'='*40}")
            logger.info(f"📚 阶段: {stage_name}")
            logger.info(f"{'='*40}")

            texts = load_stage_data(stage_name)
            epochs = config.stage_epochs[stage_name]
            self.train_stage(stage_name, texts, epochs)

        # 最终保存
        self.save_checkpoint("final")
        logger.info(f"\n✅ 训练完成! 总步数: {self.step}, 最佳Loss: {self.best_loss:.4f}")

    def _get_completed_stage(self):
        """根据已有checkpoint推断已完成到哪个阶段"""
        for s in config.stage_epochs.keys():
            ckpt = os.path.join(self.cfg.checkpoint_dir, f"stage_{s}.pt")
            if not os.path.exists(ckpt):
                return s
        return "古文全语料"  # 全部完成


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="指定checkpoint路径")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        config.use_amp = False

    trainer = Trainer(config)

    if args.resume:
        trainer.build_model()
        trainer.load_checkpoint(args.resume)
        trainer.train()
    else:
        trainer.train()