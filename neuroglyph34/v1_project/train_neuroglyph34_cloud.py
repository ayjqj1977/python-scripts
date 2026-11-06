#!/usr/bin/env python3
"""
NeuroGlyph-34 云端大规模训练脚本
架构: 5路CNN + 5路Embedding → 融合256维 → 8层Transformer → LM Head
数据: 古文语料 + 知识图谱 + JSON语料 + 现代中文补充
特性: 自动断点续训、混合精度、梯度累积、原子保存
"""
import os, sys, json, math, time, random, logging, argparse, warnings, glob, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = project_dir
    checkpoint_dir = os.path.join(project_dir, "checkpoints_cloud")
    log_dir = os.path.join(project_dir, "logs_cloud")
    modern_text_dir = os.path.join(project_dir, "modern_text")

    # 数据文件
    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"
    corpus_files = [
        "corpus_combined.txt",
        "corpus_knowledge_graph.txt",
        "corpus_training.txt",
        "corpus_training_combined.txt",
    ]
    json_corpus_dir = "7_训练语料"

    # 模型 - 云端加大版
    d_input = 34
    vocab_size = 7987
    special_tokens = 4
    cnn_channels = [64, 64, 32, 128, 64]   # 四角/五笔/部首/笔顺/拼音
    cnn_kernels = [2, 2, 2, 3, 3]
    embed_dims = [16, 16, 16, 16, 32]       # 总笔画/外笔画/同部同画/结构/符号
    d_fusion = 256
    n_layers = 8
    n_heads = 8
    d_ff = 1024
    dropout = 0.15
    max_seq_len = 256

    # 训练
    micro_batch = 16           # 每步GPU实际batch
    gradient_accumulation = 8  # 累积8步 → 等效batch=128
    total_steps = 100_000
    lr = 1e-4
    warmup_steps = 3000
    weight_decay = 0.1
    grad_clip = 1.0
    eval_interval = 2000
    save_interval = 5000
    checkpoint_keep = 3       # 保留最近N个checkpoint

    # 混合精度
    use_amp = True

    # 现代语料下载
    auto_download_modern = True
    modern_corpus_urls = [
        # 小规模公开语料，优先使用
        "https://raw.githubusercontent.com/InsaneLife/ChineseNLPCorpus/master/sample.txt",
    ]


# ==================== 数据：所有语料合并加载 ====================

def load_json_corpus(dir_path):
    """加载7_训练语料目录下所有JSON文件，提取文本行"""
    texts = []
    if not os.path.isdir(dir_path):
        logger.warning(f"JSON语料目录不存在: {dir_path}")
        return texts
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 尝试多种JSON结构
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str) and len(v) > 1:
                                texts.append(v)
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str):
                        texts.append(v)
                    elif isinstance(v, list):
                        texts.extend([x for x in v if isinstance(x, str)])
            logger.info(f"  {fname}: {len(texts)} 条文本")
        except Exception as e:
            logger.warning(f"  跳过 {fname}: {e}")
    return texts


def load_all_texts(config):
    """合并所有语料源，返回去重后的字符流"""
    all_text = []
    total_chars = 0

    # 1. 加载大型文本语料
    for cf in config.corpus_files:
        fpath = os.path.join(config.data_dir, cf)
        if not os.path.exists(fpath):
            logger.warning(f"语料文件不存在: {fpath}")
            continue
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        logger.info(f"加载 {cf} ({size_mb:.1f}MB)...")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        # 规范化：合并空白、去除全角空格以外的不可见字符
        text = re.sub(r'[^\S\n]+', '', text)
        text = re.sub(r'\n+', '\n', text)
        all_text.append(text)
        total_chars += len(text)
        logger.info(f"  → {len(text):,} 字符")

    # 2. 加载JSON语料
    json_dir = os.path.join(config.data_dir, config.json_corpus_dir)
    json_texts = load_json_corpus(json_dir)
    if json_texts:
        json_combined = "\n".join(json_texts)
        json_combined = re.sub(r'[^\S\n]+', '', json_combined)
        all_text.append(json_combined)
        logger.info(f"JSON语料合计: {len(json_combined):,} 字符")
        total_chars += len(json_combined)

    # 3. 加载现代语料（如果存在）
    modern_dir = config.modern_text_dir
    if os.path.isdir(modern_dir):
        for fname in sorted(os.listdir(modern_dir)):
            fpath = os.path.join(modern_dir, fname)
            if fname.endswith((".txt", ".json", ".csv")):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        modern_text = f.read()
                    modern_text = re.sub(r'[^\S\n]+', '', modern_text)
                    all_text.append(modern_text)
                    logger.info(f"现代语料 {fname}: {len(modern_text):,} 字符")
                    total_chars += len(modern_text)
                except Exception as e:
                    logger.warning(f"跳过 {fname}: {e}")

    if not all_text:
        raise RuntimeError("没有找到任何语料文件！")

    # 合并、去重连续重复行、限制总长
    combined = "\n".join(all_text)
    combined = re.sub(r'\n{3,}', '\n\n', combined)

    logger.info(f"语料总计: {total_chars:,} 字符 ({total_chars/1e6:.1f}MB)")
    return combined


def download_modern_corpus(config):
    """尝试下载现代中文语料"""
    import urllib.request
    os.makedirs(config.modern_text_dir, exist_ok=True)

    downloaded = 0
    for url in config.modern_corpus_urls:
        try:
            fname = url.split("/")[-1] or f"modern_{downloaded}.txt"
            fpath = os.path.join(config.modern_text_dir, fname)
            if os.path.exists(fpath):
                logger.info(f"现代语料已存在: {fname}")
                downloaded += 1
                continue
            logger.info(f"下载现代语料: {url[:80]}...")
            urllib.request.urlretrieve(url, fpath)
            size = os.path.getsize(fpath)
            logger.info(f"  下载完成: {size:,} bytes → {fname}")
            downloaded += 1
        except Exception as e:
            logger.warning(f"下载失败: {e}")

    # 回退：生成少量现代中文示例文本用于平衡训练
    if downloaded == 0:
        fallback_path = os.path.join(config.modern_text_dir, "modern_fallback.txt")
        if not os.path.exists(fallback_path):
            logger.info("生成现代中文回退语料...")
            modern_samples = generate_modern_fallback()
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(modern_samples)
            logger.info(f"  回退语料: {len(modern_samples):,} 字符")


def generate_modern_fallback():
    """生成结构化的现代中文示例（用于平衡古文分布）"""
    import itertools
    templates = [
        "中国的经济发展取得了举世瞩目的成就，人民生活水平不断提高。",
        "人工智能技术正在深刻改变着我们的生活方式和工作模式。",
        "教育事业的发展是国家繁荣昌盛的重要基础。",
        "科技创新是引领发展的第一动力，必须坚持自主创新。",
        "环境保护是每个人的责任，绿色发展理念需要深入人心。",
        "文化交流促进了不同文明之间的相互理解和尊重。",
        "健康是人生最宝贵的财富，需要养成良好的生活习惯。",
        "数字经济已经成为推动全球经济增长的重要引擎。",
        "法律面前人人平等，这是现代法治社会的基本原则。",
        "信息时代的知识更新速度前所未有，终身学习成为必然选择。",
        "城市的现代化建设需要兼顾历史文化保护与可持续发展。",
        "食品安全关系到每个人的健康和生命安全。",
        "互联网的普及改变了信息传播的方式和社会交往的模式。",
        "农业现代化是乡村振兴战略的核心内容之一。",
        "科学精神的培养需要从小抓起，激发探索未知的兴趣。",
        "交通运输网络的建设极大促进了区域经济的协调发展。",
        "传统文化需要在创新中传承，在传承中发展。",
        "医疗保障体系的完善是衡量社会进步的重要标志。",
        "和平与发展仍然是当今时代的主题。",
        "青年是国家的未来，需要为他们创造良好的成长环境。",
    ]
    # 句子组合成段落
    paragraphs = []
    for i in range(200):
        combo = "。".join(random.sample(templates, min(5, len(templates))))
        paragraphs.append(combo + "。")
    return "\n".join(paragraphs)


# ==================== 模型 ====================

class NeuroGlyph34Large(nn.Module):
    """34维 → 5路CNN + 5路Embedding → 融合 → 8层Transformer → LM Head"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # --- 5路CNN流 ---
        # 维度分配: 四角0-4(5), 五笔5-8(4), 部首9-11(3), 笔顺16-25(10), 拼音26-31(6)
        cnn_slices = [
            (0, 5),     # 四角
            (5, 9),     # 五笔
            (10, 13),   # 部首索引
            (16, 26),   # 笔顺
            (26, 32),   # 拼音
        ]

        self.cnn_convs = nn.ModuleList()
        for (start, end), ch, ks in zip(cnn_slices, config.cnn_channels, config.cnn_kernels):
            in_dim = end - start
            self.cnn_convs.append(nn.Sequential(
                nn.Conv1d(in_dim, ch, kernel_size=ks, padding=ks - 1),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=ks, padding=ks - 1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            ))

        # --- 5路Embedding流 ---
        embedding_slices = [
            (9, 10),    # 总笔画
            (13, 14),   # 外笔画
            (14, 15),   # 同部同画
            (15, 16),   # 结构
            (32, 34),   # 符号
        ]
        embedding_cardinalities = [100, 100, 50, 20, 20]

        self.embeddings = nn.ModuleList()
        for (start, end), card, dim in zip(embedding_slices, embedding_cardinalities, config.embed_dims):
            self.embeddings.append(nn.Embedding(card, dim))

        # --- 融合层 ---
        total_cnn = sum(config.cnn_channels)   # 64+64+32+128+64 = 352
        total_emb = sum(config.embed_dims)      # 16+16+16+16+32 = 96
        total_features = total_cnn + total_emb   # 448
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.d_fusion),
            nn.LayerNorm(config.d_fusion),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --- 位置编码 ---
        self.pos_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_fusion))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_fusion,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.ln_final = nn.LayerNorm(config.d_fusion)

        # --- LM Head ---
        self.lm_head = nn.Linear(config.d_fusion, config.vocab_size, bias=False)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, attention_mask=None):
        """
        x: (B, seq_len, 34) - 34维特征向量序列
        """
        B, T, _ = x.shape

        # CNN流：对每个batch的每个位置维度做1D卷积
        cnn_outputs = []
        cnn_slices = [(0, 5), (5, 9), (10, 13), (16, 26), (26, 32)]
        for (s, e), conv in zip(cnn_slices, self.cnn_convs):
            feat = x[:, :, s:e]                          # (B, T, in_dim)
            feat = feat.permute(0, 2, 1)                 # (B, in_dim, T)
            feat = conv(feat)                            # (B, ch, 1)
            feat = feat.squeeze(-1)                      # (B, ch)
            feat = feat.unsqueeze(1).expand(-1, T, -1)   # (B, T, ch)
            cnn_outputs.append(feat)

        # Embedding流
        emb_outputs = []
        emb_slices = [(9, 10), (13, 14), (14, 15), (15, 16), (32, 34)]
        for (s, e), emb in zip(emb_slices, self.embeddings):
            indices = x[:, :, s].long().clamp(0, emb.num_embeddings - 1)  # (B, T)
            feat = emb(indices)                                            # (B, T, dim)
            emb_outputs.append(feat)

        # 拼接所有流 → 融合
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)  # (B, T, 448)
        fused = self.fusion(merged)                             # (B, T, 256)

        # 位置编码
        fused = fused + self.pos_encoding[:, :T, :]

        # 因果注意力掩码
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        else:
            causal_mask = attention_mask

        # Transformer
        hidden = self.transformer(fused, mask=causal_mask, is_causal=True)
        hidden = self.ln_final(hidden)

        # LM Head
        logits = self.lm_head(hidden)  # (B, T, vocab_size)
        return logits

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==================== 数据集 ====================

class NeuroGlyphDataset(Dataset):
    """将字符序列转为34维特征序列 + 下一字符标签"""

    def __init__(self, text, char_to_idx, embedding_matrix, config: Config):
        self.config = config
        self.seq_len = config.max_seq_len
        self.char_to_idx = char_to_idx
        self.embedding_matrix = torch.FloatTensor(embedding_matrix)

        # 将文本转为token序列
        tokens = []
        for ch in text:
            if ch in char_to_idx:
                tokens.append(char_to_idx[ch])
            elif ch == '\n':
                tokens.append(1)  # UNK/SOS
        self.tokens = torch.LongTensor(tokens)

        # 切分序列
        stride = self.seq_len // 2
        self.starts = list(range(0, max(1, len(self.tokens) - self.seq_len - 1), stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        token_seq = self.tokens[start:end]       # (seq_len,)
        target_seq = self.tokens[start + 1:end + 1]  # (seq_len,)
        # 特征向量
        feat_seq = self.embedding_matrix[token_seq]  # (seq_len, 34)
        return feat_seq, target_seq


# ==================== 训练引擎 ====================

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        self.best_loss = float("inf")
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None

        # 创建目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # 加载嵌入和词表
        emb_path = os.path.join(config.data_dir, config.embedding_file)
        self.embedding_matrix = np.load(emb_path)
        with open(os.path.join(config.data_dir, config.vocab_file), "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        logger.info(f"词表大小: {len(self.char_to_idx)}, 嵌入矩阵: {self.embedding_matrix.shape}")
        logger.info(f"设备: {self.device}")
        logger.info(f"混合精度: {'ON' if self.scaler else 'OFF'}")

    def prepare_data(self):
        """加载并预处理所有语料"""
        logger.info("=" * 60)
        logger.info("准备训练数据...")

        # 尝试下载现代语料
        if self.config.auto_download_modern:
            download_modern_corpus(self.config)

        # 加载所有语料
        text = load_all_texts(self.config)

        # 创建数据集
        dataset = NeuroGlyphDataset(text, self.char_to_idx, self.embedding_matrix, self.config)
        self.train_loader = DataLoader(
            dataset, batch_size=self.config.micro_batch, shuffle=True,
            num_workers=0 if self.device.type == "cuda" else 0,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        logger.info(f"数据集: {len(dataset):,} 个序列, {len(self.train_loader):,} 个batch")
        logger.info(f"等效batch: {self.config.micro_batch}×{self.config.gradient_accumulation}={self.config.micro_batch * self.config.gradient_accumulation}")

    def build_model(self):
        """构建模型、优化器、调度器、损失函数"""
        self.model = NeuroGlyph34Large(self.config).to(self.device)
        total, trainable = self.model.get_param_count()
        logger.info(f"模型参数: {total:,} 总参数, {trainable:,} 可训练")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.config.total_steps - self.config.warmup_steps, eta_min=self.config.lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_steps])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD=0

    def save_checkpoint(self, name="latest"):
        """原子保存checkpoint：先写临时文件再重命名"""
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "config": {k: v for k, v in vars(self.config).items() if not k.startswith("_")},
        }
        fpath = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        tmp_path = fpath + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, fpath)
        logger.info(f"Checkpoint 已保存: {fpath} (step={self.step})")

        # 清理旧checkpoint
        all_ckpts = sorted(
            glob.glob(os.path.join(self.config.checkpoint_dir, "step_*.pt")),
            key=os.path.getmtime, reverse=True
        )
        for old in all_ckpts[self.config.checkpoint_keep:]:
            os.remove(old)

    def load_checkpoint(self, path=None):
        """加载checkpoint，返回True表示成功"""
        if path is None:
            # 自动找最新
            ckpt_files = glob.glob(os.path.join(self.config.checkpoint_dir, "*.pt"))
            if not ckpt_files:
                return False
            path = max(ckpt_files, key=os.path.getmtime)

        logger.info(f"加载checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.step = checkpoint["step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.scaler and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        return True

    def train(self):
        self.prepare_data()
        self.build_model()

        # 尝试恢复
        if self.load_checkpoint():
            logger.info(f"从 step={self.step} 恢复训练")
        else:
            logger.info("全新训练开始")

        logger.info("=" * 60)
        logger.info(f"总步数: {self.config.total_steps}, 当前步: {self.step}")
        logger.info(f"保存间隔: {self.config.save_interval}, 评估间隔: {self.config.eval_interval}")

        self.model.train()
        train_iter = iter(self.train_loader)
        start_time = time.time()
        total_loss = 0.0
        accum_loss = 0.0

        while self.step < self.config.total_steps:
            self.step += 1

            # 梯度累积
            self.optimizer.zero_grad()
            micro_losses = []

            for _ in range(self.config.gradient_accumulation):
                try:
                    feat_seq, target_seq = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    feat_seq, target_seq = next(train_iter)

                feat_seq = feat_seq.to(self.device)      # (B, T, 34)
                target_seq = target_seq.to(self.device)   # (B, T)

                if self.scaler:
                    with autocast():
                        logits = self.model(feat_seq)    # (B, T, V)
                        loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                        loss = loss / self.config.gradient_accumulation
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(feat_seq)
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                    loss = loss / self.config.gradient_accumulation
                    loss.backward()

                micro_losses.append(loss.item() * self.config.gradient_accumulation)

            # 梯度裁剪 + 参数更新
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # 记录
            step_loss = sum(micro_losses) / len(micro_losses)
            accum_loss += step_loss
            total_loss += step_loss

            # 日志
            if self.step % 100 == 0:
                elapsed = time.time() - start_time
                steps_left = self.config.total_steps - self.step
                avg_speed = self.step / elapsed * 100  # steps/100s
                eta_min = steps_left / (self.step / elapsed) / 60
                avg_loss = accum_loss / 100
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {self.step:>6} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"ETA: {eta_min:.0f}min | Speed: {avg_speed:.1f}s/100step"
                )
                accum_loss = 0.0

            # 评估
            if self.step % self.config.eval_interval == 0:
                avg = total_loss / self.step
                logger.info(f"--- 评估 @ step {self.step} | 平均Loss: {avg:.4f} ---")
                if avg < self.best_loss:
                    self.best_loss = avg
                    self.save_checkpoint("best")

            # 保存
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}")
                self.save_checkpoint("latest")

        # 训练结束保存
        self.save_checkpoint("final")
        self.save_checkpoint("latest")
        elapsed = time.time() - start_time
        logger.info(f"训练完成! 总耗时: {elapsed/3600:.1f}小时, 最终Loss: {total_loss/self.step:.4f}")


# ==================== 入口 ====================
if __name__ == "__main__":
    import re  # 用于语料规范化

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None, help="覆盖总步数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--resume", type=str, default=None, help="指定checkpoint路径恢复")
    parser.add_argument("--cpu", action="store_true", help="强制CPU训练")
    args = parser.parse_args()

    config = Config()
    if args.steps:
        config.total_steps = args.steps
    if args.lr:
        config.lr = args.lr
    if args.cpu:
        config.use_amp = False

    trainer = Trainer(config)
    if args.resume:
        trainer.prepare_data()
        trainer.build_model()
        trainer.load_checkpoint(args.resume)
        trainer.train()
    else:
        trainer.train()