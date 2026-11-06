# -*- coding: utf-8 -*-
"""
NeuroGlyph-34 Cloud V3 — 验证集+生成采样+形近字判别+阶段追踪
===========================================================
V3 改进:
  1. 验证集: 每阶段10%数据 → 每epoch评估loss + perplexity
  2. 生成采样: 每阶段结束后用种子提示词贪婪生成，直观检查质量
  3. 形近字测试: 计算模型内部表征的余弦相似度，量化判别力
  4. 阶段追踪: checkpoint 保存 current_stage + current_epoch，可靠续训
  5. 定时保存: save_interval 真正生效，每N步存step checkpoint
  6. 重复轮次: 若 resumed 到的阶段已训练部分epoch，从剩余epoch继续

架构: 34维特征 → 5路CNN+5路Emb → 融合384维 → 10层12头Transformer → LM Head
参数量: ~21M
"""
import os, sys, json, math, time, random, logging, argparse, warnings, glob, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # 自包含：数据文件已复制到本目录，直接上传整个文件夹即可云端运行
    data_dir = project_dir  # D:\Desktop\neuroglyph34_cloud_v2\
    checkpoint_dir = os.path.join(project_dir, "checkpoints_v3")
    log_dir = os.path.join(project_dir, "logs_v3")

    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"

    d_input = 34
    vocab_size = 7987
    special_tokens = 4

    cnn_channels = [64, 64, 64, 64, 64]
    cnn_kernels = [2, 2, 2, 3, 3]
    embed_dims = [16, 16, 16, 16, 32]

    total_cnn = sum(cnn_channels)      # 320
    total_emb = sum(embed_dims)         # 96
    fusion_input = total_cnn + total_emb  # 416

    d_fusion = 384
    n_layers = 10
    n_heads = 12
    d_ff = 1536
    dropout = 0.15
    max_seq_len = 256

    micro_batch = 16
    gradient_accumulation = 8
    lr = 1e-4
    warmup_steps = 2000
    weight_decay = 0.1
    grad_clip = 1.0
    save_interval = 5000
    checkpoint_keep = 3
    use_amp = True

    # 验证集比例
    val_split = 0.1

    # 每阶段生成采样的种子提示词
    sample_prompts = [
        "春眠不觉晓",
        "床前明月光",
        "人之初性本善",
        "子曰学而时习之",
        "天地玄黄宇宙",
        "关关雎鸠在河",
        "白日依山尽",
        "锄禾日当午",
    ]
    sample_max_len = 24  # 生成最多 N 个新字符
    sample_top_k = 40     # top-k 采样
    sample_temperature = 0.8

    # 形近字对比损失
    sim_char_weight = 0.02  # 对比损失权重（0 禁用）
    sim_char_margin = 0.3   # 余弦相似度低于此值不惩罚

    # 形近字测试对 — 语义相近但特征不同的汉字
    similar_char_pairs = [
        ("人", "入"), ("已", "己"), ("未", "末"), ("干", "千"),
        ("土", "士"), ("刀", "力"), ("天", "夫"), ("太", "犬"),
        ("日", "曰"), ("目", "且"), ("甲", "由"), ("王", "玉"),
        ("午", "牛"), ("外", "处"), ("贝", "见"), ("爪", "瓜"),
        ("凡", "风"), ("大", "太"), ("本", "木"), ("矢", "失"),
        ("白", "百"), ("田", "由"), ("申", "电"), ("古", "占"),
        ("名", "各"), ("困", "因"), ("杏", "呆"), ("休", "体"),
        ("晴", "睛"), ("辨", "辩"), ("徒", "徙"), ("栗", "粟"),
        ("湍", "惴"), ("薄", "簿"), ("坠", "堕"), ("浙", "淅"),
        ("戍", "戌"), ("崇", "祟"), ("侯", "候"), ("竿", "竽"),
        ("氾", "汜"), ("赢", "羸"), ("券", "卷"), ("灸", "炙"),
        ("茶", "荼"), ("治", "冶"), ("准", "淮"), ("撤", "撒"),
    ]

    stage_epochs = {
        "单字": 3,
        "词语": 4,
        "成语": 4,
        "诗词": 6,
        "古文全语料": 18,
    }

    total_steps = 300_000


config = Config()

# ==================== 数据加载 ====================

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
            elif isinstance(item, dict):
                text = " ".join(str(v) for v in item.values() if isinstance(v, str))
            else:
                continue

            # 处理列表类型值（如诗词的 paragraphs 是行 list）
            if isinstance(text, list):
                for line in text:
                    if isinstance(line, str) and len(line) > 0:
                        texts.append(line)
            elif isinstance(text, str) and len(text) > 0:
                texts.append(text)
    except Exception as e:
        logger.warning(f"  加载JSON失败 {filepath}: {e}")
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
    except Exception as e:
        logger.warning(f"  加载TXT失败 {filepath}: {e}")
    return texts


def load_stage_data(stage_name):
    data_dir = config.data_dir
    texts = []

    if stage_name == "单字":
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

        mengxue_dir = os.path.join(data_dir, "corpus")
        for fname in sorted(os.listdir(mengxue_dir)):
            if fname.startswith("蒙学_") and fname.endswith(".json"):
                full = os.path.join(mengxue_dir, fname)
                t = load_json_texts(full, "content", 3000)
                texts.extend(t)
                logger.info(f"  蒙学/{fname}: {len(t)} 条")

        for cf, limit in [
            ("corpus_combined.txt", 100000),
            ("corpus_training.txt", 50000),
        ]:
            full = os.path.join(data_dir, cf)
            if os.path.exists(full):
                t = load_txt_lines(full, limit)
                texts.extend(t)
                logger.info(f"  {cf}: {len(t)} 条")

    texts = list(set(texts))
    random.shuffle(texts)
    logger.info(f"  阶段 [{stage_name}] 合计: {len(texts)} 条")
    return texts


# ==================== 数据集 ====================

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


def create_loader(texts, char_to_idx, embedding_matrix, batch_size, seq_len=256, shuffle=True):
    if len(texts) < 10:
        return None
    dataset = NeuroGlyphDataset(texts, char_to_idx, embedding_matrix, seq_len)
    if len(dataset) < 2:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def create_train_val_loaders(texts, char_to_idx, embedding_matrix, batch_size, seq_len=256, val_split=0.1):
    """划分训练集和验证集"""
    if len(texts) < 100:
        # 数据太少，全用作训练
        train_loader = create_loader(texts, char_to_idx, embedding_matrix, batch_size, seq_len, shuffle=True)
        return train_loader, None

    # 先创建完整 dataset
    all_dataset = NeuroGlyphDataset(texts, char_to_idx, embedding_matrix, seq_len)
    n_total = len(all_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(all_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


# ==================== 模型 ====================

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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        cnn_slices = [
            (0, 5),
            (5, 9),
            (10, 13),
            (16, 26),
            (26, 32),
        ]
        self.cnn_streams = nn.ModuleList()
        for (s, e), ch2, ks in zip(cnn_slices, cfg.cnn_channels, cfg.cnn_kernels):
            in_dim = e - s
            self.cnn_streams.append(StreamCNN(in_dim, in_dim * 2, ch2, ks))
        self.cnn_slices = cnn_slices

        self.emb_total_strokes = nn.Embedding(37, cfg.embed_dims[0])
        self.emb_radical_extra = nn.Embedding(27, cfg.embed_dims[1])
        self.emb_same_group = nn.Embedding(57, cfg.embed_dims[2])
        self.emb_structure = nn.Embedding(15, cfg.embed_dims[3])
        self.emb_symbol = nn.Embedding(256, cfg.embed_dims[4])

        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_input, cfg.d_fusion),
            nn.LayerNorm(cfg.d_fusion),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.pos_encoding = nn.Parameter(
            torch.zeros(1, cfg.max_seq_len, cfg.d_fusion)
        )
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

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
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T)
            feat = feat.permute(0, 2, 1)
            cnn_outputs.append(feat)

        emb_outputs = []
        emb_outputs.append(self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)))
        emb_outputs.append(self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)))
        emb_outputs.append(self.emb_same_group(x[:, :, 14].long().clamp(0, 56)))
        emb_outputs.append(self.emb_structure(x[:, :, 15].long().clamp(0, 14)))
        emb_outputs.append(self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)))

        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged)
        fused = fused + self.pos_encoding[:, :T, :]

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        logits = self.lm_head(hidden)
        return logits

    def get_hidden(self, x):
        """返回 Transformer 输出（不经过 LM Head），用于形近字测试"""
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T)
            feat = feat.permute(0, 2, 1)
            cnn_outputs.append(feat)

        emb_outputs = []
        emb_outputs.append(self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)))
        emb_outputs.append(self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)))
        emb_outputs.append(self.emb_same_group(x[:, :, 14].long().clamp(0, 56)))
        emb_outputs.append(self.emb_structure(x[:, :, 15].long().clamp(0, 14)))
        emb_outputs.append(self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)))

        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged)
        fused = fused + self.pos_encoding[:, :T, :]

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        return hidden

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
        # V3: 阶段追踪
        self.current_stage = None
        self.current_epoch = 0

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        emb_path = os.path.join(cfg.data_dir, cfg.embedding_file)
        self.embedding_matrix = np.load(emb_path)
        with open(os.path.join(cfg.data_dir, cfg.vocab_file), "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)

        # V3: 构建 idx→char 反向映射（用于生成采样）
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # V3: 过滤形近字对，只保留在词表中的
        self.similar_pairs = [
            (a, b) for a, b in cfg.similar_char_pairs
            if a in self.char_to_idx and b in self.char_to_idx
        ]
        logger.info(f"词表: {len(self.char_to_idx)}, 嵌入: {self.embedding_matrix.shape}")
        logger.info(f"设备: {self.device}, AMP: {'ON' if self.scaler else 'OFF'}")
        logger.info(f"形近字测试对: {len(self.similar_pairs)}/48")

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

    # ── V3: 验证函数 ──
    @torch.no_grad()
    def validate(self, val_loader):
        """计算验证集 Loss 和 Perplexity"""
        if val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for feat_seq, target_seq in val_loader:
            feat_seq = feat_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            logits = self.model(feat_seq)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1)
            )
            # 计数非填充token
            valid_tokens = (target_seq != 0).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        self.model.train()
        if total_tokens == 0:
            return None, None

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    # ── V3: 生成采样 ──
    @torch.no_grad()
    def generate_sample(self):
        """用种子提示词 top-k 采样生成文本，打破贪婪解码的重复模式"""
        self.model.eval()
        results = []
        max_len = self.cfg.max_seq_len
        top_k = self.cfg.sample_top_k
        temperature = self.cfg.sample_temperature

        for prompt in self.cfg.sample_prompts:
            # 将提示词转为 token 序列
            tokens = []
            for ch in prompt:
                if ch in self.char_to_idx:
                    tokens.append(self.char_to_idx[ch])
                else:
                    break

            if len(tokens) < 2:
                continue

            input_tokens = tokens[: min(len(tokens), max_len - self.cfg.sample_max_len)]
            generated = list(prompt[: len(input_tokens)])

            for _ in range(self.cfg.sample_max_len):
                # 构造输入
                seq = torch.LongTensor(input_tokens).unsqueeze(0)  # (1, T)
                emb_mat = torch.from_numpy(self.embedding_matrix).float().to(self.device)
                feat = emb_mat[seq]  # (1, T, 34)

                logits = self.model(feat)  # (1, T, vocab)
                next_logits = logits[0, -1, :]  # (vocab,)

                # top-k + temperature 采样，打破重复循环
                next_logits = next_logits / temperature
                topk_vals, topk_indices = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                probs = F.softmax(topk_vals, dim=0)
                chosen = torch.multinomial(probs, 1).item()
                next_token = topk_indices[chosen].item()

                if next_token == 0:  # padding token → 停止
                    break

                gen_char = self.idx_to_char.get(next_token, "?")
                generated.append(gen_char)
                input_tokens.append(next_token)
                input_tokens = input_tokens[-max_len:]  # 保持长度

            results.append("".join(generated))

        # 后处理：在 JSON 污染首次出现处截断
        clean_results = []
        for r in results:
            for stop_pattern in ["{'", "':'", "','"]:
                idx = r.find(stop_pattern)
                if idx > 0:
                    r = r[:idx]
            clean_results.append(r)
        results = clean_results

        self.model.train()
        return results

    # ── V3: 形近字判别力测试 ──
    @torch.no_grad()
    def test_similar_chars(self):
        """对形近字对计算模型内部表征的余弦相似度"""
        self.model.eval()

        # 收集所有形近字的单字表征
        all_chars = set()
        for a, b in self.similar_pairs:
            all_chars.add(a)
            all_chars.add(b)

        char_hidden = {}
        for ch in all_chars:
            idx = self.char_to_idx[ch]
            # 构造单字序列（重复两次以形成上下文）
            tokens = torch.LongTensor([idx, idx]).unsqueeze(0).to(self.device)  # (1, 2)
            emb_mat = torch.from_numpy(self.embedding_matrix).float().to(self.device)
            feat = emb_mat[tokens]  # (1, 2, 34)
            hidden = self.model.get_hidden(feat)  # (1, 2, 384)
            # 取最后一个位置的 hidden 作为该字的表征
            char_hidden[ch] = hidden[0, -1, :]  # (384,)

        # 计算每对余弦相似度
        similarities = []
        for a, b in self.similar_pairs:
            if a in char_hidden and b in char_hidden:
                sim = F.cosine_similarity(
                    char_hidden[a].unsqueeze(0), char_hidden[b].unsqueeze(0)
                ).item()
                similarities.append((a, b, sim))

        # 排序：相似度越高越危险（模型难区分）
        similarities.sort(key=lambda x: x[2], reverse=True)

        self.model.train()
        return similarities

    # ── V3: 形近字对比损失 ──
    def compute_similar_char_loss(self):
        """对形近字对的 hidden state 计算对比损失，拉开表征距离"""
        if not self.similar_pairs:
            return torch.tensor(0.0, device=self.device)

        self.model.eval()
        all_chars = set()
        for a, b in self.similar_pairs:
            all_chars.add(a); all_chars.add(b)

        emb_mat = torch.from_numpy(self.embedding_matrix).float().to(self.device)
        char_hidden = {}
        for ch in all_chars:
            idx = self.char_to_idx[ch]
            tokens = torch.LongTensor([idx, idx]).unsqueeze(0).to(self.device)
            feat = emb_mat[tokens]
            hidden = self.model.get_hidden(feat)
            char_hidden[ch] = hidden[0, -1, :]  # (384,)

        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for a, b in self.similar_pairs:
            if a in char_hidden and b in char_hidden:
                sim = F.cosine_similarity(
                    char_hidden[a].unsqueeze(0), char_hidden[b].unsqueeze(0)
                ).squeeze()
                # 只惩罚相似度超过 margin 的对
                loss += torch.clamp(sim - self.cfg.sim_char_margin, min=0)
                count += 1

        self.model.train()
        return loss / max(count, 1)

    # ── Checkpoint（V3: 含阶段追踪） ──
    def save_checkpoint(self, name="latest"):
        ckpt = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            # V3: 阶段追踪
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
        }
        fpath = os.path.join(self.cfg.checkpoint_dir, f"{name}.pt")
        tmp = fpath + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, fpath)
        logger.info(f"Checkpoint: {fpath} (step={self.step}, stage={self.current_stage}, epoch={self.current_epoch})")

        # cleanup
        all_ckpts = sorted(
            glob.glob(os.path.join(self.cfg.checkpoint_dir, "step_*.pt")),
            key=os.path.getmtime, reverse=True,
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
        # V3: 恢复阶段追踪
        self.current_stage = ckpt.get("current_stage", None)
        self.current_epoch = ckpt.get("current_epoch", 0)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        logger.info(f"  阶段: {self.current_stage}, epoch: {self.current_epoch}")
        return True

    # ── 训练单阶段（V3: 含验证 + 定时保存） ──
    def train_stage(self, stage_name, texts, epochs):
        if len(texts) < 100:
            logger.info(f"  阶段 [{stage_name}] 数据不足，跳过")
            return

        # V3: 划分训练/验证集
        train_loader, val_loader = create_train_val_loaders(
            texts, self.char_to_idx, self.embedding_matrix,
            self.cfg.micro_batch, self.cfg.max_seq_len, self.cfg.val_split
        )
        if train_loader is None:
            logger.info(f"  阶段 [{stage_name}] DataLoader创建失败，跳过")
            return

        self.current_stage = stage_name

        n_epochs = len(train_loader) * self.cfg.gradient_accumulation
        logger.info(
            f"  阶段 [{stage_name}]: {len(texts)} 条, {len(train_loader)} batch/epoch"
        )
        if val_loader:
            logger.info(f"  验证集: {len(val_loader.dataset)} 条")

        start_epoch = 0

        # V3: 若续训到本阶段且已有部分epoch完成，跳过
        if self.current_epoch > 0 and self.current_epoch < epochs:
            start_epoch = self.current_epoch
            logger.info(f"  续训: 从 epoch {start_epoch + 1}/{epochs} 开始")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            train_iter = iter(train_loader)

            while True:
                self.optimizer.zero_grad()
                micro_losses = []

                for _ in range(self.cfg.gradient_accumulation):
                    try:
                        feat_seq, target_seq = next(train_iter)
                    except StopIteration:
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
                    break

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

                # V3: 定时保存
                if self.step % self.cfg.save_interval == 0 and self.step > 0:
                    self.save_checkpoint(f"step_{self.step}")

            avg_loss = epoch_loss / max(batch_count, 1)

            # V3: 验证
            val_loss, val_ppl = self.validate(val_loader)
            val_str = ""
            if val_loss is not None:
                val_str = f" | Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"

            # V3: 形近字对比损失（每 epoch 末反向传播一次）
            sim_loss = None
            if self.cfg.sim_char_weight > 0:
                self.optimizer.zero_grad()
                sim_loss = self.compute_similar_char_loss()
                weighted = sim_loss * self.cfg.sim_char_weight
                if weighted.item() > 0:
                    weighted.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                sim_str = f" | SimChar Loss: {sim_loss.item():.4f}"
            else:
                sim_str = ""

            logger.info(
                f"  阶段 [{stage_name}] Epoch {epoch+1}/{epochs} | "
                f"Avg Loss: {avg_loss:.4f}{val_str}{sim_str} | Step: {self.step}"
            )

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best")

        # V3: 阶段结束 — 生成采样 + 形近字测试
        logger.info(f"  [{stage_name}] 生成采样:")
        samples = self.generate_sample()
        for i, s in enumerate(samples):
            logger.info(f"    {i+1}. {s}")

        logger.info(f"  [{stage_name}] 形近字判别力测试 (Top 10 最相似):")
        sims = self.test_similar_chars()
        for a, b, sim in sims[:10]:
            status = "⚠️ 难区分" if sim > 0.85 else ("✓" if sim < 0.5 else "  ~")
            logger.info(f"    {a}-{b}: {sim:.4f} {status}")

        # 保存阶段完成标记
        self.current_epoch = epochs
        self.save_checkpoint(f"stage_{stage_name}")
        self.save_checkpoint("latest")

    # ── 主训练流程（V3: 更好的续训逻辑） ──
    def train(self):
        self.build_model()

        stages = list(config.stage_epochs.keys())

        # 尝试断点续训
        resumed = self.load_checkpoint()
        start_stage_idx = 0
        start_epoch = 0

        if resumed and self.current_stage:
            # V3: 从 checkpoint 恢复阶段信息
            try:
                start_stage_idx = stages.index(self.current_stage)
            except ValueError:
                logger.warning(f"  未知阶段 {self.current_stage}，从头训练")
                start_stage_idx = 0

            # 检查当前阶段是否已全部完成
            stage_eps = config.stage_epochs.get(self.current_stage, 0)
            if self.current_epoch >= stage_eps:
                # 当前阶段已完成，但跨文件续训链路未保存 stage_*.pt → 从下一阶段开始
                logger.info(f"  阶段 [{self.current_stage}] 已完成，从下一阶段开始")
                start_stage_idx += 1
                self.current_epoch = 0
            else:
                start_epoch = self.current_epoch
                logger.info(
                    f"  从阶段 [{self.current_stage}] epoch {start_epoch+1}/{stage_eps} 续训"
                )

        logger.info("=" * 60)
        logger.info("课程学习训练计划 (V3):")
        for i, name in enumerate(stages):
            ep = config.stage_epochs[name]
            if i < start_stage_idx:
                marker = " ✓ 已完成"
            elif i == start_stage_idx and start_epoch == 0:
                marker = " ← 开始"
            elif i == start_stage_idx:
                marker = f" ← 续训 (epoch {start_epoch+1}/{ep})"
            else:
                marker = ""
            logger.info(f"  {name}: {ep} epochs{marker}")
        logger.info("=" * 60)

        for idx in range(start_stage_idx, len(stages)):
            stage_name = stages[idx]
            logger.info(f"\n{'='*40}")
            logger.info(f"阶段: {stage_name}")
            logger.info(f"{'='*40}")

            texts = load_stage_data(stage_name)
            epochs = config.stage_epochs[stage_name]

            # 仅第一个阶段可能从部分 epoch 续训，后续阶段从 0 开始
            if idx == start_stage_idx and start_epoch > 0:
                self.current_epoch = start_epoch
            else:
                self.current_epoch = 0

            self.train_stage(stage_name, texts, epochs)

        self.save_checkpoint("final")
        logger.info(f"\n训练完成! 总步数: {self.step}, 最佳Loss: {self.best_loss:.4f}")


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