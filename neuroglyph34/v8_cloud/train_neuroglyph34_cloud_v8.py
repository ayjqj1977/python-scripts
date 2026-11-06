"""
NeuroGlyph-34 Cloud V9 — 34维字形(结构化) + 64维字义(可训练) | 混合嵌入
========================================================================
V9 改进:
  1. 嵌入: 34维CNN字形(加载现有embedding_matrix.npy,冻结) + 64维可训练字义Embedding
  2. 输入: 98维 → fusion: 480维 → d_fusion=512, 8头, d_ff=2048
  3. 上下文: 512 (V8的384 → V9: 512)
  4. 语料: corpus_modern.txt (纯现代文, 1335万行) + V9语料清洗
  5. 有效batch: 192 (8×24)
"""
import os, sys, json, math, time, random, logging, argparse, warnings, glob, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = project_dir
    checkpoint_dir = os.path.join(project_dir, "checkpoints_v8")
    log_dir = os.path.join(project_dir, "logs_v8")

    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"
    corpus_file = "corpus_modern.txt"

    d_input = 34  # 结构化嵌入维度(不变)
    semantic_dim = 64  # V8新增: 可训练字义嵌入
    vocab_size = 7987
    special_tokens = 4

    cnn_channels = [64, 64, 64, 64, 64]
    cnn_kernels = [2, 2, 2, 3, 3]
    embed_dims = [16, 16, 16, 16, 32]

    total_cnn = sum(cnn_channels)       # 320
    total_emb = sum(embed_dims)         # 96
    fusion_input = total_cnn + total_emb + semantic_dim  # 320 + 96 + 64 = 480

    # V8: 架构参数
    d_fusion = 512
    n_layers = 12
    n_heads = 8        # 512/8=64, 整除
    d_ff = 2048         # 4x
    dropout = 0.15
    max_seq_len = 384   # V6: 256 → V8: 384

    micro_batch = 6             # V6: 8 → V8: 6 (更多参数)
    gradient_accumulation = 20  # V6: 16 → V8: 20 (有效batch=120)
    lr = 1.5e-4                 # V6: 1e-4 → V8: 1.5e-4
    warmup_steps = 4000         # V6: 3000 → V8: 4000
    weight_decay = 0.1
    grad_clip = 1.0

    save_interval = 5000
    checkpoint_keep = 3
    use_amp = True

    n_epochs = 30               # V6: 25 → V8: 30
    val_split = 0.05
    max_corpus_lines = 14_000_000

    sample_prompts = [
        "人工智能是", "中国的发展", "量子计算是",
        "光合作用是", "第二次世界大战", "全球变暖是指",
        "人类基因组", "太阳系的行星", "DNA双螺旋结构",
        "互联网的诞生",
    ]
    sample_max_len = 64
    sample_top_k = 40
    sample_temperature = 0.8

    sim_char_weight = 0.02
    sim_char_margin = 0.3

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

    total_steps = 400_000

config = Config()


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


class NeuroGlyph34V9(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # --- 结构化部分 (同V6, 加载预训练embedding) ---
        cnn_slices = [
            (0, 5), (5, 9), (10, 13), (16, 26), (26, 32),
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

        # --- V8新增: 可训练字义嵌入 ---
        self.semantic_embed = nn.Embedding(cfg.vocab_size, cfg.semantic_dim)

        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_input, cfg.d_fusion),
            nn.LayerNorm(cfg.d_fusion),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.pos_encoding = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_fusion))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(1, 1, cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool(),
            persistent=False
        )

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
                if m is not self.semantic_embed:  # semantic_embed 单独初始化
                    nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # 字义嵌入用较小标准差初始化,避免初始噪声过大
        nn.init.normal_(self.semantic_embed.weight, mean=0, std=0.01)

    def forward(self, x_34, token_ids):
        """x_34: (B, T, 34) 结构化嵌入, token_ids: (B, T)"""
        B, T, _ = x_34.shape

        # CNN + 离散嵌入 (同V6)
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x_34[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)

        emb_outputs = [
            self.emb_total_strokes(x_34[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x_34[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x_34[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x_34[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x_34[:, :, 32] * 16 + x_34[:, :, 33]).long().clamp(0, 255)),
        ]

        # V8: 拼接字义嵌入
        semantic = self.semantic_embed(token_ids)  # (B, T, 64)

        merged = torch.cat(cnn_outputs + emb_outputs + [semantic], dim=-1)  # 480维
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]

        causal_mask = self.causal_mask[:, :, :T, :T]
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    def get_hidden(self, x_34, token_ids):
        B, T, _ = x_34.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x_34[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)
        emb_outputs = [
            self.emb_total_strokes(x_34[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x_34[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x_34[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x_34[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x_34[:, :, 32] * 16 + x_34[:, :, 33]).long().clamp(0, 255)),
        ]
        semantic = self.semantic_embed(token_ids)
        merged = torch.cat(cnn_outputs + emb_outputs + [semantic], dim=-1)
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]
        causal_mask = self.causal_mask[:, :, :T, :T]
        hidden = self.transformer(fused, mask=causal_mask)
        return self.ln_final(hidden)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==================== 数据集 ====================
class NeuroGlyphDataset(Dataset):
    def __init__(self, texts, char_to_idx, embedding_matrix, seq_len=384, stride=None):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.embedding_matrix = torch.FloatTensor(embedding_matrix)

        all_tokens = []
        for text in texts:
            for ch in text:
                if ch in char_to_idx:
                    all_tokens.append(char_to_idx[ch])

        self.tokens = torch.LongTensor(all_tokens)
        if stride is None:
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

        feat_seq = self.embedding_matrix[token_seq]  # 34维结构化特征
        return feat_seq, token_seq, target_seq


def create_train_val_loaders(texts, char_to_idx, embedding_matrix, batch_size, seq_len=512, val_split=0.05):
    if len(texts) < 100:
        return None, None

    n_texts = len(texts)
    n_val_texts = max(1, int(n_texts * val_split))
    n_train_texts = n_texts - n_val_texts

    train_texts = texts[:n_train_texts]
    val_texts = texts[n_train_texts:]

    train_ds = NeuroGlyphDataset(train_texts, char_to_idx, embedding_matrix, seq_len, stride=seq_len // 2)
    val_ds = NeuroGlyphDataset(val_texts, char_to_idx, embedding_matrix, seq_len, stride=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        self.best_loss = float("inf")

        # 加载词表 & 嵌入矩阵
        with open(os.path.join(cfg.data_dir, cfg.vocab_file), "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.embedding_matrix = np.load(os.path.join(cfg.data_dir, cfg.embedding_file))
        logger.info(f"词表: {len(self.char_to_idx)}, 嵌入: {self.embedding_matrix.shape}")

        # 形近字对
        self.similar_pairs = []
        for a, b in cfg.similar_char_pairs:
            if a in self.char_to_idx and b in self.char_to_idx:
                self.similar_pairs.append((a, b))
        logger.info(f"形近字测试对: {len(self.similar_pairs)}/{len(cfg.similar_char_pairs)}")

    def build_model(self):
        self.model = NeuroGlyph34V9(self.cfg).to(self.device)
        total, trainable = self.model.param_count()
        logger.info(f"模型参数: {total:,} (可训练: {trainable:,})")

        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.cfg.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.cfg.total_steps - self.cfg.warmup_steps, eta_min=self.cfg.lr * 0.01)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[self.cfg.warmup_steps])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scaler = GradScaler('cuda') if self.cfg.use_amp else None

    @torch.no_grad()
    def validate(self, val_loader):
        if val_loader is None:
            return None, None
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for feat_seq, token_seq, target_seq in val_loader:
            feat_seq = feat_seq.to(self.device)
            token_seq = token_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            logits = self.model(feat_seq, token_seq)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            valid_tokens = (target_seq != 0).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
        self.model.train()
        if total_tokens == 0:
            return None, None
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    @torch.no_grad()
    def generate_sample(self):
        self.model.eval()
        results = []
        max_len = self.cfg.max_seq_len
        top_k = self.cfg.sample_top_k
        temperature = self.cfg.sample_temperature

        for prompt in self.cfg.sample_prompts:
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
                seq = torch.LongTensor(input_tokens).unsqueeze(0)  # (1, T)
                emb_mat = torch.from_numpy(self.embedding_matrix).float().to(self.device)
                feat = emb_mat[seq]  # (1, T, 34)
                logits = self.model(feat, seq.to(self.device))
                next_logits = logits[0, -1, :] / temperature
                topk_vals, topk_indices = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                probs = F.softmax(topk_vals, dim=0)
                chosen = torch.multinomial(probs, 1).item()
                next_token = topk_indices[chosen].item()
                if next_token == 0:
                    break
                gen_char = self.idx_to_char.get(next_token, "?")
                generated.append(gen_char)
                input_tokens.append(next_token)
                input_tokens = input_tokens[-max_len:]

            results.append("".join(generated))

        clean_results = []
        for r in results:
            for stop_pattern in ["{'", "':'", "','"]:
                idx = r.find(stop_pattern)
                if idx > 0:
                    r = r[:idx]
            clean_results.append(r)

        self.model.train()
        return clean_results

    @torch.no_grad()
    def test_similar_chars(self):
        self.model.eval()
        all_chars = set()
        for a, b in self.similar_pairs:
            all_chars.add(a); all_chars.add(b)

        char_hidden = {}
        for ch in all_chars:
            idx = self.char_to_idx[ch]
            tokens = torch.LongTensor([idx, idx]).unsqueeze(0).to(self.device)
            emb_mat = torch.from_numpy(self.embedding_matrix).float().to(self.device)
            feat = emb_mat[tokens]
            hidden = self.model.get_hidden(feat, tokens)
            char_hidden[ch] = hidden[0, -1, :]

        similarities = []
        for a, b in self.similar_pairs:
            if a in char_hidden and b in char_hidden:
                sim = F.cosine_similarity(
                    char_hidden[a].unsqueeze(0), char_hidden[b].unsqueeze(0)
                ).item()
                similarities.append((a, b, sim))

        similarities.sort(key=lambda x: x[2], reverse=True)
        self.model.train()
        return similarities

    def compute_similar_char_loss(self):
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
            hidden = self.model.get_hidden(feat, tokens)
            char_hidden[ch] = hidden[0, -1, :]

        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for a, b in self.similar_pairs:
            if a in char_hidden and b in char_hidden:
                sim = F.cosine_similarity(
                    char_hidden[a].unsqueeze(0), char_hidden[b].unsqueeze(0)
                ).squeeze()
                loss += torch.clamp(sim - self.cfg.sim_char_margin, min=0)
                count += 1

        self.model.train()
        return loss / max(count, 1)

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

        all_ckpts = sorted(
            glob.glob(os.path.join(self.cfg.checkpoint_dir, "step_*.pt")),
            key=os.path.getmtime, reverse=True,
        )
        for old in all_ckpts[self.cfg.checkpoint_keep:]:
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

    def train(self):
        self.build_model()

        # 加载语料
        corpus_path = os.path.join(self.cfg.data_dir, self.cfg.corpus_file)
        logger.info(f"加载语料: {corpus_path}")
        texts = []
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= self.cfg.max_corpus_lines:
                    break
                ls = line.strip()
                if ls:
                    texts.append(ls)
                if (i + 1) % 2_000_000 == 0:
                    logger.info(f"  已加载 {i+1:,} 行")

        logger.info(f"语料: {len(texts):,} 行")
        random.shuffle(texts)

        train_loader, val_loader = create_train_val_loaders(
            texts, self.char_to_idx, self.embedding_matrix,
            self.cfg.micro_batch, self.cfg.max_seq_len, self.cfg.val_split
        )
        logger.info(f"训练: {len(train_loader.dataset)} 条, 验证: {len(val_loader.dataset) if val_loader else 0} 条")
        logger.info(f"每 epoch: {len(train_loader)} batch, 有效 batch: {self.cfg.micro_batch * self.cfg.gradient_accumulation}")

        # 尝试续训
        resumed = self.load_checkpoint()
        if resumed:
            logger.info(f"续训: step={self.step}, best_loss={self.best_loss:.4f}")

        logger.info("=" * 60)
        logger.info(f"V8 训练: {self.cfg.n_epochs} epochs, 34+64混合嵌入")
        logger.info("=" * 60)

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            train_iter = iter(train_loader)

            while True:
                self.optimizer.zero_grad()
                micro_losses = []

                for _ in range(self.cfg.gradient_accumulation):
                    try:
                        feat_seq, token_seq, target_seq = next(train_iter)
                    except StopIteration:
                        break

                    feat_seq = feat_seq.to(self.device)
                    token_seq = token_seq.to(self.device)
                    target_seq = target_seq.to(self.device)

                    if self.scaler:
                        with autocast('cuda'):
                            logits = self.model(feat_seq, token_seq)
                            loss = self.criterion(
                                logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1)
                            ) / self.cfg.gradient_accumulation
                        self.scaler.scale(loss).backward()
                    else:
                        logits = self.model(feat_seq, token_seq)
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1)
                        ) / self.cfg.gradient_accumulation
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
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e} | E{epoch+1}/{self.cfg.n_epochs}"
                    )

                if self.step % self.cfg.save_interval == 0 and self.step > 0:
                    self.save_checkpoint(f"step_{self.step}")

            avg_loss = epoch_loss / max(batch_count, 1)

            val_loss, val_ppl = self.validate(val_loader)
            val_str = ""
            if val_loss is not None:
                val_str = f" | Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"

            sim_loss = None
            if self.cfg.sim_char_weight > 0:
                self.optimizer.zero_grad()
                sim_loss = self.compute_similar_char_loss()
                weighted = sim_loss * self.cfg.sim_char_weight
                if weighted.item() > 0:
                    if self.scaler:
                        self.scaler.scale(weighted).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        weighted.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        self.optimizer.step()
                sim_str = f" | SimChar Loss: {sim_loss.item():.4f}"
            else:
                sim_str = ""

            best_str = ""
            if val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best")
                best_str = " *** BEST ***"

            logger.info(
                f"=== Epoch {epoch+1}/{self.cfg.n_epochs} | "
                f"Train Loss: {avg_loss:.4f}{val_str}{sim_str}{best_str}"
            )

            # 生成采样
            samples = self.generate_sample()
            logger.info("=" * 60)
            logger.info("生成采样:")
            logger.info("=" * 60)
            for prompt, sample in zip(self.cfg.sample_prompts, samples):
                logger.info(f"  [{prompt}] → {sample}")

            # 形近字测试
            sims = self.test_similar_chars()
            logger.info("=" * 60)
            logger.info("形近字判别力测试:")
            logger.info("=" * 60)
            easy = medium = hard = 0
            logger.info("Top 15 最相似形近字对:")
            for a, b, s in sims[:15]:
                if s < 0.5:
                    label = "✓ 易区分"
                    easy += 1
                elif s < 0.75:
                    label = "~ 一般"
                    medium += 1
                else:
                    label = "✗ 难区分"
                    hard += 1
                logger.info(f"  {a}-{b}: {s:.4f}   {label}")
            for a, b, s in sims[15:]:
                if s < 0.5:
                    easy += 1
                elif s < 0.75:
                    medium += 1
                else:
                    hard += 1
            logger.info(f"汇总: 易区分({easy}), 一般({medium}), 难区分({hard})")

            # 保存epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

        logger.info("训练完成!")


# ==================== 入口 ====================
if __name__ == "__main__":
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    trainer = Trainer(config)
    trainer.train()
