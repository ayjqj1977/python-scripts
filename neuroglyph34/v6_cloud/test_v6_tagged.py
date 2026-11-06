"""
test_v6_tagged.py — 加载 best.pt 做生成采样和形近字判别力测试
与 train_neuroglyph34_cloud_v6.py 的模型架构完全一致
"""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Config ====================
class Config:
    project_dir = BASE_DIR
    data_dir = BASE_DIR
    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"

    d_input = 34
    vocab_size = 7987

    cnn_channels = [64, 64, 64, 64, 64]
    cnn_kernels = [2, 2, 2, 3, 3]
    embed_dims = [16, 16, 16, 16, 32]

    total_cnn = sum(cnn_channels)
    total_emb = sum(embed_dims)
    fusion_input = total_cnn + total_emb

    d_fusion = 516
    n_layers = 12
    n_heads = 12
    d_ff = 2048
    dropout = 0.15
    max_seq_len = 256

    sample_prompts = [
        "人工智能是", "中国的发展", "量子计算是",
        "光合作用是", "第二次世界大战", "全球变暖是指",
        "人类基因组", "太阳系的行星", "DNA双螺旋结构",
        "互联网的诞生",
    ]
    sample_max_len = 64
    sample_top_k = 40
    sample_temperature = 0.8

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

cfg = Config()

# ==================== Model ====================
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
        return self.conv(x)[:, :, :T]

class NeuroGlyph34V6(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        cnn_slices = [(0, 5), (5, 9), (10, 13), (16, 26), (26, 32)]
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
        self.pos_encoding = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_fusion))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_fusion, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln_final = nn.LayerNorm(cfg.d_fusion)
        self.lm_head = nn.Linear(cfg.d_fusion, cfg.vocab_size, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)
        emb_outputs = [
            self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)),
        ]
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    def get_hidden(self, x):
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)
        emb_outputs = [
            self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)),
        ]
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(fused, mask=causal_mask)
        return self.ln_final(hidden)

# ==================== Load & Test ====================
print(f"设备: {DEVICE}")

# Load vocab & embedding
with open(os.path.join(BASE_DIR, cfg.vocab_file), "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}
embedding_matrix = np.load(os.path.join(BASE_DIR, cfg.embedding_file))
print(f"词表: {len(char_to_idx)}, 嵌入: {embedding_matrix.shape}")

# Filter similar pairs
similar_pairs = [(a, b) for a, b in cfg.similar_char_pairs if a in char_to_idx and b in char_to_idx]
print(f"形近字测试对: {len(similar_pairs)}/{len(cfg.similar_char_pairs)}")

# Build model & load checkpoint
model = NeuroGlyph34V6(cfg).to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, "checkpoints_v6", "best.pt"),
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"加载 checkpoint: step={ckpt.get('step', '?')}, best_loss={ckpt.get('best_loss', '?'):.4f}")

# ==================== 生成采样 ====================
print("\n" + "="*60)
print("生成采样:")
print("="*60)
emb_mat = torch.from_numpy(embedding_matrix).float().to(DEVICE)

for prompt in cfg.sample_prompts:
    tokens = [char_to_idx[ch] for ch in prompt if ch in char_to_idx]
    if len(tokens) < 2:
        continue
    input_tokens = tokens[:min(len(tokens), cfg.max_seq_len - cfg.sample_max_len)]
    generated = list(prompt[:len(input_tokens)])

    for _ in range(cfg.sample_max_len):
        seq = torch.LongTensor(input_tokens).unsqueeze(0).to(DEVICE)
        feat = emb_mat[seq]
        logits = model(feat)
        next_logits = logits[0, -1, :] / cfg.sample_temperature
        topk_vals, topk_indices = torch.topk(next_logits, min(cfg.sample_top_k, next_logits.size(0)))
        probs = F.softmax(topk_vals, dim=0)
        chosen = torch.multinomial(probs, 1).item()
        next_token = topk_indices[chosen].item()
        if next_token == 0:
            break
        gen_char = idx_to_char.get(next_token, "?")
        generated.append(gen_char)
        input_tokens.append(next_token)
        input_tokens = input_tokens[-cfg.max_seq_len:]

    text = "".join(generated)
    # Clean dict artifacts
    for stop_pattern in ["{'", "':'", "','"]:
        idx = text.find(stop_pattern)
        if idx > 0:
            text = text[:idx]
    print(f"  [{prompt}] → {text}")

# ==================== 形近字判别力 ====================
print("\n" + "="*60)
print("形近字判别力测试:")
print("="*60)

all_chars = set()
for a, b in similar_pairs:
    all_chars.add(a); all_chars.add(b)

char_hidden = {}
for ch in all_chars:
    idx = char_to_idx[ch]
    tokens = torch.LongTensor([idx, idx]).unsqueeze(0).to(DEVICE)
    feat = emb_mat[tokens]
    hidden = model.get_hidden(feat)
    char_hidden[ch] = hidden[0, -1, :]

sims = []
for a, b in similar_pairs:
    if a in char_hidden and b in char_hidden:
        sim = F.cosine_similarity(char_hidden[a].unsqueeze(0), char_hidden[b].unsqueeze(0)).item()
        sims.append((a, b, sim))

sims.sort(key=lambda x: x[2], reverse=True)
print(f"\nTop 15 最相似形近字对 (越靠上越难区分):")
for a, b, sim in sims[:15]:
    status = "⚠️ 难区分" if sim > 0.85 else ("✓ 易区分" if sim < 0.5 else "  ~ 一般")
    print(f"  {a}-{b}: {sim:.4f} {status}")

# 统计
easy = sum(1 for _, _, s in sims if s < 0.5)
medium = sum(1 for _, _, s in sims if 0.5 <= s <= 0.85)
hard = sum(1 for _, _, s in sims if s > 0.85)
print(f"\n汇总: 易区分({easy}), 一般({medium}), 难区分({hard})")
print("Done.")
