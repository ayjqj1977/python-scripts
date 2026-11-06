# -*- coding: utf-8 -*-
"""V12 测试脚本 — 生成采样 + 形近字判别 + PPL 估算
V12 架构: d_model=768, n_heads=12, d_ff=3072, ~110M 参数
"""
import json, os, sys, math, torch, torch.nn.functional as F
import numpy as np

# V10 目录存放 char_to_idx.json / embedding_matrix.npy / corpus 等共用资源
V10_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neuroglyph34_cloud_v10")
CKPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(V10_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

# ── V12 架构参数 ──
VOCAB_SIZE = 7988
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 768       # V12: 512 → 768
N_LAYERS = 12
N_HEADS = 12        # V12: 8 → 12
D_FF = 3072         # V12: 2048 → 3072
DROPOUT = 0.1
MAX_SEQ_LEN = 256

from model_v10 import NeuroGlyph34V10

# ── 词汇表 & 字形嵌入 ──
char_to_idx = json.load(open(os.path.join(V10_DIR, "char_to_idx.json"), encoding="utf-8"))
idx_to_char = {int(v): k for k, v in char_to_idx.items()}

model = NeuroGlyph34V10(
    vocab_size=VOCAB_SIZE, d_glyph=D_GLYPH, d_semantic=D_SEMANTIC,
    d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
    d_ff=D_FF, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
)
embedding_matrix = np.load(os.path.join(V10_DIR, "embedding_matrix.npy"))
aligned = np.zeros((VOCAB_SIZE, embedding_matrix.shape[1]), dtype=np.float32)
aligned[1:] = embedding_matrix
model.load_glyph_matrix(aligned)

# ── 加载 V12 checkpoint ──
ckpt = torch.load(os.path.join(CKPT_DIR, "latest.pt"), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.to(DEVICE)
model.eval()

total, trainable = model.param_count()
print(f"模型加载完成 | 总参数: {total:,} | 可训练: {trainable:,}\n")


# ═══════════════════════════════════════════════════
# 1. 生成采样
# ═══════════════════════════════════════════════════
@torch.no_grad()
def generate(prompt, length=80, temp=0.8):
    ids = [char_to_idx.get(c, 0) for c in prompt]
    x = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    for _ in range(length):
        if x.shape[1] > MAX_SEQ_LEN:
            x = x[:, -MAX_SEQ_LEN:]
        logits = model(x)
        probs = F.softmax(logits[0, -1, :] / temp, dim=-1)
        tok = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[tok]], device=DEVICE)], dim=1)
    return "".join(idx_to_char.get(t, "?") for t in x[0].tolist())

prompts = [
    "近年来", "根据中华人民", "计算机科学",
    "随着经济", "教育是", "据新华社",
    "人工智能", "金融市场",
]
print("=" * 60)
print("1. 生成采样 (V12, d_model=768, n_heads=12)")
print("=" * 60)
for p in prompts:
    out = generate(p, length=80, temp=0.8)
    print(f"\n  提示: {p}")
    print(f"  生成: {out}")


# ═══════════════════════════════════════════════════
# 2. 形近字判别
# ═══════════════════════════════════════════════════
pairs = [
    ("未", "末"), ("已", "己"), ("干", "千"), ("土", "士"),
    ("人", "入"), ("大", "太"), ("本", "木"), ("困", "因"),
    ("侯", "候"), ("辨", "辩"), ("徒", "徙"), ("浙", "淅"),
    ("爪", "瓜"), ("券", "卷"), ("灸", "炙"), ("栗", "粟"),
    ("日", "曰"), ("刀", "力"),
]

glyph = model.glyph_matrix.to(DEVICE)
sem = model.semantic_embed.weight.to(DEVICE)  # add trained 64-dim
glyph = F.normalize(glyph, dim=-1)
sem = F.normalize(sem, dim=-1)
full_norm = F.normalize(torch.cat([glyph, sem], dim=-1), dim=-1)

print(f"\n{'='*60}")
print("2. 形近字判别 (cosine越低=区分越好)")
print("=" * 60)
easy = mid = hard = 0
for a, b in pairs:
    if a not in char_to_idx or b not in char_to_idx:
        continue
    ia, ib = char_to_idx[a], char_to_idx[b]
    cos = (full_norm[ia] * full_norm[ib]).sum().item()
    if cos < 0.7: easy += 1; label = "✓"
    elif cos < 0.85: mid += 1; label = "△"
    else: hard += 1; label = "✗"
    print(f"  {a}-{b}: cos={cos:.4f} {label}")

print(f"\n  汇总: 易区分 {easy}, 一般 {mid}, 难区分 {hard}")


# ═══════════════════════════════════════════════════
# 3. PPL 估算
# ═══════════════════════════════════════════════════
corpus_path = os.path.join(V10_DIR, "corpus_v10_article.txt")
if os.path.exists(corpus_path):
    print(f"\n{'='*60}")
    print("3. PPL 估算")
    print("=" * 60)
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    articles = text.split("\n\n\n")
    import random
    random.seed(42)
    samples = [a.strip() for a in articles if 50 < len(a.strip()) < 500]
    selected = random.sample(samples, min(100, len(samples)))

    total_loss = 0.0
    total_tokens = 0
    for art in selected:
        ids = [char_to_idx.get(c, 0) for c in art]
        if len(ids) < 4:
            continue
        seq = ids[:MAX_SEQ_LEN + 1]
        x = torch.tensor([seq[:-1]], device=DEVICE, dtype=torch.long)
        y = torch.tensor([seq[1:]], device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    ppl = math.exp(total_loss / max(1, total_tokens))
    print(f"  估算 PPL: {ppl:.2f} (基于 {len(selected)} 篇短文, {total_tokens} tokens)")
else:
    print(f"\n  [跳过] PPL 语料不存在: {corpus_path}")

print(f"\n{'='*60}")
print("V12 测试完成!")
print("=" * 60)
