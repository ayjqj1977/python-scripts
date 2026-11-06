# -*- coding: utf-8 -*-
"""V11 测试脚本 — 生成采样 + 形近字判别 + PPL 估算"""
import json, os, sys, math, torch, torch.nn.functional as F
import numpy as np

CKPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CKPT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 256
VOCAB_SIZE = 7988
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

from model_v10 import NeuroGlyph34V10

char_to_idx = json.load(open(os.path.join(CKPT_DIR, "char_to_idx.json"), encoding="utf-8"))
idx_to_char = {int(v): k for k, v in char_to_idx.items()}

model = NeuroGlyph34V10(
    vocab_size=VOCAB_SIZE, d_glyph=D_GLYPH, d_semantic=D_SEMANTIC,
    d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
    d_ff=D_FF, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
)
embedding_matrix = np.load(os.path.join(CKPT_DIR, "embedding_matrix.npy"))
aligned = np.zeros((VOCAB_SIZE, embedding_matrix.shape[1]), dtype=np.float32)
aligned[1:] = embedding_matrix
model.load_glyph_matrix(aligned)

ckpt = torch.load(os.path.join(CKPT_DIR, "checkpoints_v11", "latest.pt"), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.to(DEVICE)
model.eval()
print(f"模型加载完成 | 设备: {DEVICE}\n")

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

prompts = ["春日", "根据中华人民", "计算机科学", "从前有座山", "教育是"]
print("=" * 60)
print("生成采样")
print("=" * 60)
for p in prompts:
    out = generate(p, length=80, temp=0.8)
    print(f"\n提示: {p}")
    print(f"生成: {out}")

pairs = [
    ("未", "末"), ("已", "己"), ("干", "千"), ("土", "士"),
    ("人", "入"), ("大", "太"), ("本", "木"), ("困", "因"),
    ("侯", "候"), ("辨", "辩"), ("徒", "徙"), ("浙", "淅"),
    ("爪", "瓜"), ("券", "卷"), ("灸", "炙"), ("栗", "粟"),
    ("日", "曰"), ("刀", "力"),
]

glyph = model.glyph_matrix.to(DEVICE)
glyph_norm = F.normalize(glyph, dim=-1)

print(f"\n{'='*60}")
print("形近字判别 (cosine越低=区分越好)")
print("=" * 60)
easy = mid = hard = 0
for a, b in pairs:
    if a not in char_to_idx or b not in char_to_idx:
        continue
    ia, ib = char_to_idx[a], char_to_idx[b]
    cos = (glyph_norm[ia] * glyph_norm[ib]).sum().item()
    if cos < 0.7: easy += 1; label = "✓"
    elif cos < 0.85: mid += 1; label = "△"
    else: hard += 1; label = "✗"
    print(f"  {a}-{b}: cos={cos:.4f} {label}")

print(f"\n汇总: 易区分 {easy}, 一般 {mid}, 难区分 {hard}")

