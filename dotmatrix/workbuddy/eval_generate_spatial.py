# -*- coding: utf-8 -*-
"""
eval_generate_spatial.py — PointMatrixSpatial 30K 文本生成（纯娱乐）
环境: D:/MICODA/envs/py312_torch/python.exe
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step030000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
DEVICE = torch.device("cpu")

PROMPTS = [
    "人工智能",
    "今天天气",
    "中国历史悠久",
    "汉字的字形",
    "深度学习是",
    "床前明月光",
]


def load_model():
    with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    model = PointMatrixSpatialModel(vocab_size=vocab_size)
    reader = HZK32Reader(HZK32_PATH)
    glyph_matrix = np.zeros((vocab_size, 1024), dtype=np.float32)
    loaded = 0
    for idx in range(vocab_size):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                glyph_matrix[idx] = reader.get_glyph_vector(ch).astype(np.float32)
                loaded += 1
            except (ValueError, IOError):
                pass
    model.load_glyph_matrix(glyph_matrix)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    model.to(DEVICE)
    return model, char_to_idx, idx_to_char


@torch.no_grad()
def generate(model, prompt, char_to_idx, idx_to_char, max_new=60, temperature=0.9, top_k=40):
    unk_id = char_to_idx.get("<UNK>", 0)
    ids = [char_to_idx.get(ch, unk_id) for ch in prompt]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated = list(ids)
    for _ in range(max_new):
        seq = input_ids[:, -model.max_seq_len:]
        logits = model(seq)[0, -1, :] / temperature
        if top_k > 0:
            topk_vals, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask[topk_indices] = topk_vals
            logits = mask
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == 0:
            break
        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=DEVICE)
    return "".join(idx_to_char.get(i, "<UNK>") for i in generated)


def main():
    model, char_to_idx, idx_to_char = load_model()
    print("=" * 60)
    print("PointMatrixSpatial 30K — 文本生成（看看有多离谱）")
    print("=" * 60)
    for p in PROMPTS:
        out = generate(model, p, char_to_idx, idx_to_char)
        print(f"\n[Prompt] {p}")
        print(f"[生成  ] {out}")
        print("-" * 50)


if __name__ == "__main__":
    main()
