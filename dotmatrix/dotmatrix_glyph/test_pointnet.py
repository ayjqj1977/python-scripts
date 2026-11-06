# -*- coding: utf-8 -*-
"""
test_pointnet.py — PointMatrix 模型系列测试
============================================
测试 1: 文本生成（3 个 prompt，各续写 50 字）
测试 2: 形近字 glyph 向量对比（训练后 vs HZK32 原始）
测试 3: t-SNE 数据导出（前 500 高频字 64 维向量）
"""

import json
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

# 添加项目目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_pointnet import PointMatrixModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_pointnet", "ckpt_step045000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
OUTPUT_DIR = os.path.join(BASE_DIR)

DEVICE = torch.device("cpu")


# ============================================================
# 工具函数
# ============================================================

def load_model(ckpt_path):
    """加载 checkpoint，返回 model + char_to_idx + idx_to_char"""
    print(f"[Load] 词表: {CHAR_TO_IDX_PATH}")
    with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    print(f"[Load] 模型初始化: vocab_size={vocab_size}")
    model = PointMatrixModel(vocab_size=vocab_size)

    # 加载 HZK32 点阵矩阵
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
    print(f"[Load] 点阵矩阵: {loaded}/{vocab_size} 字符已加载")

    print(f"[Load] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    step = ckpt.get("step", "?")
    print(f"[Load] 训练步数: {step}")
    return model, char_to_idx, idx_to_char


def encode_prompt(text, char_to_idx):
    """将文本编码为 token ids"""
    unk_id = char_to_idx.get("<UNK>", 0)
    return [char_to_idx.get(ch, unk_id) for ch in text]


def decode_ids(ids, idx_to_char):
    """将 token ids 解码为文本"""
    return "".join(idx_to_char.get(i, "<UNK>") for i in ids)


@torch.no_grad()
def generate(model, prompt, char_to_idx, idx_to_char, max_new=50,
             temperature=0.8, top_k=50):
    """自回归生成"""
    ids = encode_prompt(prompt, char_to_idx)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    generated = list(ids)
    for _ in range(max_new):
        # 裁剪到 max_seq_len
        seq = input_ids[:, -model.max_seq_len:]
        logits = model(seq)  # (1, T, V)
        next_logits = logits[0, -1, :] / temperature  # (V,)

        # top-k 过滤
        if top_k > 0:
            topk_vals, topk_indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float("-inf"))
            mask[topk_indices] = topk_vals
            next_logits = mask

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == 0:  # UNK 截断
            break
        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=DEVICE)

    return decode_ids(generated, idx_to_char)


# ============================================================
# 测试 1: 文本生成
# ============================================================

def test_generation(model, char_to_idx, idx_to_char):
    print("\n" + "=" * 60)
    print("测试 1: 文本生成质量")
    print("=" * 60)

    prompts = ["人工智能的发展", "今天天气", "中国历史悠久"]

    for prompt in prompts:
        result = generate(model, prompt, char_to_idx, idx_to_char,
                          max_new=50, temperature=0.8, top_k=50)
        print(f"\n[Prompt] {prompt}")
        print(f"[续写 ] {result[len(prompt):]}")
        print(f"[完整 ] {result}")
        print("-" * 40)


# ============================================================
# 测试 2: 形近字 glyph 向量对比
# ============================================================

def cosine_sim(a, b):
    """余弦相似度"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def test_glyph_comparison(model, char_to_idx, idx_to_char):
    print("\n" + "=" * 60)
    print("测试 2: 形近字 glyph 向量对比")
    print("=" * 60)

    reader = HZK32Reader(HZK32_PATH)

    # 形近字对
    pairs = [
        ("己", "已"), ("己", "巳"), ("已", "巳"),
        ("未", "末"),
        ("干", "千"), ("干", "于"), ("千", "于"),
        ("大", "太"),
        ("王", "玉"),
        ("日", "曰"),
    ]

    # 提取训练后的 glyph 向量 (64 维)
    point_encoder = model.point_encoder
    glyph_buffer = model.glyph_matrix  # (vocab_size, 1024)

    def get_trained_glyph(ch):
        idx = char_to_idx.get(ch, 0)
        raw = glyph_buffer[idx:idx+1]  # (1, 1024)
        with torch.no_grad():
            feat = point_encoder(raw)  # (1, 64)
        return feat.squeeze(0).numpy()

    def get_hzk32_raw(ch):
        """原始 HZK32 1024 维点阵"""
        try:
            return reader.get_glyph_vector(ch).astype(np.float64)
        except (ValueError, IOError):
            return np.zeros(1024, dtype=np.float64)

    print(f"\n{'字对':<10} {'训练后 cos':>12} {'HZK32原始 cos':>14} {'提升':>10}")
    print("-" * 50)

    trained_sims = []
    hzk32_sims = []

    for a, b in pairs:
        # 训练后向量
        va_t = get_trained_glyph(a)
        vb_t = get_trained_glyph(b)
        sim_t = cosine_sim(va_t, vb_t)

        # HZK32 原始向量
        va_h = get_hzk32_raw(a)
        vb_h = get_hzk32_raw(b)
        sim_h = cosine_sim(va_h, vb_h)

        delta = sim_t - sim_h
        print(f"{a}/{b:<8} {sim_t:>12.6f} {sim_h:>14.6f} {delta:>+10.6f}")

        trained_sims.append(sim_t)
        hzk32_sims.append(sim_h)

    avg_t = np.mean(trained_sims)
    avg_h = np.mean(hzk32_sims)
    print("-" * 50)
    print(f"{'平均值':<10} {avg_t:>12.6f} {avg_h:>14.6f} {avg_t-avg_h:>+10.6f}")
    print(f"\n结论: 训练后形近字 glyph 向量平均余弦相似度 {avg_t:.4f}，"
          f"HZK32 原始 {avg_h:.4f}，提升 {avg_t-avg_h:+.4f}")


# ============================================================
# 测试 3: t-SNE 数据导出
# ============================================================

def test_tsne_export(model, char_to_idx, idx_to_char):
    print("\n" + "=" * 60)
    print("测试 3: t-SNE 数据导出（前 500 高频字）")
    print("=" * 60)

    # 使用 GB2312 一级汉字作为高频字（前 500 个非 UNK 汉字）
    # char_to_idx 基本按 GB2312 频率排序，取 idx 1~500 的字符
    top_chars = []
    for idx in range(1, 6726):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>" and len(ch) == 1:
            # 过滤标点：只保留汉字 (GB2312 汉字区码 B0-F7)
            try:
                gb = ch.encode("gb2312")
                if len(gb) == 2 and gb[0] >= 0xB0:
                    top_chars.append(ch)
            except UnicodeEncodeError:
                continue
        if len(top_chars) >= 500:
            break

    print(f"选取前 {len(top_chars)} 高频字")

    point_encoder = model.point_encoder
    glyph_buffer = model.glyph_matrix

    vectors = []
    valid_chars = []
    skipped = 0
    for ch in top_chars:
        idx = char_to_idx.get(ch, 0)
        raw = glyph_buffer[idx:idx+1]
        if raw.sum() == 0:
            skipped += 1
            continue  # 无点阵的字符跳过
        with torch.no_grad():
            feat = point_encoder(raw)
        vectors.append(feat.squeeze(0).numpy())
        valid_chars.append(ch)

    vectors = np.stack(vectors, axis=0)  # (N, 64)
    print(f"有效字符: {len(valid_chars)} (跳过 {skipped} 个无点阵字符)")
    print(f"向量形状: {vectors.shape}")

    # 保存
    vec_path = os.path.join(OUTPUT_DIR, "glyph_vectors_top500.npy")
    chars_path = os.path.join(OUTPUT_DIR, "chars_top500.txt")
    np.save(vec_path, vectors)
    with open(chars_path, "w", encoding="utf-8") as f:
        f.write("".join(valid_chars))

    print(f"已保存:")
    print(f"  向量文件: {vec_path}  ({vectors.shape})")
    print(f"  字符列表: {chars_path}  ({len(valid_chars)} 字)")

    return vec_path, chars_path, vectors, valid_chars


# ============================================================
# 主流程
# ============================================================

def main():
    print("PointMatrix 模型系列测试")
    print(f"设备: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")

    # 加载模型
    model, char_to_idx, idx_to_char = load_model(CKPT_PATH)

    # 测试 1: 文本生成
    test_generation(model, char_to_idx, idx_to_char)

    # 测试 2: 形近字对比
    test_glyph_comparison(model, char_to_idx, idx_to_char)

    # 测试 3: t-SNE 导出
    test_tsne_export(model, char_to_idx, idx_to_char)

    print("\n" + "=" * 60)
    print("全部测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
