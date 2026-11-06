# -*- coding: utf-8 -*-
"""
probe_width_vs_glyph.py — 验证"不把 1024 压成 64，字形信息保留更多"假说
================================================================================
用冻结的 50K Spatial 权重，把字形表征按压缩程度切成 4 档，
测每档对硬骨头形近字的分离度：

  L0 原始像素    : glyph_matrix (1024 维, 冻结 buffer, 未编码)
  L1 卷积特征    : conv 之后 (4096 维, 未压缩)
  L2 投影中间层  : conv + proj[:-1] (256 维, 最终 Linear 之前)
  L3 最终输出    : 完整 point_encoder (64 维, 现喂给 LM)

指标：
  - 重点形近字对平均 cosine（越低 = 字形越分得开）
  - 全局随机基线（前 500 高频字两两 cosine 均值）
  - 倍率 = 重点对均值 / 随机基线（越高 = 字形意识越强）

零成本：不训练，纯前向推理。直接回答用户"不压成 64 直接训"的直觉。
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step050000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
DEVICE = torch.device("cpu")

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"),
    ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"),
    ("王", "玉"),
    ("日", "曰"),
]
EXTRA_GROUPS = [
    ["人", "入", "八"], ["土", "士"], ["天", "夫"],
    ["日", "目", "白"], ["田", "由", "甲", "申"],
    ["王", "玉", "主"], ["力", "刀"], ["乌", "鸟"],
    ["问", "间", "闻"], ["风", "凤"],
]


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_all():
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
    print(f"[Load] 点阵矩阵: {loaded}/{vocab_size}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[Load] Checkpoint: {os.path.basename(CKPT_PATH)}")
    return model, char_to_idx, idx_to_char, glyph_matrix


@torch.no_grad()
def extract_levels(model, glyph_matrix):
    """返回 4 档表征矩阵 (V, dim) numpy float64。"""
    V = glyph_matrix.shape[0]
    gm_t = model.glyph_matrix  # 已是注册 buffer

    # L0 原始像素
    L0 = glyph_matrix.astype(np.float64)

    # L1/L2/L3 需要前向
    g4 = gm_t.reshape(V, 1, 32, 32)
    conv_feat = model.point_encoder.conv(g4)            # (V,4,32,32)
    conv_flat = conv_feat.reshape(V, -1)               # (V,4096)  = L1
    L1 = conv_flat.numpy().astype(np.float64)

    # L2 = proj 去掉最后一层 Linear(256->64)
    x = conv_flat
    for layer in list(model.point_encoder.proj)[:-1]:
        x = layer(x)
    L2 = x.numpy().astype(np.float64)                 # (V,256)

    # L3 = 完整 point_encoder 输出
    L3 = model.point_encoder(gm_t).numpy().astype(np.float64)  # (V,64)

    return {"L0_raw1024": L0, "L1_conv4096": L1,
            "L2_proj256": L2, "L3_out64": L3}


def pair_cos(mat, a, b, char_to_idx):
    ia, ib = char_to_idx.get(a, 0), char_to_idx.get(b, 0)
    return cosine_sim(mat[ia], mat[ib])


def global_baseline(mat, char_to_idx, idx_to_char, n=500):
    top = []
    for idx in range(1, mat.shape[0]):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>" and len(ch) == 1:
            try:
                gb = ch.encode("gb2312")
                if len(gb) == 2 and gb[0] >= 0xB0:
                    top.append(ch)
            except UnicodeEncodeError:
                continue
        if len(top) >= n:
            break
    vecs = np.stack([mat[char_to_idx[c]] for c in top], 0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sm = (vecs / norms) @ (vecs / norms).T
    iu = np.triu_indices(len(top), k=1)
    return float(sm[iu].mean())


def main():
    model, char_to_idx, idx_to_char, glyph_matrix = load_all()
    levels = extract_levels(model, glyph_matrix)

    print("\n" + "=" * 70)
    print("字形分离度 vs 压缩程度（50K 冻结权重，越低=越分得开）")
    print("=" * 70)
    print(f"{'表征档':<14} {'重点对均cos':>12} {'随机基线':>12} {'倍率':>8} {'判读':>10}")
    print("-" * 70)

    result = {}
    for name, mat in levels.items():
        fps = [pair_cos(mat, a, b, char_to_idx) for a, b in FOCUS_PAIRS]
        avg_fp = float(np.mean(fps))
        gb = global_baseline(mat, char_to_idx, idx_to_char)
        ratio = avg_fp / (gb + 1e-8)
        verdict = ("更分开" if ratio > 17 else ("持平" if ratio > 14 else "更糊"))
        print(f"{name:<14} {avg_fp:>12.4f} {gb:>12.4f} {ratio:>7.2f}x {verdict:>10}")
        result[name] = {
            "focus_avg_cos": avg_fp,
            "global_baseline": gb,
            "ratio": ratio,
            "focus_pairs": {f"{a}/{b}": pair_cos(mat, a, b, char_to_idx)
                            for a, b in FOCUS_PAIRS},
        }

    # 硬骨头逐对：L0 vs L3 对照
    print("\n" + "=" * 70)
    print("硬骨头逐对 cosine：原始像素(L0) vs 当前64维(L3)")
    print("=" * 70)
    print(f"{'字对':<10} {'L0像素':>10} {'L1卷积':>10} {'L2_256':>10} {'L3_64':>10}")
    print("-" * 50)
    for a, b in FOCUS_PAIRS + [("王", "玉"), ("风", "凤"), ("干", "千")]:
        if (a, b) in FOCUS_PAIRS or (b, a) in FOCUS_PAIRS or (a, b) in [("王","玉"),("风","凤"),("干","千")]:
            c0 = pair_cos(levels["L0_raw1024"], a, b, char_to_idx)
            c1 = pair_cos(levels["L1_conv4096"], a, b, char_to_idx)
            c2 = pair_cos(levels["L2_proj256"], a, b, char_to_idx)
            c3 = pair_cos(levels["L3_out64"], a, b, char_to_idx)
            print(f"{a}/{b:<8} {c0:>10.4f} {c1:>10.4f} {c2:>10.4f} {c3:>10.4f}")

    out = {
        "ckpt": "ckpt_step050000.pt",
        "note": "冻结权重下，字形表征压缩程度 vs 分离度；验证'不压成64'直觉",
        "levels": result,
    }
    out_path = os.path.join(BASE_DIR, "probe_width_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
