# -*- coding: utf-8 -*-
"""
fuse_struct_pixel.py — 验证"手工结构码(去拼音) 与 像素64维融合"对字形分离度的影响
======================================================================================
用户提议：34维手工码去掉拼音那六位做"头部"，和 1024 像素压缩后的 64 维融合。

本脚本零成本（冻结 50K 权重），测三种表示对硬骨头形近字的分离度：
  P  : 像素64维 (point_encoder 输出，现状)
  S  : 手工结构码12维 (18维设计码去拼音6位：四角5+五笔4+声调1+类型2)，纯结构先验
  F1 : concat([P_norm, S_norm]) -> L2归一化 (76维)
  F2 : L2norm(P_norm + 0.5 * 随机正交映射(S->64)) (64维，同维可公平对比 P)

重点回答：
  1. 手工结构码本身(S)能不能分硬骨头？ -> 若不能，融合也带不来分离度
  2. 融合(F1/F2)相对纯像素(P)分离度变了没？ -> 验证"融合改善字形"假说
注意：concat 不动像素几何，理论上分离度由 P 主导；本实验用数据确认。
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
TABLE_PATH = "D:/desktop/学习结果小文件/1_18维向量表/汉字向量字库_扩展.json"
DEVICE = torch.device("cpu")

# 18维设计码维度定义：1-5四角 / 6-9五笔 / 10-15拼音 / 16声调 / 17-18类型
# 去掉拼音(索引9-14)后保留：0-8(四角+五笔) + 15-17(声调+类型) = 12维
KEEP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17]

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"),
    ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"),
    ("王", "玉"),
    ("日", "曰"),
    ("王", "主"), ("风", "凤"), ("人", "入"), ("土", "士"), ("天", "夫"),
]


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l2norm(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    return mat / n


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


def load_struct_table():
    d = json.load(open(TABLE_PATH, encoding="utf-8"))
    chars = d["characters"]
    m = {}
    miss = 0
    for it in chars:
        ch = it["char"]
        v = it.get("vector")
        if isinstance(v, list) and len(v) == 18:
            arr = np.array(v, dtype=np.float64)
            if np.all(arr == 0) or np.all(arr[5:9] == 25):
                miss += 1  # 缺码占位，不计入有效
                continue
            m[ch] = arr
    print(f"[Load] 18维设计表: {len(m)} 字有效码, {miss} 缺码跳过")
    return m


@torch.no_grad()
def extract_pixel64(model, glyph_matrix):
    V = glyph_matrix.shape[0]
    gm_t = model.glyph_matrix
    P = model.point_encoder(gm_t).numpy().astype(np.float64)  # (V,64)
    return P


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
    vecs = l2norm(vecs)
    sm = vecs @ vecs.T
    iu = np.triu_indices(len(top), k=1)
    return float(sm[iu].mean())


def main():
    model, char_to_idx, idx_to_char, glyph_matrix = load_all()
    struct_map = load_struct_table()

    # 像素 64 维
    P = extract_pixel64(model, glyph_matrix)
    Pn = l2norm(P)

    # 手工结构码 12 维（去拼音）
    V = P.shape[0]
    S = np.zeros((V, 12), dtype=np.float64)
    hit = 0
    for idx in range(V):
        ch = idx_to_char.get(idx, "")
        if ch in struct_map:
            S[idx] = struct_map[ch][KEEP_IDX]
            hit += 1
    print(f"[Fuse] 结构码对齐命中: {hit}/{V}")
    Sn = l2norm(S)

    # 融合 F1: concat
    F1 = l2norm(np.concatenate([Pn, Sn], axis=1))  # (V,76)

    # 融合 F2: 加法映射（固定随机正交，仅维度映射，不训练）-> 64维
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 12))
    W, _ = np.linalg.qr(W)  # 64x12 正交
    Sproj = Sn @ W.T  # (V,64)
    F2 = l2norm(Pn + 0.5 * Sproj)

    reps = {"P_像素64": Pn, "S_结构码12": Sn, "F1_concat76": F1, "F2_加法64": F2}

    print("\n" + "=" * 72)
    print("融合对比：手工结构码(去拼音) vs 像素64维 — 字形分离度")
    print("=" * 72)
    print(f"{'表示':<14} {'重点对均cos':>12} {'随机基线':>12} {'倍率':>8}")
    print("-" * 54)
    result = {}
    for name, mat in reps.items():
        fps = [pair_cos(mat, a, b, char_to_idx) for a, b in FOCUS_PAIRS]
        avg_fp = float(np.mean(fps))
        gb = global_baseline(mat, char_to_idx, idx_to_char)
        ratio = avg_fp / (gb + 1e-8)
        print(f"{name:<14} {avg_fp:>12.4f} {gb:>12.4f} {ratio:>7.2f}x")
        result[name] = {
            "focus_avg_cos": avg_fp,
            "global_baseline": gb,
            "ratio": ratio,
            "focus_pairs": {f"{a}/{b}": pair_cos(mat, a, b, char_to_idx)
                            for a, b in FOCUS_PAIRS},
        }

    # 硬骨头逐对：S vs P vs F2
    print("\n" + "=" * 72)
    print("硬骨头逐对 cosine：结构码(S) / 像素(P) / 加法融合(F2)")
    print("=" * 72)
    print(f"{'字对':<10} {'S_结构12':>10} {'P_像素64':>10} {'F2_加法64':>11}")
    print("-" * 44)
    for a, b in FOCUS_PAIRS:
        cs = pair_cos(Sn, a, b, char_to_idx)
        cp = pair_cos(Pn, a, b, char_to_idx)
        cf = pair_cos(F2, a, b, char_to_idx)
        print(f"{a}/{b:<8} {cs:>10.4f} {cp:>10.4f} {cf:>11.4f}")

    out = {
        "ckpt": "ckpt_step050000.pt",
        "note": "验证手工结构码(去拼音)与像素64维融合；冻结权重零成本",
        "keep_idx": KEEP_IDX,
        "reps": result,
    }
    out_path = os.path.join(BASE_DIR, "fuse_struct_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
