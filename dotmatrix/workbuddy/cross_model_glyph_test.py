# -*- coding: utf-8 -*-
"""
cross_model_glyph_test.py — 跨模型字形分离/字形相似度对比
=========================================================
对同样的 10 个焦点形近字对，计算多种典型模型的嵌入余弦相似度，
揭示各模型嵌入空间"字形意识"的强弱（高=保留字形相似，低=忽略字形/偏语义）。

模型：
  1. raw HZK32 像素       —— 纯字形参考（上限）
  2. 18维手工特征          —— 纯设计字形特征（四角号码+五笔+拼音…）
  3. NeuroGlyph v10 34维   —— 文本上下文训练的语义嵌入
  4. PointMatrix 45K       —— 学习型字形编码器(PointNet, 全连接)  [需在 torch 环境单独跑 eval_glyph_focus]
  5. PointMatrix Spatial   —— 学习型字形编码器(卷积)              [已在 eval 中得出]

使用环境：D:/MICODA/envs/py312_torch/python.exe (含 numpy)
"""
import json
import os
import numpy as np

STUDY = "D:/desktop/学习结果小文件"
SPATIAL = os.path.join(STUDY, "dotmatrix_glyph_spatial")

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"),
    ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"),
    ("王", "玉"),
    ("日", "曰"),
]


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- 1. raw HZK32 像素（需 reader）----------
def raw_hzk32_focus():
    sys.path.insert(0, SPATIAL)
    from hzk32_reader import HZK32Reader
    reader = HZK32Reader(os.path.join(STUDY, "dotmatrix_glyph", "HZK32"))
    sims = []
    for a, b in FOCUS_PAIRS:
        va = reader.get_glyph_vector(a).astype(np.float64)
        vb = reader.get_glyph_vector(b).astype(np.float64)
        sims.append(cos(va, vb))
    return float(np.mean(sims))


# ---------- 2. 18维手工特征 ----------
def model_18dim_focus():
    fp = os.path.join(STUDY, "1_18维向量表", "汉字向量字库_扩展.json")
    d = json.load(open(fp, encoding="utf-8"))
    vecs = {}
    for item in d["characters"]:
        vecs[item["char"]] = np.asarray(item["vector"], dtype=np.float64)
    sims = []
    missing = []
    for a, b in FOCUS_PAIRS:
        if a in vecs and b in vecs:
            sims.append(cos(vecs[a], vecs[b]))
        else:
            missing.append(f"{a}/{b}")
    if missing:
        print("  [18维] 缺失字对:", missing)
    return float(np.mean(sims)), vecs


# ---------- 3. NeuroGlyph v10 34维 ----------
def model_v10_focus():
    em = np.load(os.path.join(STUDY, "neuroglyph34_cloud_v10", "embedding_matrix.npy"))
    c2i = json.load(open(os.path.join(STUDY, "neuroglyph34_cloud_v10", "char_to_idx.json"), encoding="utf-8"))
    sims = []
    missing = []
    for a, b in FOCUS_PAIRS:
        if a in c2i and b in c2i:
            va = em[c2i[a]]
            vb = em[c2i[b]]
            sims.append(cos(va, vb))
        else:
            missing.append(f"{a}/{b}")
    if missing:
        print("  [v10] 缺失字对:", missing)
    return float(np.mean(sims)), em, c2i


# ---------- 随机基线（同模型内，top 高频字两两余弦均值）----------
def random_baseline_18(vecs):
    chars = [c for c in vecs if np.linalg.norm(vecs[c]) > 0][:600]
    rng = np.random.default_rng(0)
    idx = list(range(len(chars)))
    rng.shuffle(idx)
    n = min(400, len(idx) // 2)
    sims = []
    for i in range(n):
        a, b = chars[idx[2 * i]], chars[idx[2 * i + 1]]
        sims.append(cos(vecs[a], vecs[b]))
    return float(np.mean(sims))


def random_baseline_v10(em, c2i):
    # 用 <UNK> 之后的前 600 个字
    chars = [c for c, i in c2i.items() if i > 0 and i < 600]
    rng = np.random.default_rng(0)
    idx = list(range(len(chars)))
    rng.shuffle(idx)
    n = min(400, len(idx) // 2)
    sims = []
    for i in range(n):
        a, b = chars[idx[2 * i]], chars[idx[2 * i + 1]]
        sims.append(cos(em[c2i[a]], em[c2i[b]]))
    return float(np.mean(sims))


if __name__ == "__main__":
    import sys
    results = {}

    print("=" * 60)
    print("跨模型字形相似度对比（焦点形近字对平均余弦）")
    print("=" * 60)

    # raw HZK32
    raw = raw_hzk32_focus()
    results["raw_hzk32_pixels"] = raw
    print(f"\n[raw HZK32 像素]        焦点均值 = {raw:.4f}   (纯字形参考上限)")

    # 18维
    v18, vecs18 = model_18dim_focus()
    base18 = random_baseline_18(vecs18)
    results["handcraft_18dim"] = {"focus": v18, "random_base": base18, "ratio": v18 / (base18 + 1e-8)}
    print(f"[18维手工特征]          焦点均值 = {v18:.4f} | 随机基线 = {base18:.4f} | 倍率 = {v18 / (base18 + 1e-8):.2f}x")

    # v10
    v10, em, c2i = model_v10_focus()
    base10 = random_baseline_v10(em, c2i)
    results["neuroglyph_v10_34dim"] = {"focus": v10, "random_base": base10, "ratio": v10 / (base10 + 1e-8)}
    print(f"[NeuroGlyph v10 34维]   焦点均值 = {v10:.4f} | 随机基线 = {base10:.4f} | 倍率 = {v10 / (base10 + 1e-8):.2f}x")

    # PointMatrix (从已有 eval 结果读取)
    results["pointmatrix_45k_pointnet"] = {"focus": None, "note": "见 eval_glyph_focus --model old 输出"}
    results["pointmatrix_spatial_35k"] = {"focus": 0.4881, "note": "已测"}

    out = os.path.join(SPATIAL, "cross_model_glyph_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {out}")
