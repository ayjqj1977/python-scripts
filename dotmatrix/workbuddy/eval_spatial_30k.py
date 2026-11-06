# -*- coding: utf-8 -*-
"""
eval_spatial_30k.py — 评估 PointMatrixSpatial 30K checkpoint
===========================================================
目标：提取训练后的 glyph 嵌入，计算形近字余弦相似度，
     与旧版 PointMatrix (45K, 报告值 0.417) 对比。

使用环境：D:/MICODA/envs/py312_torch/python.exe (torch 2.5.1, numpy 2.0.1)
"""
import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认 30K；可用 --ckpt 指定任意 checkpoint
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step030000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
DEVICE = torch.device("cpu")

# 与 test_pointnet.py 完全一致的重点形近字对（用于和旧版 0.417 对比）
FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"),
    ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"),
    ("王", "玉"),
    ("日", "曰"),
]

# 额外扩展组（更全面观察）
EXTRA_GROUPS = [
    ["人", "入", "八"],
    ["土", "士"],
    ["天", "夫"],
    ["日", "目", "白"],
    ["田", "由", "甲", "申"],
    ["王", "玉", "主"],
    ["力", "刀"],
    ["乌", "鸟"],
    ["问", "间", "闻"],
    ["风", "凤"],
]


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_model(ckpt_path=None):
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
    print(f"[Load] 点阵矩阵: {loaded}/{vocab_size} 字符已加载")

    _ckpt_path = ckpt_path or CKPT_PATH
    print(f"[Load] Checkpoint: {os.path.basename(_ckpt_path)}")
    ckpt = torch.load(_ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", "?")
        ppl = ckpt.get("val_ppl", ckpt.get("ppl", "?"))
        loss = ckpt.get("val_loss", ckpt.get("loss", "?"))
        print(f"[Load] 训练步数: {step} | val_ppl: {ppl} | val_loss: {loss}")
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(DEVICE)
    return model, char_to_idx, idx_to_char, reader, glyph_matrix


@torch.no_grad()
def get_trained_glyph(ch, model, char_to_idx):
    idx = char_to_idx.get(ch, 0)
    raw = model.glyph_matrix[idx:idx + 1]
    feat = model.point_encoder(raw)
    return feat.squeeze(0).numpy()


def get_hzk32_raw(ch, reader):
    try:
        return reader.get_glyph_vector(ch).astype(np.float64)
    except (ValueError, IOError):
        return np.zeros(1024, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="评估 PointMatrixSpatial checkpoint")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint 路径，默认 ckpt_step030000.pt")
    args = parser.parse_args()
    ckpt_path = args.ckpt or CKPT_PATH
    ckpt_name = os.path.basename(ckpt_path)
    # 从文件名提取步数（ckpt_step035000.pt -> 35000）
    try:
        step_num = int("".join(filter(str.isdigit,
                        ckpt_name.split("step")[-1].split(".")[0])))
    except Exception:
        step_num = "?"

    print("=" * 64)
    print(f"PointMatrixSpatial {step_num} 评估")
    print("=" * 64)
    model, char_to_idx, idx_to_char, reader, glyph_matrix = load_model(ckpt_path)

    # ---------- 重点形近字对（与旧版 0.417 对比）----------
    print("\n" + "=" * 64)
    print("A. 重点形近字对（trained cos | HZK32 raw cos | 分离度）")
    print("=" * 64)
    print(f"{'字对':<8} {'trained':>10} {'raw':>10} {'Δ(分离)':>10}")
    print("-" * 42)
    trained_sims, raw_sims = [], []
    for a, b in FOCUS_PAIRS:
        va_t = get_trained_glyph(a, model, char_to_idx)
        vb_t = get_trained_glyph(b, model, char_to_idx)
        sim_t = cosine_sim(va_t, vb_t)
        va_h = get_hzk32_raw(a, reader)
        vb_h = get_hzk32_raw(b, reader)
        sim_h = cosine_sim(va_h, vb_h)
        delta = sim_h - sim_t  # 越大说明模型把这对推得越开（区分越好）
        print(f"{a}/{b:<6} {sim_t:>10.6f} {sim_h:>10.6f} {delta:>+10.6f}")
        trained_sims.append(sim_t)
        raw_sims.append(sim_h)

    avg_t = float(np.mean(trained_sims))
    avg_h = float(np.mean(raw_sims))
    print("-" * 42)
    print(f"{'平均':<8} {avg_t:>10.6f} {avg_h:>10.6f} {avg_h - avg_t:>+10.6f}")
    print(f"\n  → 新版 Spatial({step_num}) 重点字对平均 cosine = {avg_t:.6f}")
    # 同指标实测：旧 PointMatrix 在 35/40/45K 饱和于 ~0.537
    OLD_SAT = 0.537
    print(f"  → 旧版 PointMatrix 同指标饱和值(35-45K)     = {OLD_SAT:.6f}")
    print(f"  → 相对旧版饱和值变化 = {avg_t - OLD_SAT:+.6f}  "
          f"({'更好(更分离)' if avg_t < OLD_SAT else '更差(更混淆)'})")

    # ---------- 扩展组 ----------
    print("\n" + "=" * 64)
    print("B. 扩展形近字组（trained cos）")
    print("=" * 64)
    extra_results = []
    for grp in EXTRA_GROUPS:
        sims = []
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                va = get_trained_glyph(grp[i], model, char_to_idx)
                vb = get_trained_glyph(grp[j], model, char_to_idx)
                sims.append(cosine_sim(va, vb))
                print(f"  {grp[i]} vs {grp[j]}: {sims[-1]:.6f}")
        print(f"  → 组均值: {np.mean(sims):.6f}\n")
        extra_results.append((grp, float(np.mean(sims))))

    # ---------- 全局 top-500 随机基线 ----------
    print("=" * 64)
    print("C. 全局随机基线（前500高频字两两 cosine 均值）")
    print("=" * 64)
    top_chars = []
    for idx in range(1, 6726):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>" and len(ch) == 1:
            try:
                gb = ch.encode("gb2312")
                if len(gb) == 2 and gb[0] >= 0xB0:
                    top_chars.append(ch)
            except UnicodeEncodeError:
                continue
        if len(top_chars) >= 500:
            break

    vecs = []
    for ch in top_chars:
        vecs.append(get_trained_glyph(ch, model, char_to_idx))
    vecs = np.stack(vecs, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sim_matrix = (vecs / norms) @ (vecs / norms).T
    iu = np.triu_indices(len(top_chars), k=1)
    all_sims = sim_matrix[iu]
    global_mean = float(all_sims.mean())
    print(f"  字符数: {len(top_chars)}")
    print(f"  随机基线均值: {global_mean:.6f}")
    # 重点字对均值除以随机基线 = 区分倍率
    ratio = avg_t / (global_mean + 1e-8)
    print(f"  重点字对均值 / 随机基线 = {ratio:.3f}x")

    # ---------- 保存结果 ----------
    out = {
        "step": step_num,
        "ckpt": ckpt_name,
        "model": "PointMatrixSpatial",
        "focus_pairs_avg_trained_cos": avg_t,
        "focus_pairs_avg_raw_cos": avg_h,
        "old_pointmatrix_saturation_trained_cos": 0.537,
        "delta_vs_old": avg_t - 0.537,
        "global_random_baseline": global_mean,
        "focus_vs_random_ratio": ratio,
        "extra_groups": {",".join(g): s for g, s in extra_results},
    }
    out_path = os.path.join(BASE_DIR, "eval_spatial_30k_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
