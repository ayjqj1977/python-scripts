# -*- coding: utf-8 -*-
"""全面复核：文件夹里所有"能直接取向量"的模型，统一测字形分离度。
指标：焦点形近字对平均余弦 / 随机基线平均余弦 = 字形区分力指数 (越高=越有字形意识)
"""
import os, sys, json
import numpy as np

BASE = "D:/desktop/学习结果小文件"
SPATIAL = os.path.join(BASE, "dotmatrix_glyph_spatial")

focus_pairs = [("己","已"),("己","巳"),("已","巳"),("未","末"),("干","千"),
               ("干","于"),("千","于"),("大","太"),("王","玉"),("日","曰")]

def cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_charidx(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        k0 = next(iter(d))
        if isinstance(k0, str):
            return lambda ch: d.get(ch)
        else:
            rev = {v: k for k, v in d.items()}
            return lambda ch: rev.get(ch)
    return None

def glyph_index(vecs, idx_of, rng, label):
    # 焦点对
    focus_sims = []
    for a, b in focus_pairs:
        ia, ib = idx_of(a), idx_of(b)
        if ia is None or ib is None:
            continue
        focus_sims.append(cos(vecs[ia], vecs[ib]))
    avg_focus = float(np.mean(focus_sims)) if focus_sims else float("nan")
    # 随机基线：同词表内随机抽 300 对
    N = vecs.shape[0]
    pool = list(range(N))
    rng.shuffle(pool)
    rand_sims = []
    for i in range(300):
        ia, ib = pool[2*i], pool[2*i+1]
        if ia == ib:
            continue
        rand_sims.append(cos(vecs[ia], vecs[ib]))
    avg_rand = float(np.mean(rand_sims))
    gi = avg_focus / avg_rand if avg_rand != 0 else float("nan")
    print(f"  {label}: dim={vecs.shape[1]} N={N} | 焦点={avg_focus:.4f} 随机={avg_rand:.4f} 区分力={gi:.2f}x")
    return {"label": label, "dim": int(vecs.shape[1]), "N": int(N),
            "focus": round(avg_focus,4), "random": round(avg_rand,4),
            "glyph_index": round(gi,2)}

rng = np.random.default_rng(42)
results = []

print("="*70)
print("A. NeuroGlyph 34维 (文本上下文嵌入) — 全版本轨迹")
print("="*70)
for v in ["v2","v4","v5","v6","v10","project"]:
    if v == "project":
        npy = os.path.join(BASE, "neuroglyph34_project/embedding_matrix.npy")
        c2i = os.path.join(BASE, "neuroglyph34_project/char_to_idx.json")
    else:
        npy = os.path.join(BASE, f"neuroglyph34_cloud_{v}/embedding_matrix.npy")
        c2i = os.path.join(BASE, f"neuroglyph34_cloud_{v}/char_to_idx.json")
    try:
        em = np.load(npy)
        idx_of = load_charidx(c2i)
        r = glyph_index(em, idx_of, rng, f"NeuroGlyph {v}")
        results.append(r)
    except Exception as e:
        print(f"  NeuroGlyph {v}: ERROR {e}")

print("\n"+"="*70)
print("B. 云端训练系列 (文本上下文, 大词表)")
print("="*70)
for name, npy, vj in [
    ("云端训练", os.path.join(BASE,"云端训练/embedding_large.npy"), os.path.join(BASE,"云端训练/vocab_large.json")),
    ("云端完整训练", os.path.join(BASE,"云端完整训练/embedding_large.npy"), os.path.join(BASE,"云端完整训练/vocab_large.json")),
]:
    try:
        em = np.load(npy)
        vd = json.load(open(vj, encoding="utf-8"))
        c2i = vd.get("char_to_idx", vd)
        idx_of = load_charidx(c2i) if isinstance(c2i, dict) else None
        r = glyph_index(em, idx_of, rng, name)
        results.append(r)
    except Exception as e:
        print(f"  {name}: ERROR {e}")

print("\n"+"="*70)
print("C. PointMatrix 预提取字形向量 glyph_vectors_top500.npy (64维)")
print("="*70)
try:
    gv = np.load(os.path.join(BASE, "dotmatrix_glyph/glyph_vectors_top500.npy"))
    c2i_full = json.load(open(os.path.join(BASE,"dotmatrix_glyph_spatial/char_to_idx_full.json"), encoding="utf-8"))
    # 假设 top500 按 char_to_idx_full 的 idx 顺序 (0..499)
    idx_of = lambda ch: c2i_full.get(ch)
    # 验证：日/曰 应高度相似
    chk = cos(gv[idx_of("日")], gv[idx_of("曰")])
    print(f"  [校验] 日/曰 余弦={chk:.3f} (应>0.5 才能说明顺序正确)")
    r = glyph_index(gv, idx_of, rng, "PointMatrix glyph_top500")
    results.append(r)
except Exception as e:
    print(f"  glyph_vectors_top500: ERROR {e}")

print("\n"+"="*70)
print("汇总 (按字形区分力降序)")
print("="*70)
for r in sorted(results, key=lambda x: -x["glyph_index"]):
    print(f"  {r['label']:30s} 区分力={r['glyph_index']:6.2f}x  焦点={r['focus']:.4f} 随机={r['random']:.4f} dim={r['dim']}")

json.dump(results, open(os.path.join(SPATIAL, "cross_model_full_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已保存 -> cross_model_full_result.json")
