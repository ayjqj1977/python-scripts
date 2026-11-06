# -*- coding: utf-8 -*-
"""mine_glyph_pairs.py — 从 6726 字里按像素 L2 距离挖近形字对。
修正: 像素 L2 最大仅 ~32, 阈值必须很小; 先用直方图定阈值(目标~2000对), 再只物化阈值内的对。
像素缓存到 PIXEL_cache.npy 避免重复读盘。
"""
import os, sys, json, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
PIX_CACHE = os.path.join(BASE_DIR, "PIXEL_cache.npy")
OUT = os.path.join(BASE_DIR, "near_pairs_mined.json")
TARGET = 2000          # 目标近形字对总数
THR_HI = 30.0          # 直方图上限


def load_vocab():
    c2i = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    return c2i, {v: k for k, v in c2i.items()}, len(c2i)


def load_pixels(c2i, i2c, V):
    sys.path.insert(0, BASE_DIR)
    from hzk32_reader import HZK32Reader
    reader = HZK32Reader(HZK32_PATH)
    gm = np.zeros((V, 1024), np.float32); loaded = 0
    for idx in range(V):
        ch = i2c.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                gm[idx] = reader.get_glyph_vector(ch).astype(np.float32); loaded += 1
            except (ValueError, IOError):
                pass
    return gm, loaded


def main():
    t0 = time.time()
    c2i, i2c, V = load_vocab()
    if os.path.exists(PIX_CACHE):
        PIXEL = np.load(PIX_CACHE)
        print(f"[Data] 从缓存加载像素 {PIXEL.shape}")
    else:
        PIXEL, npx = load_pixels(c2i, i2c, V)
        np.save(PIX_CACHE, PIXEL)
        print(f"[Data] vocab={V} 像素有效={npx}, 已缓存")

    valid = [i for i in range(V) if PIXEL[i].any()]
    P = PIXEL[valid].astype(np.float32)
    nv = len(valid)
    print(f"[Mine] 有效点阵 {nv} 字, 两两 {nv*(nv-1)//2:,} 对")

    CH = 512
    nb = (nv + CH - 1) // CH
    norms = (P ** 2).sum(1)

    # --- Pass 1: 直方图定阈值 (只计数, 不物化) ---
    edges = np.arange(0, THR_HI + 0.5, 0.5)
    hist = np.zeros(len(edges) - 1, np.int64)
    # 累计各阈值下对数
    thr_grid = np.arange(1.0, THR_HI + 0.5, 0.5)
    counts_at = np.zeros(len(thr_grid), np.int64)
    for ai in range(nb):
        A = P[ai*CH:(ai+1)*CH]; na = norms[ai*CH:(ai+1)*CH]
        for bi in range(ai, nb):
            B = P[bi*CH:(bi+1)*CH]; nb_ = norms[bi*CH:(bi+1)*CH]
            d2 = na[:, None] + nb_[None, :] - 2.0 * (A @ B.T)
            if ai == bi:
                d2 = d2[np.triu_indices(A.shape[0], k=1)]
            else:
                d2 = d2.reshape(-1)
            d = np.sqrt(np.maximum(d2, 0.0))
            h, _ = np.histogram(d, bins=edges)
            hist += h
            # 各阈值计数(只用下三角去重, 已通过 triu/reshape 保证)
            counts_at += np.array([(d < t).sum() for t in thr_grid], np.int64)
    print(f"[Hist] 距离分布(前若干bin):")
    for i in range(min(20, len(hist))):
        if hist[i] > 0:
            print(f"   dist[{edges[i]:.1f},{edges[i+1]:.1f}) : {hist[i]:,} 对")
    chosen = 11.0   # 固定紧阈值: 聚焦"模型真正难分"的低L2硬负(到硬骨头平均水平)
    print(f"[Mine] 选定阈值 dist<{chosen} (固定, 聚焦硬负; 自适应目标{TARGET}仅供参考)")

    # --- Pass 2: 物化阈值内的对 (数量小, 可承受) ---
    pairs = []
    valid_arr = np.array(valid, np.int64)
    for ai in range(nb):
        A = P[ai*CH:(ai+1)*CH]
        gi_base = valid_arr[ai*CH:(ai+1)*CH]
        for bi in range(ai, nb):
            B = P[bi*CH:(bi+1)*CH]
            gj_base = valid_arr[bi*CH:(bi+1)*CH]
            d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * (A @ B.T)
            d2 = np.maximum(d2, 0.0)
            if ai == bi:
                ii, jj = np.triu_indices(A.shape[0], k=1)
            else:
                ii, jj = np.meshgrid(np.arange(A.shape[0]), np.arange(B.shape[0]), indexing="ij")
                ii, jj = ii.reshape(-1), jj.reshape(-1)
            d = np.sqrt(d2[ii, jj])
            m = d < chosen
            ii, jj, d = ii[m], jj[m], d[m]
            for x, y, z in zip(gi_base[ii].tolist(), gj_base[jj].tolist(), d.tolist()):
                pairs.append((x, y, round(z, 3)))
    pairs.sort(key=lambda t: t[2])
    print(f"[Mine] 实际物化 {len(pairs):,} 对")

    # 训练/留出拆分 (按对索引随机分 80/20, 保证字可能重叠但成对不泄露)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    n_tr = int(len(pairs) * 0.8)
    tr = [pairs[i] for i in order[:n_tr]]
    ho = [pairs[i] for i in order[n_tr:]]
    print(f"[Split] 训练近形对={len(tr):,}  留出测试={len(ho):,}")
    print(f"[Sample 训练] " + "  ".join(f"{i2c[a]}/{i2c[b]}({z})" for a, b, z in tr[:12]))
    print(f"[Sample 留出] " + "  ".join(f"{i2c[a]}/{i2c[b]}({z})" for a, b, z in ho[:12]))

    json.dump({
        "note": f"按像素L2距离<{chosen}挖出的近形字对; 训练{len(tr)}/留出{len(ho)}",
        "threshold": chosen, "n_total": len(pairs), "n_train": len(tr), "n_heldout": len(ho),
        "train": [{"a": i2c[x], "b": i2c[y], "da": x, "db": y, "dist": z} for x, y, z in tr],
        "heldout": [{"a": i2c[x], "b": i2c[y], "da": x, "db": y, "dist": z} for x, y, z in ho],
    }, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[Save] {OUT}  ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
