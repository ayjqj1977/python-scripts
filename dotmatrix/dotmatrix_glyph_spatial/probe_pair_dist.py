# -*- coding: utf-8 -*-
"""probe_pair_dist.py — 校准: 已知硬骨头对的像素L2距离, 判断阈值合理性。"""
import os, sys, json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
PIX_CACHE = os.path.join(BASE_DIR, "PIXEL_cache.npy")

c2i = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
i2c = {v: k for k, v in c2i.items()}
PIXEL = np.load(PIX_CACHE)

FOCUS = [("己","已"),("己","巳"),("已","巳"),("未","末"),("干","千"),("干","于"),
         ("千","于"),("大","太"),("王","玉"),("日","曰"),("王","主"),("风","凤"),
         ("人","入"),("土","士"),("天","夫")]
HELDOUT = [("刀","力"),("儿","几"),("午","牛"),("田","由"),("田","甲"),("天","夭"),
           ("贝","见"),("勾","句"),("乒","乓"),("大","犬"),("木","本"),("日","目"),
           ("王","五"),("人","个"),("土","王")]

def d(a, b):
    va, vb = PIXEL[c2i[a]], PIXEL[c2i[b]]
    return float(np.sqrt(((va - vb) ** 2).sum()))

def report(name, pairs):
    print(f"\n=== {name} ===")
    ds = []
    for a, b in pairs:
        if a in c2i and b in c2i:
            v = d(a, b); ds.append(v)
            print(f"  {a}/{b}  L2={v:.3f}")
    print(f"  -> {name} 平均 {np.mean(ds):.3f}, 最小 {min(ds):.3f}, 最大 {max(ds):.3f}")

report("FOCUS(训练用硬骨头)", FOCUS)
report("HELDOUT(留出)", HELDOUT)
