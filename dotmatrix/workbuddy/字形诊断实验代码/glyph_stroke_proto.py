import json, os, math

BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH = "D:/desktop/mmh_graphics.txt"
VOCAB = os.path.join(BASE, "char_to_idx_full.json")
TREE = os.path.join(BASE, "tree_data.json")
EMB = os.path.join(BASE, "glyph_embeddings.json")
PAIRS = os.path.join(BASE, "near_pairs_mined.json")
OUT = os.path.join(BASE, "stroke_features.json")

# ---------- 1) 解析 graphics.txt ----------
gdata = {}
with open(GRAPH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ch = o.get("character")
        if ch and o.get("medians"):
            gdata[ch] = o["medians"]
print(f"[load] graphics.txt 字 {len(gdata)}")

# ---------- 2) 每字笔画特征(按 上->下 排序) ----------
def stroke_feats(medians):
    ss = []
    for med in medians:
        pts = [(float(x), float(y)) for x, y in med]
        if len(pts) < 2:
            continue
        L = sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                for i in range(len(pts)-1))
        dx = pts[-1][0]-pts[0][0]
        dy = pts[-1][1]-pts[0][1]
        ang = math.atan2(dy, dx)
        cx = sum(p[0] for p in pts)/len(pts)/1024.0
        cy = sum(p[1] for p in pts)/len(pts)/1024.0
        ss.append({"L": L/1024.0, "ang": ang, "cx": cx, "cy": cy})
    if not ss:
        return []
    tot = sum(s["L"] for s in ss)
    # 长度排位 (1=最长), 在原顺序上算
    order = sorted(range(len(ss)), key=lambda i: -ss[i]["L"])
    rank = [0]*len(ss)
    for r, i in enumerate(order):
        rank[i] = r+1
    # 每笔带 rank, 再按 上->下 (cy 大->小) 排序
    for s, rk in zip(ss, rank):
        s["rank"] = rk
    ss.sort(key=lambda s: -s["cy"])
    feats = []
    for s in ss:
        feats.append({
            "L": round(s["L"], 4),
            "fl": round(s["L"]/tot, 4),
            "sin": round(math.sin(s["ang"]), 3),
            "cos": round(math.cos(s["ang"]), 3),
            "x": round(s["cx"], 3),
            "y": round(s["cy"], 3),
            "rank": s["rank"],
        })
    return feats

stroke_db = {}
for ch, med in gdata.items():
    fs = stroke_feats(med)
    if fs:
        stroke_db[ch] = fs
print(f"[feat] 有笔画特征 {len(stroke_db)} 字")

# ---------- 3) V 矢量笔画相似度 ----------
def v_sim(a, b):
    fa, fb = stroke_db.get(a), stroke_db.get(b)
    if not fa or not fb:
        return None
    nA, nB = len(fa), len(fb)
    base = 1.0 - abs(nA-nB)/max(nA, nB)
    m = min(nA, nB)
    if m == 0:
        return 0.0
    gs = []
    for i in range(m):
        sa, sb = fa[i], fb[i]
        lsim = 1.0 - abs(sa["fl"]-sb["fl"])
        rsim = 1.0 - abs(sa["rank"]-sb["rank"])/max(nA, nB)
        dsim = (sa["sin"]*sb["sin"]+sa["cos"]*sb["cos"]+1.0)/2.0
        pd = math.hypot(sa["x"]-sb["x"], sa["y"]-sb["y"])
        psim = max(0.0, 1.0 - pd/0.4)
        g = 0.28*lsim + 0.50*rsim + 0.12*dsim + 0.10*psim
        gs.append(g)
    return base * (sum(gs)/m)

# 诊断: 找出差异最大的笔画
def v_diag(a, b):
    fa, fb = stroke_db.get(a), stroke_db.get(b)
    if not fa or not fb:
        return None
    m = min(len(fa), len(fb))
    diffs = []
    for i in range(m):
        d = abs(fa[i]["L"]-fb[i]["L"]) + 0.05*abs(fa[i]["rank"]-fb[i]["rank"])
        diffs.append((d, i, fa[i], fb[i]))
    diffs.sort(reverse=True)
    top = diffs[0]
    return top

# ---------- 4) T 组合树(从 tree_data 复用规则) ----------
td = json.load(open(TREE, encoding="utf-8"))
def t_sim(a, b):
    ta, tb = td.get(a), td.get(b)
    if not ta or not tb:
        return None
    if ta["rad"] != tb["rad"]:
        return 0.0
    det = max(0.0, sum(ta["det"][i]*tb["det"][i] for i in range(len(ta["det"]))) /
              (math.sqrt(sum(x*x for x in ta["det"]))*math.sqrt(sum(x*x for x in tb["det"]))+1e-12))
    return 0.5*det if ta["str"] != tb["str"] else det

# ---------- 5) C = 组合树 × 矢量笔画 ----------
def c_sim(a, b):
    t = t_sim(a, b)
    v = v_sim(a, b)
    if t is None or v is None:
        return None
    return t * v

# ---------- 6) R/S 余弦(已有嵌入) ----------
emb = json.load(open(EMB, encoding="utf-8"))["embed"]
def cos_mode(mode, a, b):
    va, vb = emb.get(mode, {}).get(a), emb.get(mode, {}).get(b)
    if va is None or vb is None:
        return None
    import numpy as np
    a1 = np.array(va, dtype=float); b1 = np.array(vb, dtype=float)
    return float(a1@b1/(np.linalg.norm(a1)*np.linalg.norm(b1)))

# ---------- 7) 评测 ----------
pairs = json.load(open(PAIRS, encoding="utf-8"))
def mean_mode(fn, plist):
    vs = [fn(p["a"], p["b"]) for p in plist]
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)), len(vs)

import numpy as np
print("\n=== 留出近形字对 均值 (越低=分得越开) ===")
for name, fn in [("R纯像素", lambda a,b: cos_mode("R",a,b)),
                 ("S结构化", lambda a,b: cos_mode("S",a,b)),
                 ("T组合树", t_sim),
                 ("V矢量笔画", v_sim),
                 ("C组合(T×V)", c_sim)]:
    m, n = mean_mode(fn, pairs["heldout"])
    print(f"  {name:10s}: {m:+.4f}  (n={n})")

# ---------- 8) spot-check ----------
spots = [("末","未"),("田","由"),("王","玉"),("风","冈"),("日","曰"),("土","士"),
         ("上","月"),("后","右"),("干","千"),("大","太"),("王","玉"),("由","甲"),
         ("刀","力"),("天","夫"),("千","干")]
print("\n=== spot-check 各模式对比 ===")
print(f"{'对':6s} {'R':>7s} {'S':>7s} {'T':>7s} {'V':>7s} {'C':>7s}")
for a, b in spots:
    r = cos_mode("R", a, b); s = cos_mode("S", a, b)
    t = t_sim(a, b); v = v_sim(a, b); c = c_sim(a, b)
    def f(x): return f"{x*100:6.1f}%" if x is not None else "   -"
    print(f"{a}/{b:4s} {f(r)} {f(s)} {f(t)} {f(v)} {f(c)}")
    d = v_diag(a, b)
    if d and d[0] > 0.05:
        i, sa, sb = d[1], d[2], d[3]
        print(f"         ↳ 矢量诊断: 第{i}笔 长 {sa['L']:.3f}(排{sa['rank']}) vs {sb['L']:.3f}(排{sb['rank']})")

# ---------- 9) 存笔画特征给 app ----------
json.dump(stroke_db, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\n[done] 笔画特征存 {OUT} ({len(stroke_db)} 字)")

# ---------- 10) 存结果 JSON ----------
res = {
    "heldout_mean": {
        "R纯像素": mean_mode(lambda a,b: cos_mode("R",a,b), pairs["heldout"])[0],
        "S结构化": mean_mode(lambda a,b: cos_mode("S",a,b), pairs["heldout"])[0],
        "T组合树": mean_mode(t_sim, pairs["heldout"])[0],
        "V矢量笔画": mean_mode(v_sim, pairs["heldout"])[0],
        "C组合(T×V)": mean_mode(c_sim, pairs["heldout"])[0],
    },
    "spotcheck": {},
}
for a, b in [("末","未"),("田","由"),("王","玉"),("风","冈"),("日","曰"),("土","士"),
             ("上","月"),("后","右"),("干","千"),("大","太"),("由","甲"),("刀","力"),("天","夫")]:
    res["spotcheck"][f"{a}/{b}"] = {
        "R": cos_mode("R",a,b), "S": cos_mode("S",a,b),
        "T": t_sim(a,b), "V": v_sim(a,b), "C": c_sim(a,b),
    }
json.dump(res, open(os.path.join(BASE, "glyph_stroke_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("[done] 结果存 glyph_stroke_result.json")
