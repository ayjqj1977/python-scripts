# -*- coding: utf-8 -*-
"""
笔画级关系编码原型 (stroke-level RELATION encoding)
====================================================
思路: 之前 C = 组合树(T) × 矢量(V) 已把留出均值压到 0.065, 但末/未, 田/由, 由/甲
这几对仍卡在 0.55~0.66. 根因: 它们部首同、结构同、笔顺类型序列完全相同
(末/未 都是 横横竖撇捺), 只有"哪一笔更长"这种离散几何关系不同.

本脚本把"笔画级关系"显式编码为离散事实:
  - 笔画数
  - 笔顺类型序列 (xlsx, 6 类)
  - 每笔长度排位 (1=最长)  <- 来自 medians 几何
  - 最长笔是"第几笔"(从上数) <- 离散关系关键事实
  - 长度偏序: 对每对(i,j)记 sign(Li-Lj)
然后两个字的"关系相似度" R = 这些离散事实的吻合度; 末/未 因最长笔不同(第1笔 vs 第2笔)
而 R 很低, 从而 G = C × R 把它们压下去.

数据:
  - stroke_features.json (medians 几何, 每字每笔 {L,fl,sin,cos,x,y,rank})
  - struct32_aligned.json (nopy26: 笔顺类型序列)
  - tree_data.json (组合树: rad/str/det)
  - near_pairs_mined.json (留出错对)
"""
import os, json, math
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return json.load(open(os.path.join(BASE, name), encoding="utf-8"))

SF = load("stroke_features.json")          # char -> [ {L,fl,sin,cos,x,y,rank}, ... ]
NO = load(os.path.join("D:/desktop/学习结果小文件/neuroglyph34_project", "struct32_aligned.json"))["nopy26"]
TREE = load("tree_data.json")
PAIRS = load("near_pairs_mined.json")

# ---------- 工具 ----------
def cosine(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb+1e-12)

# 组合树规则 (与 app 一致)
def treeSim(a, b):
    ta, tb = TREE.get(a), TREE.get(b)
    if not ta or not tb: return None
    if ta["rad"] != tb["rad"]: return 0.0
    det = max(0.0, cosine(ta["det"], tb["det"]))
    return 0.5*det if ta["str"] != tb["str"] else det

# 矢量笔画 (与 app 一致: 按上->下对齐, 重权长度排位)
def vSim(a, b):
    fa, fb = SF.get(a), SF.get(b)
    if not fa or not fb: return None
    nA, nB = len(fa), len(fb)
    base = 1 - abs(nA-nB)/max(nA, nB)
    m = min(nA, nB)
    if m == 0: return 0.0
    gs = []
    for i in range(m):
        sa, sb = fa[i], fb[i]
        lsim = 1 - abs(sa["fl"]-sb["fl"])
        rsim = 1 - abs(sa["rank"]-sb["rank"])/max(nA, nB)
        dsim = (sa["sin"]*sb["sin"]+sa["cos"]*sb["cos"]+1)/2
        pd = math.hypot(sa["x"]-sb["x"], sa["y"]-sb["y"])
        psim = max(0.0, 1-pd/0.4)
        gs.append(0.28*lsim+0.50*rsim+0.12*dsim+0.10*psim)
    return base*(sum(gs)/m)

# ---------- 笔画级关系编码 ----------
def relation_facts(ch):
    """返回字的离散关系事实."""
    feats = SF.get(ch)
    if not feats: return None
    n = len(feats)
    # 笔顺类型序列 (xlsx)
    v = NO.get(ch)
    if v is None:
        type_seq = [0]*n
    else:
        nn = int(v[9]); type_seq = [int(x) for x in v[16:16+nn]]
        type_seq = (type_seq+[0]*n)[:n]
    # 每笔长度排位 rank (1=最长)
    ranks = [f["rank"] for f in feats]
    longest_idx = ranks.index(min(ranks)) if ranks else 0   # 排位1=最长
    # 长度偏序 sign(Li-Lj)
    Ls = [f["L"] for f in feats]
    order = []
    for i in range(n):
        for j in range(i+1, n):
            order.append(1 if Ls[i] > Ls[j] else (0 if Ls[i] < Ls[j] else 2))
    return {"n": n, "type_seq": type_seq, "longest_idx": longest_idx, "order": order}

def relSim(a, b):
    """关系相似度 0..1. 用'按笔序对齐的长度轮廓'乘积, 让少数差异笔真正起作用."""
    fa, fb = relation_facts(a), relation_facts(b)
    if fa is None or fb is None: return None
    na, nb = fa["n"], fb["n"]
    # 1) 笔画数
    cnt_f = 1.0 - 0.6*abs(na-nb)/max(na, nb, 1)
    # 2) 笔顺类型序列 (位置对齐)
    m = min(na, nb)
    if m == 0: return 0.0
    type_match = sum(1 for i in range(m) if fa["type_seq"][i] == fb["type_seq"][i])/m
    type_f = 0.3 + 0.7*type_match
    # 3) 按笔序对齐的长度轮廓 (乘积形式: 少数差异笔也能压低总分)
    La = [f["L"] for f in SF.get(a)]; Lb = [f["L"] for f in SF.get(b)]
    maxL = max(La[:m]+Lb[:m]) + 1e-9
    prof = 1.0
    for i in range(m):
        prof *= max(0.0, 1 - abs(La[i]-Lb[i])/maxL)
    return cnt_f*(0.30*type_f + 0.70*prof)

# ---------- 评估 ----------
def gSim(a, b, gate=0.2):
    t, v, r = treeSim(a, b), vSim(a, b), relSim(a, b)
    if t is None or v is None or r is None: return None
    return t*v*r   # C×R, 0..1 (t,v,r 均为 0..1 分数)

def mean_mode(fn):
    vs = [fn(p["a"], p["b"]) for p in PAIRS["heldout"]]
    vs = [v for v in vs if v is not None]
    return float(sum(vs)/len(vs)), len(vs)

print("=== 留出近形字对 (均值, 越低越好) ===")
gv, _ = mean_mode(lambda a, b: gSim(a, b))
print(f"  G=C×R 均值 = {gv:.4f}")
rv, _ = mean_mode(lambda a, b: relSim(a, b))
print(f"  (参考) R单独 均值 = {rv:.4f}")
cv, _ = mean_mode(lambda a, b: treeSim(a,b) * vSim(a,b))
print(f"  (参考) C=T×V 均值 = {cv:.4f}")

print("\n=== Spot-check 对照表 (数值=相似度%, 越低越分得开) ===")
spot = [("末","未"),("田","由"),("由","甲"),("王","玉"),("日","曰"),("上","月"),
        ("风","冈"),("后","右"),("干","千"),("土","士"),("天","夫"),("刀","力"),("人","入")]
print(f"{'对':<6}{'T树':>7}{'V矢':>7}{'C=T×V':>9}{'R关系':>8}{'G=C×R':>9}")
for a, b in spot:
    t = treeSim(a, b); v = vSim(a, b); r = relSim(a, b)
    c = t*v if (t is not None and v is not None) else None
    g = gSim(a, b)
    def pct(x): return f"{x*100:5.1f}%" if x is not None else "  缺"
    print(f"{a}/{b:<4}{pct(t):>7}{pct(v):>7}{pct(c):>9}{pct(r):>8}{pct(g):>9}")

# 关系诊断: 末/未 到底差在哪
print("\n=== 关系诊断: 末 vs 未 ===")
fa, fb = relation_facts("末"), relation_facts("未")
names = {0:"点",1:"横",2:"竖",3:"撇",4:"捺",5:"折"}
print(f"末: 笔画数={fa['n']} 最长笔=第{fa['longest_idx']+1}笔(从上) 笔顺=[{' '.join(names.get(x,'?') for x in fa['type_seq'])}]")
print(f"未: 笔画数={fb['n']} 最长笔=第{fb['longest_idx']+1}笔(从上) 笔顺=[{' '.join(names.get(x,'?') for x in fb['type_seq'])}]")
print(f"-> 最长笔不同({'同' if fa['longest_idx']==fb['longest_idx'] else '异'}), 笔顺类型序列{'同' if fa['type_seq']==fb['type_seq'] else '异'}")

# 存结果
result = {"heldout": {"R": rv, "C": cv, "G": gv},
          "spotcheck": {f"{a}/{b}": {"T": treeSim(a,b), "V": vSim(a,b),
                        "C": treeSim(a,b)*vSim(a,b),
                        "R": relSim(a,b), "G": gSim(a,b)} for a, b in spot}}
json.dump(result, open(os.path.join(BASE, "glyph_relation_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[done] 结果存 glyph_relation_result.json")
