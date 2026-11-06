# -*- coding: utf-8 -*-
"""
官方 backbone 原型 (official backbone: 部首三维 + 笔顺十位 + 矢量几何)
====================================================
按用户纠正后的真实编码体系重写关系路 R:
  - 部首三维 (xlsx 部首索引1/2/3) -> 组合树 T 的 rad/str 层   (官方部首编码)
  - 笔顺十位 (xlsx 笔顺1/10)      -> 关系路 R2 的笔顺类型匹配 (官方 T9 十位)
  - 矢量几何 (medians)            -> V 矢量笔画 + R2 的长度轮廓/排位顺序
关系路 R2 改为乘积式: cnt_f × type_match × prof × rankord
  rankord = 长度排位顺序一致度, 专治"笔顺同但哪笔长顺序反"的 末/未/土/士 类
G2 = T × V × R2
对比旧 G = T × V × R(混合式 0.3+0.7type_f 等)
完全规则计算, 无需 torch, 纯 numpy/标准库即可跑 (轻薄本友好).
"""
import os, json, math
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return json.load(open(os.path.join(BASE, name), encoding="utf-8"))

SF   = load("stroke_features.json")                                                 # medians 几何
NO   = load(os.path.join("D:/desktop/学习结果小文件/neuroglyph34_project",
                         "struct32_aligned.json"))["nopy26"]                        # 官方位: 笔顺1-10 等
TREE = load("tree_data.json")                                                       # 组合树: rad/str/det
PAIRS = load("near_pairs_mined.json")                                              # 留出近形字对

# ---------- 工具 ----------
def cosine(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb+1e-12)

# 组合树规则 (与 app 一致, rad/str 来自 xlsx 官方部首三维+结构)
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

# ---------- 旧 R (混合式, 作为对比基线) ----------
def relSim_old(a, b):
    fa, fb = SF.get(a), SF.get(b)
    if not fa or not fb: return None
    na, nb = len(fa), len(fb); m = min(na, nb)
    if m == 0: return 0.0
    cnt_f = 1.0 - 0.6*abs(na-nb)/max(na, nb, 1)
    v, v2 = NO.get(a), NO.get(b)
    if v and v2:
        cap = min(int(v[9]), 10); cap2 = min(int(v2[9]), 10)
        ta = [int(x) for x in v[16:16+cap]]; tb = [int(x) for x in v2[16:16+cap2]]
        mm = min(len(ta), len(tb)); type_match = sum(1 for i in range(mm) if ta[i]==tb[i])/mm
    else:
        type_match = 0.0
    type_f = 0.3 + 0.7*type_match
    La = [f["L"] for f in fa][:m]; Lb = [f["L"] for f in fb][:m]
    maxL = max(La+Lb)+1e-9; prof = 1.0
    for i in range(m): prof *= max(0.0, 1-abs(La[i]-Lb[i])/maxL)
    return cnt_f*(0.30*type_f + 0.70*prof)

# ---------- 新 R2 (乘积式, 官方笔顺 + 矢量排位顺序) ----------
def relSim2(a, b):
    fa, fb = SF.get(a), SF.get(b)
    if not fa or not fb: return None
    na, nb = len(fa), len(fb); m = min(na, nb)
    if m == 0: return 0.0
    # 1) 笔画数因子
    cnt_f = 1.0 - 0.6*abs(na-nb)/max(na, nb, 1)
    # 2) 官方笔顺十位 (xlsx 笔顺1-10) 位置对齐匹配率
    v, v2 = NO.get(a), NO.get(b)
    if v and v2:
        cap = min(int(v[9]), 10); cap2 = min(int(v2[9]), 10)
        ta = [int(x) for x in v[16:16+cap]]; tb = [int(x) for x in v2[16:16+cap2]]
        mm = min(len(ta), len(tb)); type_match = sum(1 for i in range(mm) if ta[i]==tb[i])/mm
    else:
        type_match = 0.0
    # 3) 长度轮廓 (乘积: 少数差异笔也压低)
    La = [f["L"] for f in fa][:m]; Lb = [f["L"] for f in fb][:m]
    maxL = max(La+Lb)+1e-9; prof = 1.0
    for i in range(m): prof *= max(0.0, 1-abs(La[i]-Lb[i])/maxL)
    # 4) 长度排位顺序一致度 (rank 1=最长): 专治"哪笔长顺序反"
    ra = [f["rank"] for f in fa][:m]; rb = [f["rank"] for f in fb][:m]
    mt = tot = 0
    for i in range(m):
        for j in range(i+1, m):
            sa = 1 if ra[i] < ra[j] else (0 if ra[i] > ra[j] else 2)
            sb = 1 if rb[i] < rb[j] else (0 if rb[i] > rb[j] else 2)
            tot += 1; mt += (1 if sa == sb else 0)
    rankord = mt/tot if tot else 1.0
    # 乘积式: 任一差异都直接相乘压低总分
    return cnt_f * type_match * prof * rankord

# ---------- 组合 ----------
def gSim_old(a, b):
    t, v, r = treeSim(a, b), vSim(a, b), relSim_old(a, b)
    if t is None or v is None or r is None: return None
    return t*v*r
def gSim_new(a, b):
    t, v, r = treeSim(a, b), vSim(a, b), relSim2(a, b)
    if t is None or v is None or r is None: return None
    return t*v*r

# ---------- 评估 ----------
def mean_mode(fn):
    vs = [fn(p["a"], p["b"]) for p in PAIRS["heldout"]]
    vs = [v for v in vs if v is not None]
    return float(sum(vs)/len(vs)), len(vs)

print("=== 留出近形字对 (均值, 越低越好) ===")
gv2, _ = mean_mode(gSim_new); print(f"  G2=T×V×R2(官方乘积式) 均值 = {gv2:.4f}")
gv1, _ = mean_mode(gSim_old); print(f"  (对比) G =T×V×R (旧混合式) 均值 = {gv1:.4f}")

print("\n=== Spot-check (相似度%, 越低越分得开) ===")
spot = [("末","未"),("田","由"),("由","甲"),("王","玉"),("日","曰"),("上","月"),
        ("风","冈"),("后","右"),("干","千"),("土","士"),("天","夫"),("刀","力"),("人","入")]
print(f"{'对':<6}{'G旧':>8}{'G2新':>8}{'R2':>8}")
for a, b in spot:
    g1, g2, r2 = gSim_old(a, b), gSim_new(a, b), relSim2(a, b)
    def p(x): return f"{x*100:5.1f}%" if x is not None else "  缺"
    print(f"{a}/{b:<4}{p(g1):>8}{p(g2):>8}{p(r2):>8}")

print("\n=== 诊断: 末 vs 未 排位顺序 ===")
fa, fb = SF["末"], SF["未"]
ra = [f["rank"] for f in fa]; rb = [f["rank"] for f in fb]
print(f"末 rank={ra}  未 rank={rb}")
print(f"-> 上横(笔0)在末排{ra[0]} / 在未排{rb[0]}; 中横(笔1)在末排{ra[1]} / 在未排{rb[1]}")
print(f"-> '上横/中横谁更长'顺序: 末={'上横长' if ra[0]<ra[1] else '中横长'}, "
      f"未={'上横长' if rb[0]<rb[1] else '中横长'}  => 顺序相反 => rankord 压低")

result = {"heldout": {"G_old": gv1, "G2_new": gv2},
          "spotcheck": {f"{a}/{b}": {"G_old": gSim_old(a, b), "G2_new": gSim_new(a, b), "R2": relSim2(a, b)}
                        for a, b in spot}}
json.dump(result, open(os.path.join(BASE, "glyph_official_backbone_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[done] 存 glyph_official_backbone_result.json")
