# -*- coding: utf-8 -*-
"""
义近字判别 —— 分级回归版（"软谱"替代"硬边界"）
================================================================
前作(train_semnear.py)用 0/1「是否同中类」做分类，发现部首绝对主力、几何弱但有信号，
并暴露样例 松·柏/红·绿 被词林单义码(homograph)压到跨大类、模型靠部首拉回"两棵树"。
本篇把硬标签换成词林分层相似度的连续谱：
    跨大类 0.0 | 同大类 0.65 | 同中类 0.80 | 同小类 0.90 | 同词群 0.96 (田久乐/朱新华系数)
用 Ridge 回归预测该谱值，5-seed 字级 80/20 划分，报告 Spearman/Pearson/R²；
并对比 全特征 / 仅部首 / 仅几何，统计块重要性(s32+det 占比)与前作硬分类对照，
回答：几何特征在"软谱"任务上是否比"硬边界"里更出彩。

特征构造、块边界、词林码处理 与 train_semnear.py 完全一致。
"""
import json, os, random
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
try:
    from scipy.stats import spearmanr, pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

BASE  = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
BASE2 = r"D:/desktop/学习结果小文件/neuroglyph34_project"

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

tree = load(os.path.join(BASE, "tree_data.json"))
sf   = load(os.path.join(BASE, "stroke_features.json"))
no   = load(os.path.join(BASE2, "struct32_aligned.json"))["nopy26"]
cilin = load(os.path.join(BASE, "cilin_index.json"))
char_codes = cilin["char_codes"]

# ---------- 1. 组装样本（与 train_semnear.py 完全一致） ----------
samples = []
rad_vocab, str_vocab = {}, {}
for ch, cc in char_codes.items():
    if cc["src"] == "none":
        continue
    t = tree.get(ch); s = sf.get(ch); v = no.get(ch)
    if not (t and s and v):
        continue
    code = cc["codes"][0]
    rad = t["rad"]; st = t["str"]; scnt = len(s)
    rad_vocab.setdefault(rad, len(rad_vocab))
    str_vocab.setdefault(st, len(str_vocab))
    samples.append((ch, code, rad, st, scnt,
                    np.array(v, float), np.array(t["det"], float)))

n_rad = len(rad_vocab); n_str = len(str_vocab)
s32_list = np.array([s[5] for s in samples], float)
det_list = np.array([s[6] for s in samples], float)
scnt_list = np.array([s[4] for s in samples], float)
rad_oh = np.zeros((len(samples), n_rad)); str_oh = np.zeros((len(samples), n_str))
for i, s in enumerate(samples):
    rad_oh[i, rad_vocab[s[2]]] = 1
    str_oh[i, str_vocab[s[3]]] = 1

def standardize(X):
    X = np.asarray(X, float); return (X - X.mean(0)) / (X.std(0) + 1e-9)

s32_z = standardize(s32_list); det_z = standardize(det_list)
scnt_z = standardize(scnt_list).reshape(-1, 1)
rad_z = standardize(rad_oh); str_z = standardize(str_oh)
X_full = np.hstack([rad_z, str_z, scnt_z, s32_z, det_z]).astype(np.float32)   # (N,283)

i_rad0, i_rad1 = 0, n_rad
i_str0, i_str1 = i_rad1, i_rad1 + n_str
i_scnt0, i_scnt1 = i_str1, i_str1 + 1
i_s320, i_s321 = i_scnt1, i_scnt1 + 26
i_det0, i_det1 = i_s321, i_s321 + 32
assert i_det1 == X_full.shape[1] == 283

# 词林码（全码字符串）per sample
codes = [s[1] for s in samples]

def grade_of(ca, cb):
    """词林分层相似度目标值（连续谱）"""
    if ca[0] != cb[0]:
        return 0.0                      # 跨大类
    L = 0
    for k in range(min(len(ca), len(cb))):
        if ca[k] == cb[k]: L += 1
        else: break
    if L >= 7: return 0.99             # 同原子词群
    if L >= 5: return 0.96             # 同词群
    if L >= 4: return 0.90             # 同小类
    if L >= 2: return 0.80             # 同中类
    return 0.65                        # 仅同大类

# ---------- 2. 字对生成（覆盖全谱，按字级子集 S 内生成，避免字泄漏） ----------
rng = random.Random(20260730)

def gen_pairs(S, per_grade=20000, pool_cap=80, pairs_per_group=1200, cross_n=20000):
    """在样本索引集合 S 内，生成覆盖各谱级的字对。"""
    mid_groups = defaultdict(list); big_groups = defaultdict(list)
    for i in S:
        mid_groups[codes[i][0:2]].append(i)
        big_groups[codes[i][0]].append(i)
    bins = defaultdict(list)   # grade -> [(i,j)]

    def add(i, j):
        g = grade_of(codes[i], codes[j])
        if len(bins[g]) < per_grade:
            bins[g].append((i, j))

    # (a) 同中类组内采样 -> 覆盖 0.80/0.90/0.96
    for m, idxs in mid_groups.items():
        idxs = idxs[:pool_cap]
        if len(idxs) < 2: continue
        rng.shuffle(idxs)
        got = 0; tries = 0
        while got < pairs_per_group and tries < pairs_per_group * 60:
            tries += 1
            a, b = rng.sample(idxs, 2)
            if a == b: continue
            add(a, b); got += 1
    # (b) 同大类异中类 -> 0.65
    for bg, idxs in big_groups.items():
        idxs = idxs[:pool_cap]
        if len(idxs) < 2: continue
        rng.shuffle(idxs)
        got = 0; tries = 0
        while got < pairs_per_group and tries < pairs_per_group * 60:
            tries += 1
            a, b = rng.sample(idxs, 2)
            if a == b or codes[a][0:2] == codes[b][0:2]: continue
            add(a, b); got += 1
    # (c) 跨大类随机 -> 0.0
    got = 0; tries = 0
    while got < cross_n and tries < cross_n * 60:
        tries += 1
        a, b = rng.sample(S, 2)
        if codes[a][0] != codes[b][0]:
            add(a, b); got += 1
    return bins

def build_XY(bins):
    pairs = []
    for g, lst in bins.items():
        pairs.extend(lst)
    rng.shuffle(pairs)
    X = np.empty((len(pairs), 283), np.float32)
    y = np.empty(len(pairs), np.float32)
    for k, (i, j) in enumerate(pairs):
        X[k] = np.abs(X_full[i] - X_full[j])
        y[k] = grade_of(codes[i], codes[j])
    return X, y

# ---------- 3. 评估 ----------
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if HAVE_SCIPY:
        return float(spearmanr(a, b).correlation)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if HAVE_SCIPY:
        return float(pearsonr(a, b)[0])
    return float(np.corrcoef(a, b)[0, 1])

def evaluate(X, y, block=None, seeds=(0,1,2,3,4)):
    if block is not None:
        if block == "radical":  sl = slice(i_rad0, i_rad1)
        elif block == "geom":   sl = slice(i_rad1, 283)
        else: raise KeyError(block)
        X = X[:, sl]
    rhos, rs, r2s = [], [], []
    full_coef = None
    for sd in seeds:
        r = np.random.RandomState(sd)
        ix = np.arange(len(y)); r.shuffle(ix)
        nt = int(0.2 * len(y)); te, tr = ix[:nt], ix[nt:]
        if block == "geom":
            alpha = 5.0   # 几何块弱信号，稍强正则稳定
        else:
            alpha = 1.0
        mdl = Ridge(alpha=alpha, max_iter=5000)
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        rhos.append(spearman(pred, y[te]))
        rs.append(pearson(pred, y[te]))
        r2s.append(float(1 - np.sum((y[te]-pred)**2)/np.sum((y[te]-y[te].mean())**2)))
        if block is None and sd == 0:
            full_coef = np.abs(mdl.coef_)
    return {"rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
            "r_mean": float(np.mean(rs)), "r2_mean": float(np.mean(r2s)),
            "coef_abs": full_coef}

slices_map = {"部首": slice(i_rad0,i_rad1), "结构": slice(i_str0,i_str1),
              "笔画数": slice(i_scnt0,i_scnt1), "s32": slice(i_s320,i_s321),
              "det": slice(i_det0,i_det1)}

# ---------- 4. 5-seed 字级 80/20 主评估 ----------
N = len(samples)
all_rho, all_rad_rho, all_geom_rho = [], [], []
imp_acc = None
for sd in (0,1,2,3,4):
    r = np.random.RandomState(sd + 100)
    ix = np.arange(N); r.shuffle(ix)
    nt = int(0.2 * N); test_idx, train_idx = ix[:nt], ix[nt:]
    bins_tr = gen_pairs(list(train_idx))
    bins_te = gen_pairs(list(test_idx))
    Xtr, ytr = build_XY(bins_tr)
    Xte, yte = build_XY(bins_te)
    # 全特征
    mdl = Ridge(alpha=1.0, max_iter=5000); mdl.fit(Xtr, ytr); pred = mdl.predict(Xte)
    all_rho.append(spearman(pred, yte))
    # 仅部首
    mdlr = Ridge(alpha=1.0, max_iter=5000); mdlr.fit(Xtr[:, i_rad0:i_rad1], ytr)
    all_rad_rho.append(spearman(mdlr.predict(Xte[:, i_rad0:i_rad1]), yte))
    # 仅几何
    mdlg = Ridge(alpha=5.0, max_iter=5000); mdlg.fit(Xtr[:, i_rad1:283], ytr)
    all_geom_rho.append(spearman(mdlg.predict(Xte[:, i_rad1:283]), yte))
    if sd == 0:
        imp_acc = np.abs(mdl.coef_)
        # 调试：看各谱级样本量
        print("[seed0 训练字对谱分布]", {("%.2f"%g): len(v) for g, v in bins_tr.items()})

res_full = {"rho_mean": float(np.mean(all_rho)), "rho_std": float(np.std(all_rho)),
            "radical_rho": float(np.mean(all_rad_rho)),
            "geom_rho": float(np.mean(all_geom_rho))}
imp = {k: float(imp_acc[sl].sum() / (imp_acc.sum()+1e-12)) for k, sl in slices_map.items()}
geom_share = imp["s32"] + imp["det"]
print("\n==== 分级回归（词林相似度谱）====")
print("  全特征  Spearman ρ = %.3f±%.3f" % (res_full["rho_mean"], res_full["rho_std"]))
print("  仅部首  ρ = %.3f" % res_full["radical_rho"])
print("  仅几何  ρ = %.3f" % res_full["geom_rho"])
print("  块重要性 |coef| 占比:", {k: "%.1f%%"%(100*v) for k,v in imp.items()})
print("  >> 几何块(s32+det)占比 = %.1f%%  (前作硬分类为 9.2%%)" % (100*geom_share))

# ---------- 5. demo 样例：预测谱值 vs 真实谱值 ----------
mdl = Ridge(alpha=1.0, max_iter=5000)
bins_all = gen_pairs(list(range(N)))
Xall, yall = build_XY(bins_all)
mdl.fit(Xall, yall)
ch2i = {s[0]: i for i, s in enumerate(samples)}
demo = [("江","河"),("松","柏"),("狗","猫"),("红","绿"),("江","树"),("狗","书"),("火","水"),("山","石")]
demo_out = []
print("\n=== 直观样例（预测谱值 vs 真实词林谱值）===")
for a, b in demo:
    if a in ch2i and b in ch2i:
        i, j = ch2i[a], ch2i[b]
        x = np.abs(X_full[i]-X_full[j]).reshape(1,-1)
        pr = float(mdl.predict(x)[0]); tg = grade_of(codes[i], codes[j])
        print("  %s-%s : 预测 %.3f  真实 %.3f" % (a, b, pr, tg))
        demo_out.append({"a":a,"b":b,"pred":round(pr,3),"true":round(tg,3)})

# ---------- 6. 保存 ----------
out = {
    "n_samples": N,
    "full_rho_mean": res_full["rho_mean"], "full_rho_std": res_full["rho_std"],
    "radical_rho": res_full["radical_rho"], "geom_rho": res_full["geom_rho"],
    "block_importance": imp, "geom_share": geom_share,
    "note": "两两|特征差| Ridge 回归；目标=词林分层相似度谱(跨大类0/同大类0.65/同中类0.80/同小类0.90/同词群0.96)；5-seed 字级80/20",
    "demo_pairs": demo_out,
}
with open(os.path.join(BASE, "semnear_reg_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[OK] -> semnear_reg_result.json")
