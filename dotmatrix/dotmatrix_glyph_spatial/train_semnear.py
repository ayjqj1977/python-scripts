# -*- coding: utf-8 -*-
"""
义近字判别 —— 预结构化判别头的"语义邻近"任务
================================================================
问题：两个汉字在语义上是否"义近"（近义/同一语义场）？
标签来源：同义词词林(扩展版) 的层级编码 —— 同中类(前2位)视为"义近"，跨大类视为"非义近"。
特征：与之前完全一致的可解读结构化字形特征（部首/结构/笔画/s32/det），每字 283 维，标准化。
两两表示：对称差 |fA - fB|（283 维）——"两字在这些维度上差多少"。

科学目的（与形近对照）：
  - 形近 = 几何(s32/det)的领地（PointMatrix 实验已证 cos=0.417 三方案最优）
  - 义近 = 语义的领地；若部首(形旁)能预测"义近"而几何不能，就坐实"几何与语义分工"。
  - 同时做"同大类对照"：正负例都限制在同一大类内，消除"部首→大类"的平凡泄漏，看细粒度义近信号是否还在。

评估：5-seed 80/20 划分（测试集是没见过的字对），报告平衡准确率 / ROC-AUC；
      主模型用 sklearn LogisticRegression(可解释 coef_)，并对比 仅部首 / 仅几何 变体。
"""
import json, os, random
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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
da_lei = cilin["da_lei"]

# ---------- 1. 组装样本（与 train_* 完全一致） ----------
samples = []   # (ch, label, mid, small, rad, st, scnt, s32, det)
rad_vocab, str_vocab = {}, {}
for ch, cc in char_codes.items():
    if cc["src"] == "none":
        continue
    t = tree.get(ch); s = sf.get(ch); v = no.get(ch)
    if not (t and s and v):
        continue
    code = cc["codes"][0]
    label = code[0]            # 大类
    mid = code[0:2]            # 中类
    small = code[0:4]          # 小类
    rad = t["rad"]; st = t["str"]; scnt = len(s)
    rad_vocab.setdefault(rad, len(rad_vocab))
    str_vocab.setdefault(st, len(str_vocab))
    samples.append((ch, label, mid, small, rad, st, scnt,
                    np.array(v, float), np.array(t["det"], float)))

n_rad = len(rad_vocab); n_str = len(str_vocab)
s32_list = np.array([s[7] for s in samples], float)   # 注意下标：s[7]=s32, s[8]=det
det_list = np.array([s[8] for s in samples], float)
scnt_list = np.array([s[6] for s in samples], float)
rad_oh = np.zeros((len(samples), n_rad)); str_oh = np.zeros((len(samples), n_str))
for i, s in enumerate(samples):
    rad_oh[i, rad_vocab[s[4]]] = 1
    str_oh[i, str_vocab[s[5]]] = 1

def standardize(X):
    X = np.asarray(X, float); return (X - X.mean(0)) / (X.std(0) + 1e-9)

s32_z = standardize(s32_list); det_z = standardize(det_list)
scnt_z = standardize(scnt_list).reshape(-1, 1)
rad_z = standardize(rad_oh); str_z = standardize(str_oh)
X_full = np.hstack([rad_z, str_z, scnt_z, s32_z, det_z]).astype(np.float32)   # (N,283)

# 块索引边界（与 X_full 列序一致）
i_rad0, i_rad1 = 0, n_rad
i_str0, i_str1 = i_rad1, i_rad1 + n_str
i_scnt0, i_scnt1 = i_str1, i_str1 + 1
i_s320, i_s321 = i_scnt1, i_scnt1 + 26
i_det0, i_det1 = i_s321, i_s321 + 32
assert i_det1 == X_full.shape[1] == 283, (i_det1, X_full.shape)

def block_slices(name):
    if name == "radical":  return slice(i_rad0, i_rad1)
    if name == "structure":return slice(i_str0, i_str1)
    if name == "scnt":     return slice(i_scnt0, i_scnt1)
    if name == "s32":      return slice(i_s320, i_s321)
    if name == "det":      return slice(i_det0, i_det1)
    if name == "geom":     return slice(i_rad1, 283)   # 结构+笔画+s32+det
    if name == "nogeom":   return slice(0, i_rad1)     # 仅部首
    raise KeyError(name)

print("[样本] %d 字, n_rad=%d n_str=%d" % (len(samples), n_rad, n_str))

# ---------- 2. 字对生成 ----------
rng = random.Random(12345)
N = len(samples)
chl = [s[1] for s in samples]   # 大类
chm = [s[2] for s in samples]   # 中类
chs = [s[3] for s in samples]   # 小类

def gen_pos_by_mid(cap_per_mid=3000, cap_total=60000):
    """同中类字对为正例（C(m,2)，每中类截断）"""
    groups = defaultdict(list)
    for i in range(N):
        groups[chm[i]].append(i)
    pairs = []
    for m, idxs in groups.items():
        m = len(idxs)
        if m < 2: continue
        # 截断该中类的组合数，避免大组主导
        take = min(cap_per_mid, m)
        idxs = idxs[:take]
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                pairs.append((idxs[a], idxs[b]))
                if len(pairs) >= cap_total:
                    return pairs
    return pairs

def sample_neg_diff_class(target, max_try=20*10**6):
    """随机取两字，要求跨大类"""
    out = []
    tries = 0
    while len(out) < target and tries < max_try:
        tries += 1
        i, j = rng.randrange(N), rng.randrange(N)
        if i == j: continue
        if chl[i] != chl[j]:
            out.append((i, j))
    return out

def sample_neg_same_class_diff_mid(target, max_try=20*10**6):
    """同大类、但不同中类（用于消除大类泄漏的对照）"""
    out = []
    tries = 0
    while len(out) < target and tries < max_try:
        tries += 1
        i, j = rng.randrange(N), rng.randrange(N)
        if i == j: continue
        if chl[i] == chl[j] and chm[i] != chm[j]:
            out.append((i, j))
    return out

pos = gen_pos_by_mid()
npos = len(pos)
neg_main = sample_neg_diff_class(npos)
neg_ctrl = sample_neg_same_class_diff_mid(npos)
print("[字对] 正例(同中类)=%d  负例跨大类=%d  负例同大类异中类=%d" % (npos, len(neg_main), len(neg_ctrl)))

def build_XY(pos_pairs, neg_pairs):
    pairs = pos_pairs + neg_pairs
    y = np.array([1]*len(pos_pairs) + [0]*len(neg_pairs), int)
    X = np.empty((len(pairs), X_full.shape[1]), np.float32)
    for k, (i, j) in enumerate(pairs):
        X[k] = np.abs(X_full[i] - X_full[j])
    return X, y

X_main, y_main = build_XY(pos, neg_main)
X_ctrl, y_ctrl = build_XY(pos, neg_ctrl)
print("[矩阵] MAIN %.1f MB | CTRL %.1f MB" % (X_main.nbytes/1e6, X_ctrl.nbytes/1e6))

# ---------- 3. 训练 + 评估 ----------
def evaluate(X, y, block=None, seeds=(0,1,2,3,4)):
    if block is not None:
        sl = block_slices(block); X = X[:, sl]
    aucs, baccs = [], []
    coef_acc = None
    for sd in seeds:
        r = np.random.RandomState(sd)
        ix = np.arange(len(y)); r.shuffle(ix)
        nt = int(0.2*len(y)); te, tr = ix[:nt], ix[nt:]
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=3000, solver="saga")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
        if sd == 0:
            coef_acc = np.abs(clf.coef_[0])
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "bacc_mean": float(np.mean(baccs)), "bacc_std": float(np.std(baccs)),
        "coef_abs": coef_acc,
    }

def block_importance(coef_abs, slices_map):
    out = {}
    tot = coef_abs.sum() + 1e-12
    for name, sl in slices_map.items():
        out[name] = float(coef_abs[sl].sum() / tot)
    return out

print("\n==== MAIN：正=同中类, 负=跨大类 ====")
res_main_full = evaluate(X_main, y_main)
print("  全特征  AUC=%.3f±%.3f  平衡acc=%.3f±%.3f" %
      (res_main_full["auc_mean"], res_main_full["auc_std"],
       res_main_full["bacc_mean"], res_main_full["bacc_std"]))
res_main_rad = evaluate(X_main, y_main, "radical")
res_main_geom = evaluate(X_main, y_main, "geom")
print("  仅部首  AUC=%.3f  平衡acc=%.3f" % (res_main_rad["auc_mean"], res_main_rad["bacc_mean"]))
print("  仅几何  AUC=%.3f  平衡acc=%.3f" % (res_main_geom["auc_mean"], res_main_geom["bacc_mean"]))

# 块重要性（全特征，seed0 coef）
slices_map = {"部首": slice(i_rad0,i_rad1), "结构": slice(i_str0,i_str1),
              "笔画数": slice(i_scnt0,i_scnt1), "s32": slice(i_s320,i_s321),
              "det": slice(i_det0,i_det1)}
imp_main = block_importance(res_main_full["coef_abs"], slices_map)
print("  块重要性(全特征 |coef| 占比):",
      {k: "%.1f%%" % (100*v) for k, v in imp_main.items()})

print("\n==== CONTROL：正=同中类, 负=同大类异中类（消除大类泄漏）====")
res_ctrl_full = evaluate(X_ctrl, y_ctrl)
print("  全特征  AUC=%.3f±%.3f  平衡acc=%.3f±%.3f" %
      (res_ctrl_full["auc_mean"], res_ctrl_full["auc_std"],
       res_ctrl_full["bacc_mean"], res_ctrl_full["bacc_std"]))
res_ctrl_rad = evaluate(X_ctrl, y_ctrl, "radical")
res_ctrl_geom = evaluate(X_ctrl, y_ctrl, "geom")
print("  仅部首  AUC=%.3f  平衡acc=%.3f" % (res_ctrl_rad["auc_mean"], res_ctrl_rad["bacc_mean"]))
print("  仅几何  AUC=%.3f  平衡acc=%.3f" % (res_ctrl_geom["auc_mean"], res_ctrl_geom["bacc_mean"]))
imp_ctrl = block_importance(res_ctrl_full["coef_abs"], slices_map)
print("  块重要性(全特征 |coef| 占比):",
      {k: "%.1f%%" % (100*v) for k, v in imp_ctrl.items()})

# ---------- 4. 直观样例（用 MAIN 全特征模型） ----------
clf = LogisticRegression(penalty="l2", C=1.0, max_iter=3000, solver="saga")
clf.fit(X_main, y_main)
def pair_score(i, j):
    x = np.abs(X_full[i] - X_full[j]).reshape(1, -1)
    return float(clf.predict_proba(x)[0, 1])
ch2i = {s[0]: i for i, s in enumerate(samples)}
demo_pairs = [("江","河"),("松","柏"),("狗","猫"),("红","绿"),
              ("江","树"),("狗","书"),("火","水"),("山","石")]
print("\n=== 直观样例（义近得分 0~1）===")
demo_out = []
for a, b in demo_pairs:
    if a in ch2i and b in ch2i:
        sc = pair_score(ch2i[a], ch2i[b])
        same_mid = chm[ch2i[a]] == chm[ch2i[b]]
        print("  %s-%s : %.3f  (同中类=%s)" % (a, b, sc, same_mid))
        demo_out.append({"a": a, "b": b, "score": round(sc, 3), "same_mid": same_mid})

# ---------- 5. 保存 ----------
out = {
    "n_samples": N, "n_pairs_pos": npos,
    "MAIN": {
        "auc_mean": res_main_full["auc_mean"], "auc_std": res_main_full["auc_std"],
        "bacc_mean": res_main_full["bacc_mean"], "bacc_std": res_main_full["bacc_std"],
        "radical_only": {"auc": res_main_rad["auc_mean"], "bacc": res_main_rad["bacc_mean"]},
        "geom_only": {"auc": res_main_geom["auc_mean"], "bacc": res_main_geom["bacc_mean"]},
        "block_importance": imp_main,
    },
    "CONTROL": {
        "auc_mean": res_ctrl_full["auc_mean"], "auc_std": res_ctrl_full["auc_std"],
        "bacc_mean": res_ctrl_full["bacc_mean"], "bacc_std": res_ctrl_full["bacc_std"],
        "radical_only": {"auc": res_ctrl_rad["auc_mean"], "bacc": res_ctrl_rad["bacc_mean"]},
        "geom_only": {"auc": res_ctrl_geom["auc_mean"], "bacc": res_ctrl_geom["bacc_mean"]},
        "block_importance": imp_ctrl,
    },
    "demo_pairs": demo_out,
    "note": "两两|特征差|二分类LR；正=同词林中类(前2位)，MAIN负=跨大类，CONTROL负=同大类异中类；5-seed 80/20",
}
with open(os.path.join(BASE, "semnear_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[OK] -> semnear_result.json")
