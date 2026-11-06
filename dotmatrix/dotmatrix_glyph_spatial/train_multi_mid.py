# -*- coding: utf-8 -*-
"""
跨大类「内部中类判别」扩展实验 —— 回答 Q1：形旁只在"实体类(物)"有效吗？
============================================================================
思路：把实验 D（仅在"物 B"类内部用字形判别 18 中类）推广到其它「类内中类充足且相对平衡」的大类：
      A 人(14中类)、D 抽象(14)、E 特征(6)、H 活动(14)，外加 B 物(18) 作参照。
对每个大类内部：
  - 用 C_full 结构化字形特征（部首+结构+笔画+s32+det）做多类 softmax LR 判别头
  - 5-seed 按字 80/20 划分，报告 held-out 测试准确率 vs 基线(猜最多中类)
  - 统计 top 特征里「部首(形旁) vs 几何(s32/det/笔画/结构)」的占比，看信号来源
"""
import json, os, random
import numpy as np

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

# ---------- 1. 组装样本（与 train_discriminator.py 完全一致，确保特征可比） ----------
samples = []
rad_vocab, str_vocab = {}, {}
for ch, cc in char_codes.items():
    if cc["src"] == "none":
        continue
    t = tree.get(ch); s = sf.get(ch); v = no.get(ch)
    if not (t and s and v):
        continue
    label = cc["codes"][0][0]
    mid = cc["codes"][0][0:2]
    rad = t["rad"]; st = t["str"]; scnt = len(s)
    rad_vocab.setdefault(rad, len(rad_vocab))
    str_vocab.setdefault(st, len(str_vocab))
    samples.append((ch, label, mid, rad, st, scnt,
                    np.array(v, float), np.array(t["det"], float)))

n_rad = len(rad_vocab); n_str = len(str_vocab)
s32_list = np.array([s[6] for s in samples], float)
det_list = np.array([s[7] for s in samples], float)
scnt_list = np.array([s[5] for s in samples], float)
rad_oh = np.zeros((len(samples), n_rad)); str_oh = np.zeros((len(samples), n_str))
for i, s in enumerate(samples):
    rad_oh[i, rad_vocab[s[3]]] = 1
    str_oh[i, str_vocab[s[4]]] = 1

def standardize(X):
    X = np.asarray(X, float); return (X - X.mean(0)) / (X.std(0) + 1e-9)

s32_z = standardize(s32_list); det_z = standardize(det_list)
scnt_z = standardize(scnt_list).reshape(-1, 1)
rad_z = standardize(rad_oh); str_z = standardize(str_oh)
X_full = np.hstack([rad_z, str_z, scnt_z, s32_z, det_z])   # (N,283)

NAMES = (["部首%d" % r for r in range(n_rad)]
         + ["结构%d" % k for k in range(n_str)]
         + ["笔画数"]
         + ["s32_%d" % i for i in range(26)]
         + ["det_%d" % i for i in range(32)])

def softmax(Z):
    Z -= Z.max(1, keepdims=True); e = np.exp(Z); return e / e.sum(1, keepdims=True)

def train_lr(Xtr, ytr, n_class, iters=3000, lr=0.02, reg=0.5):
    N, D = Xtr.shape
    W = np.zeros((D, n_class))
    prior = np.bincount(ytr, minlength=n_class).astype(float)
    b = np.log(prior / prior.sum() + 1e-9)
    mW = np.zeros_like(W); mb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Y = np.zeros((N, n_class)); Y[np.arange(N), ytr] = 1
    for t in range(1, iters + 1):
        P = softmax(Xtr @ W + b)
        gW = (Xtr.T @ (P - Y)) / N + reg * W
        gb = (P - Y).sum(0) / N
        mW = b1 * mW + (1 - b1) * gW; mb = b1 * mb + (1 - b1) * gb
        vW = b2 * mW * mW + (1 - b2) * (gW ** 2); vb = b2 * mb * mb + (1 - b2) * (gb ** 2)
        mWh = mW / (1 - b1 ** t); mbh = mb / (1 - b1 ** t)
        vWh = vW / (1 - b2 ** t); vbh = vb / (1 - b2 ** t)
        W -= lr * mWh / (np.sqrt(vWh) + eps)
        b -= lr * mbh / (np.sqrt(vbh) + eps)
    return W, b

def acc(Xte, yte, W, b):
    return float((softmax(Xte @ W + b).argmax(1) == yte).mean())

def split_idx(n, seed):
    r = random.Random(seed); ix = list(range(n)); r.shuffle(ix)
    nt = int(0.2 * n); return ix[:nt], ix[nt:]

def is_rad(n):  return n.startswith("部首")
def is_geom(n): return n.startswith("s32") or n.startswith("det") or n == "笔画数" or n.startswith("结构")

# ---------- 2. 逐大类跑内部中类判别 ----------
TARGETS = ["B", "A", "D", "E", "H"]
rng = random.Random(42)
results = {}
print("%-4s %-6s %5s %4s %7s %7s %8s  %s" %
      ("类", "名", "字数", "中类", "测试", "基线", "增益pp", "top15特征(部首/几何)"))
for lab in TARGETS:
    idx = [i for i, s in enumerate(samples) if s[1] == lab]
    mids = sorted(set(samples[i][2] for i in idx))
    if len(mids) < 3:
        print(f"  {lab} 中类过少({len(mids)})，跳过"); continue
    mid_idx = {m: k for k, m in enumerate(mids)}
    ysub = np.array([mid_idx[samples[i][2]] for i in idx])
    Xsub = X_full[idx]
    nmid = len(mids)
    accs = []
    for sd in (0, 1, 2, 3, 4):
        te, tr = split_idx(len(idx), sd)
        W, b = train_lr(Xsub[tr], ysub[tr], nmid)
        accs.append(acc(Xsub[te], ysub[te], W, b))
    test = float(np.mean(accs))
    mc = np.bincount(ysub[split_idx(len(idx), 0)[1]]).argmax()
    base = float((ysub[split_idx(len(idx), 0)[0]] == mc).mean())
    # 全量训一个，取 top 特征来源
    Wf, _ = train_lr(Xsub, ysub, nmid, iters=4000)
    imp = np.mean(np.abs(Wf), axis=1)            # 每特征平均|权重|
    top = np.argsort(-imp)[:15]
    nr = sum(1 for j in top if is_rad(NAMES[j]))
    ng = sum(1 for j in top if is_geom(NAMES[j]))
    results[lab] = {
        "name": da_lei.get(lab, "?"), "n": len(idx), "n_mid": nmid,
        "mid_labels": mids, "test_acc": test, "baseline_acc": base,
        "gain_pp": 100 * (test - base),
        "top_names": [NAMES[j] for j in top],
        "top_rad_count": nr, "top_geom_count": ng,
    }
    print("%-4s %-6s %5d %4d %6.1f%% %6.1f%% %+7.1fpp  %2d/%2d" %
          (lab, da_lei.get(lab, "?"), len(idx), nmid,
           100 * test, 100 * base, 100 * (test - base), nr, ng))

out = {"targets": TARGETS, "da_lei": da_lei, "results": results,
       "note": "C_full 特征(283维)：部首211 + 结构13 + 笔画1 + s32_26 + det_32；5-seed 按字80/20；softmax LR 判别头"}
with open(os.path.join(BASE, "train_multi_mid_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[OK] -> train_multi_mid_result.json")
