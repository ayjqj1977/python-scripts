# -*- coding: utf-8 -*-
"""
可训练判别器：用「用户手工设计的结构化字形特征」预测「同义词词林语义大类」
================================================================================
动机：汉字的字形里到底藏没藏语义信号？形声字理论说"形旁表义"——木旁多半植物、犭旁多半动物。
      本实验用真实机器学习（多类逻辑回归判别头，本地 CPU 秒训，零 torch）来验证这件事。

哲学定位：输入特征全部是用户手工设计的、可解读的结构化维度（部首/结构/笔画/s32/det）；
          只有"怎么组合这些特征去下判断"是模型自己训出来的——这正是从 G2(零训练公式)
          迈向"会学习"的最小一步，且踩在预结构化地盘上。

三个模型对比，隔离"字形结构化特征"各自的贡献：
  A. 仅部首 one-hot          —— 测"形旁表义"有多强（最朴素的可解读信号）
  B. 仅几何结构(s32+det+笔画数+结构类型, 不含部首) —— 测"抛开部首标签，纯字形结构能否预测语义"
  C. 全部(部首+结构+笔画数+s32+det)              —— 上限

评估：按"字"随机 80/20 划分训练/测试（测试集是模型没见过的新字，真·泛化）。
"""
import json, sys, os, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

BASE = os.environ.get("BASE")
BASE2 = os.environ.get("BASE2")

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

tree = load(os.path.join(BASE, "tree_data.json"))
sf   = load(os.path.join(BASE, "stroke_features.json"))
no   = load(os.path.join(BASE2, "struct32_aligned.json"))["nopy26"]
cilin = load(os.path.join(BASE, "cilin_index.json"))
char_codes = cilin["char_codes"]
da_lei = cilin["da_lei"]

rng = random.Random(42)

# ---------- 1. 组装样本 ----------
samples = []  # (char, label_letter, rad, str, stroke_count, s32, det)
rad_vocab, str_vocab = {}, {}
for ch, cc in char_codes.items():
    if cc["src"] == "none":
        continue
    t = tree.get(ch); s = sf.get(ch); v = no.get(ch)
    if not (t and s and v):
        continue
    label = cc["codes"][0][0]            # 大类首字母
    mid = cc["codes"][0][0:2]             # 中类(用于"物"类内部形旁表义检验)
    rad = t["rad"]; st = t["str"]
    scnt = len(s)
    rad_vocab.setdefault(rad, len(rad_vocab))
    str_vocab.setdefault(st, len(str_vocab))
    samples.append((ch, label, mid, rad, st, scnt, np.array(v, float), np.array(t["det"], float)))

print("[样本] 可用 %d 字, 大类 %d 个" % (len(samples), len(set(s[1] for s in samples))))
labels = sorted(set(s[1] for s in samples))
label_idx = {l: i for i, l in enumerate(labels)}
print("[大类]", " ".join("%s=%s" % (l, da_lei[l]) for l in labels))

# ---------- 2. 基础特征块 ----------
s32_list = np.array([s[6] for s in samples], float)   # (N,26)
det_list = np.array([s[7] for s in samples], float)   # (N,32)
scnt_list = np.array([s[5] for s in samples], float)
n_rad = len(rad_vocab); n_str = len(str_vocab)
rad_onehot = np.zeros((len(samples), n_rad))
str_onehot = np.zeros((len(samples), n_str))
for i, s in enumerate(samples):
    rad_onehot[i, rad_vocab[s[3]]] = 1
    str_onehot[i, str_vocab[s[4]]] = 1

def standardize(X):
    X = np.asarray(X, float)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd

s32_z = standardize(s32_list)
det_z = standardize(det_list)
scnt_z = standardize(scnt_list).reshape(-1, 1)
rad_z = standardize(rad_onehot)
str_z = standardize(str_onehot)

FEATS = {
    "A_radical": rad_z,
    "B_geom":    np.hstack([s32_z, det_z, scnt_z, str_z]),
    "C_full":    np.hstack([rad_z, str_z, scnt_z, s32_z, det_z]),
}
FEAT_NAMES = {
    "A_radical": ["部首%d" % r for r in range(n_rad)],
    "B_geom":    ["s32_%d" % i for i in range(26)] + ["det_%d" % i for i in range(32)]
                + ["笔画数"] + ["结构%d" % k for k in range(n_str)],
    "C_full":    ["部首%d" % r for r in range(n_rad)] + ["结构%d" % k for k in range(n_str)]
                + ["笔画数"] + ["s32_%d" % i for i in range(26)] + ["det_%d" % i for i in range(32)],
}
y = np.array([label_idx[s[1]] for s in samples])

# ---------- 稳定的多类 LR (softmax + Adam, 偏置=类别先验, 强L2) ----------
def softmax(Z):
    Z -= Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)

def train_lr(Xtr, ytr, n_class, iters=2500, lr=0.02, reg=0.5):
    N, D = Xtr.shape
    W = np.zeros((D, n_class))
    prior = np.bincount(ytr, minlength=n_class).astype(float)
    prior = np.log(prior / prior.sum() + 1e-9)
    b = prior.copy()
    mW = np.zeros_like(W); mb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    Y = np.zeros((N, n_class)); Y[np.arange(N), ytr] = 1
    for t in range(1, iters + 1):
        P = softmax(Xtr @ W + b)
        gW = (Xtr.T @ (P - Y)) / N + reg * W
        gb = (P - Y).sum(0) / N
        mW = beta1 * mW + (1 - beta1) * gW
        mb = beta1 * mb + (1 - beta1) * gb
        vW = beta2 * mW * mW + (1 - beta2) * (gW ** 2)
        vb = beta2 * mb * mb + (1 - beta2) * (gb ** 2)
        mWh = mW / (1 - beta1 ** t); mbh = mb / (1 - beta1 ** t)
        vWh = vW / (1 - beta2 ** t); vbh = vb / (1 - beta2 ** t)
        W -= lr * mWh / (np.sqrt(vWh) + eps)
        b -= lr * mbh / (np.sqrt(vbh) + eps)
    return W, b

def acc(Xte, yte, W, b):
    return float((softmax(Xte @ W + b).argmax(1) == yte).mean())

# ---------- 3. 多 seed 评估（看 C_full / B_geom 的 +1.8pp 是否稳定） ----------
def split_idx(n, seed):
    r = random.Random(seed)
    ix = list(range(n)); r.shuffle(ix)
    nt = int(0.2 * n)
    return ix[:nt], ix[nt:]

def run_key(key, seeds=(0, 1, 2, 3, 4)):
    X = FEATS[key]
    accs = []
    for sd in seeds:
        te, tr = split_idx(len(samples), sd)
        W, b = train_lr(X[tr], y[tr], len(labels))
        accs.append(acc(X[te], y[te], W, b))
    return float(np.mean(accs)), float(np.std(accs))

print("\n==== 多 seed (5) 评估 ====")
multiseed = {}
for key in ["A_radical", "B_geom", "C_full"]:
    m, sd = run_key(key)
    multiseed[key] = {"mean": m, "std": sd}
    print("[%s] 测试准确率 = %.1f%% ± %.1f%%" % (key, 100 * m, 100 * sd))

mc = np.bincount(y[split_idx(len(samples), 0)[1]]).argmax()
base_acc = float((y[split_idx(len(samples), 0)[0]] == mc).mean())
print("[基线] 永远猜最多类 -> 测试 %.1f%%" % (100 * base_acc))

# ---------- 4. 实验 D：仅在"物(B)"类内部，用字形预测"中类"(动物/植物/自然物…) ----------
# 这是"形旁表义"假说最该成立的地方——同属"物"，靠形旁区分动物/植物。
B_idx = [i for i, s in enumerate(samples) if s[1] == "B"]
mid_labels = sorted(set(samples[i][2] for i in B_idx))
mid_idx = {m: k for k, m in enumerate(mid_labels)}
yB = np.array([mid_idx[samples[i][2]] for i in B_idx])
XB = FEATS["C_full"][B_idx]
print("\n==== 实验 D：物(B)类内部中类判别（形旁表义检验）====")
print("  物类字数 %d, 中类 %d 个: %s" % (len(B_idx), len(mid_labels),
      " ".join("%s=%d" % (m, sum(1 for i in B_idx if samples[i][2]==m)) for m in mid_labels)))
dB_accs = []
for sd in (0, 1, 2, 3, 4):
    te, tr = split_idx(len(B_idx), sd)
    W, b = train_lr(XB[tr], yB[tr], len(mid_labels))
    dB_accs.append(acc(XB[te], yB[te], W, b))
dB_mean = float(np.mean(dB_accs))
mcB = np.bincount(yB[split_idx(len(B_idx), 0)[1]]).argmax()
dB_base = float((yB[split_idx(len(B_idx), 0)[0]] == mcB).mean())
print("  测试准确率 = %.1f%%  |  基线(猜最多中类) = %.1f%%  |  提升 = +%.1fpp"
      % (100 * dB_mean, 100 * dB_base, 100 * (dB_mean - dB_base)))

# ---------- 5. 可解读性（用 seed=0 的 C_full 权重） ----------
te0, tr0 = split_idx(len(samples), 0)
Wc, _ = train_lr(FEATS["C_full"][tr0], y[tr0], len(labels))
names = FEAT_NAMES["C_full"]
top_per_class = {}
for ci, lab in enumerate(labels):
    wcol = Wc[:, ci]
    order = np.argsort(-np.abs(wcol))[:8]
    top_per_class[lab] = [(names[j], round(float(wcol[j]), 3)) for j in order]

# ---------- 6. 保存评估结果 ----------
out = {
    "n_samples": len(samples), "labels": labels, "da_lei": da_lei,
    "baseline_acc": base_acc,
    "multiseed": multiseed,
    "expD": {"n_B": len(B_idx), "mid_labels": mid_labels,
             "test_acc": dB_mean, "baseline_acc": dB_base,
             "gain_pp": 100 * (dB_mean - dB_base)},
    "top_per_class": top_per_class, "n_rad": n_rad, "n_str": n_str,
}
with open(os.path.join(BASE, "discriminator_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n==== 可解读性：C_full 中预测各语义类最强的字形维度（权重|·|Top8）====")
for lab in labels:
    print("  %s(%s):" % (lab, da_lei[lab]), ", ".join("%s(%+.2f)" % (n_, w_) for n_, w_ in top_per_class[lab]))

# ---------- 7. 最终可部署模型（全量训练） + 导出供网页预测 ----------
Xfull = FEATS["C_full"]
Wfinal, bfinal = train_lr(Xfull, y, len(labels), iters=4000)
# 标准化参数（对全量重算，供网页复现特征）
def meanstd(X):
    X = np.asarray(X, float); return X.mean(0).tolist(), X.std(0).tolist()
rad_ms = meanstd(rad_z); str_ms = meanstd(str_z)
scnt_ms = meanstd(scnt_z.reshape(-1, 1)); s32_ms = meanstd(s32_z); det_ms = meanstd(det_z)
# 逐字原始组件（网页据此重建特征向量）
char_comp = {}
for i, s in enumerate(samples):
    ch = s[0]
    char_comp[ch] = {"r": rad_vocab[s[3]], "st": str_vocab[s[4]],
                     "n": int(s[5]), "s32": s[6].tolist(), "det": s[7].tolist(),
                     "lab": s[1]}
rad_idx2rad = {v: k for k, v in rad_vocab.items()}
str_idx2name = {v: k for k, v in str_vocab.items()}
model = {
    "labels": labels, "da_lei": da_lei,
    "W": Wfinal.tolist(), "b": bfinal.tolist(),
    "n_rad": n_rad, "n_str": n_str,
    "rad_ms": rad_ms, "str_ms": str_ms, "scnt_ms": scnt_ms,
    "s32_ms": s32_ms, "det_ms": det_ms,
    "rad_idx2rad": {str(k): v for k, v in rad_idx2rad.items()},
    "str_idx2name": {str(k): v for k, v in str_idx2name.items()},
    "top_per_class": top_per_class,
    "char_comp": char_comp,
}
# 实验 D 的最终部署模型（物类内部 18 中类判别，信号最强、最能展示"真·学习"）
Wd, bd = train_lr(XB, yB, len(mid_labels), iters=4000)
# 中类代表字(从词林原文取首词, 让 demo 可读)
mid_example = {}
import re as _re
with open(os.path.join(BASE, "cilin_extended.txt"), encoding="utf-8") as _f:
    for _line in _f:
        _line=_line.rstrip("\n")
        if not _line.strip(): continue
        _m=_re.match(r"^([A-Za-z0-9]+)([=#@]?)\s+(.*)$", _line)
        if not _m: continue
        _code=_m.group(1)+(_m.group(2) or "="); _ws=_m.group(3).split()
        if not _ws: continue
        _mid=_code[0:2]
        if _mid in mid_labels and _mid not in mid_example:
            mid_example[_mid]=_ws[0]
model["D"] = {"mid_labels": mid_labels, "mid_example": mid_example,
              "W": Wd.tolist(), "b": bd.tolist()}

with open(os.path.join(BASE, "model_for_web.json"), "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False, separators=(",", ":"))
print("\n[OK] 评估 -> discriminator_result.json")
print("[OK] 部署模型 -> model_for_web.json (字组件 %d, 大类权重 %dx%d, 中类权重 %dx%d)" %
      (len(char_comp), Wfinal.shape[0], Wfinal.shape[1], Wd.shape[0], Wd.shape[1]))
