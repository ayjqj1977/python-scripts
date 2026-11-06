# -*- coding: utf-8 -*-
"""
免训练复现脚本 —— 别人拿到 model_for_web.json 即可复现"字形→语义"预测，无需跑训练。
用法:  python reproduce.py 树
      python reproduce.py 狗
"""
import json, math, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "model_for_web.json"), encoding="utf-8"))
cc = M["char_comp"]

def z(x, ms, i):
    return (x - ms[0][i]) / (ms[1][i] + 1e-9)

def feat(ch):
    c = cc[ch]; f = []
    for i in range(M["n_rad"]):
        f.append(z(1 if i == c["r"] else 0, M["rad_ms"], i))
    for i in range(M["n_str"]):
        f.append(z(1 if i == c["st"] else 0, M["str_ms"], i))
    f.append(z(c["n"], M["scnt_ms"], 0))
    for i in range(26):
        f.append(z(c["s32"][i], M["s32_ms"], i))
    for i in range(32):
        f.append(z(c["det"][i], M["det_ms"], i))
    return f

def predict(W, b, f):
    z2 = [sum(f[i] * W[i][j] for i in range(len(f))) + b[j] for j in range(len(b))]
    mx = max(z2); e = [math.exp(x - mx) for x in z2]; s = sum(e)
    return [x / s for x in e]

ch = sys.argv[1] if len(sys.argv) > 1 else "树"
if ch not in cc:
    print("该字无字形特征或词林标签:", ch); sys.exit()

p = predict(M["W"], M["b"], feat(ch))
order = sorted(zip(M["labels"], p), key=lambda t: -t[1])
print(f"字【{ch}】-> 语义大类预测:")
for l, pr in order[:3]:
    print(f"   {l} {M['da_lei'][l]}: {pr*100:.1f}%")
if order[0][0] == "B":
    D = M["D"]; pd = predict(D["W"], D["b"], feat(ch))
    od = sorted(zip(D["mid_labels"], pd), key=lambda t: -t[1])
    print("   物类内部中类: " + ", ".join(
        f"{m}{D['mid_example'].get(m,'')}({pr*100:.0f}%)" for m, pr in od[:3]))
