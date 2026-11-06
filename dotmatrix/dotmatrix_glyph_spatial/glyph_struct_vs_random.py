# -*- coding: utf-8 -*-
"""
glyph_struct_vs_random.py — 纯字形判别力对照实验 (留出测试版)
============================================================
把"语义/LM目标"彻底甩开，只练一件事：字形区分力。
对比两条路谁能把近形字(硬骨头)分开：

  Arm R (纯随机)   : 输入=1024维原始点阵, 编码器随机初始化, 从像素里自己学结构
  Arm S (预先结构化): 输入=26维结构码(四角+五笔+总笔画+部首+结构+笔顺), 随机初始化,
                      喂的是"点→线→部首→字"已被人预先编码好的结构
  Arm F (融合头)   : 像素 + 结构 双输入, 可学习融合头 (plan X)

结构码维度(nopy26) = 四角5 + 五笔4 + 总笔画1 + 部首索引3 + 部首外笔画1
                     + 同部同画序号1 + 结构1 + 笔顺10 —— 正是"点成线、线成部首、部首成字"

训练目标(纯字形, 无语义):
  1) 对比损失(SimCLR式): 同一字两增强视图互正例, batch内其余字为负例 -> 推开所有不同字
  2) 近形字硬负分离: 对 FOCUS_PAIRS 显式惩罚 cosine>margin (这些对用于训练监督)
验证:
  HELD_OUT_PAIRS: 一组训练时从未见过的近形字, 仅用于最终泛化测试

判别力指标:
  focus_avg / heldout_avg = 近形字对平均cosine (越低越好, 负=推到反向=极强分离)
  baseline  = 随机字对平均cosine (越低越好, 但不能坍缩到全1)
  分离裕度 margin = baseline - focus_avg (越高越好)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
STRUCT_JSON = "D:/desktop/学习结果小文件/neuroglyph34_project/struct32_aligned.json"
DEVICE = torch.device("cpu")

# 训练时用于硬负监督的近形字(看过的题)
FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"), ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"), ("王", "玉"), ("日", "曰"),
    ("王", "主"), ("风", "凤"), ("人", "入"), ("土", "士"), ("天", "夫"),
]
# 留出测试: 训练时从未出现过的近形字(没看过的题)
HELD_OUT_PAIRS = [
    ("刀", "力"), ("儿", "几"), ("午", "牛"), ("田", "由"), ("田", "甲"),
    ("天", "夭"), ("贝", "见"), ("勾", "句"), ("乒", "乓"), ("大", "犬"),
    ("木", "本"), ("日", "目"), ("王", "五"), ("人", "个"), ("土", "王"),
]

BATCH = 256
EPOCHS = 30
LR = 1e-3
TEMP = 0.10
FOCUS_MARGIN = 0.10
FOCUS_LAMBDA = 0.5
PX_DROPOUT = 0.15
PX_NOISE = 0.05
ST_NOISE = 0.03
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------- 数据 ----------------
def load_vocab():
    with open(CHAR_TO_IDX_PATH, encoding="utf-8") as f:
        c2i = json.load(f)
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


def load_struct(c2i, i2c, V):
    sj = json.load(open(STRUCT_JSON, encoding="utf-8"))
    nopy26 = sj["nopy26"]
    S = np.zeros((V, 26), np.float32); hit = 0
    for idx in range(V):
        ch = i2c.get(idx, "")
        if ch in nopy26:
            arr = np.asarray(nopy26[ch], np.float32)
            if arr.shape[0] >= 26:
                S[idx] = arr[:26]; hit += 1
    return S, hit


# ---------------- 模型 ----------------
class PixelEncoder(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class StructEncoder(nn.Module):
    def __init__(self, in_dim=26, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FusionEncoder(nn.Module):
    def __init__(self, p_in=1024, s_in=26, pH=256, sH=64, out_dim=64):
        super().__init__()
        self.pixel = nn.Sequential(nn.Linear(p_in, pH), nn.ReLU())
        self.struct = nn.Sequential(nn.Linear(s_in, sH), nn.ReLU())
        self.head = nn.Linear(pH + sH, out_dim)
    def forward(self, p, s):
        return F.normalize(self.head(torch.cat([self.pixel(p), self.struct(s)], -1)), dim=-1)


# ---------------- 损失 ----------------
def contrastive_loss(ea, eb, temp=TEMP):
    N = ea.shape[0]
    sim = ea @ eb.T / temp
    labels = torch.arange(N, device=ea.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def focus_loss(emb, pairs_local):
    loss = 0.0
    for i, j in pairs_local:
        loss += F.relu((emb[i] * emb[j]).sum() - FOCUS_MARGIN)
    return loss / len(pairs_local)


def aug_pixel(x):
    x = x.clone()
    mask = (x > 0) & (torch.rand_like(x) < PX_DROPOUT)
    x[mask] = 0.0
    return x + torch.randn_like(x) * PX_NOISE


def aug_struct(x):
    return x + torch.randn_like(x) * ST_NOISE


# ---------------- 指标 ----------------
def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l2norm(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True); n[n == 0] = 1
    return mat / n


def global_baseline(E, c2i, i2c, n=500):
    top = []
    for idx in range(1, E.shape[0]):
        ch = i2c.get(idx, "")
        if ch and ch != "<UNK>" and len(ch) == 1:
            try:
                gb = ch.encode("gb2312")
                if len(gb) == 2 and gb[0] >= 0xB0:
                    top.append(ch)
            except UnicodeEncodeError:
                continue
        if len(top) >= n:
            break
    vecs = l2norm(np.stack([E[c2i[c]] for c in top], 0))
    sm = vecs @ vecs.T
    iu = np.triu_indices(len(top), k=1)
    return float(sm[iu].mean())


def evaluate(encoder, arm, PIXEL, STRUCT, c2i, i2c, pair_lists):
    with torch.no_grad():
        if arm == "R":
            E = encoder(torch.from_numpy(PIXEL)).numpy().astype(np.float64)
        elif arm == "S":
            E = encoder(torch.from_numpy(STRUCT)).numpy().astype(np.float64)
        else:
            E = encoder(torch.from_numpy(PIXEL), torch.from_numpy(STRUCT)).numpy().astype(np.float64)
    out = {}
    for name, pairs in pair_lists.items():
        fps = [cosine_sim(E[c2i[a]], E[c2i[b]]) for a, b in pairs if a in c2i and b in c2i]
        out[name] = {"avg": float(np.mean(fps)),
                     "pairs": {f"{a}/{b}": cosine_sim(E[c2i[a]], E[c2i[b]]) for a, b in pairs}}
    return out, global_baseline(E, c2i, i2c)


def train_arm(arm, encoder, PIXEL_t, STRUCT_t, c2i, i2c, focus_px, focus_st, focus_pairs_local):
    opt = torch.optim.Adam(encoder.parameters(), lr=LR)
    N = PIXEL_t.shape[0]
    focus_px, focus_st = focus_px.to(DEVICE), focus_st.to(DEVICE)
    print(f"\n=== 训练 Arm {arm} ===")
    pair_lists = {"focus(看过)": FOCUS_PAIRS, "heldout(没看)": HELD_OUT_PAIRS}
    for ep in range(EPOCHS):
        perm = torch.randperm(N); tot = 0.0; nb = 0
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            px = PIXEL_t[idx]; st = STRUCT_t[idx]
            if arm == "R":
                ea, eb = encoder(aug_pixel(px)), encoder(aug_pixel(px))
            elif arm == "S":
                ea, eb = encoder(aug_struct(st)), encoder(aug_struct(st))
            else:
                ea = encoder(aug_pixel(px), aug_struct(st))
                eb = encoder(aug_pixel(px), aug_struct(st))
            Lc = contrastive_loss(ea, eb)
            if arm == "R":
                fe = encoder(aug_pixel(focus_px))
            elif arm == "S":
                fe = encoder(aug_struct(focus_st))
            else:
                fe = encoder(aug_pixel(focus_px), aug_struct(focus_st))
            Lf = focus_loss(fe, focus_pairs_local)
            loss = Lc + FOCUS_LAMBDA * Lf
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        res, gb = evaluate(encoder, arm, PIXEL_t.numpy(), STRUCT_t.numpy(), c2i, i2c, pair_lists)
        fa = res["focus(看过)"]["avg"]; ho = res["heldout(没看)"]["avg"]
        print(f"  ep{ep+1:02d} loss={tot/nb:.3f}  focus_avg={fa:+.4f}  heldout_avg={ho:+.4f}  baseline={gb:.4f}")
    res, gb = evaluate(encoder, arm, PIXEL_t.numpy(), STRUCT_t.numpy(), c2i, i2c, pair_lists)
    return {"focus_avg_cos": res["focus(看过)"]["avg"],
            "heldout_avg_cos": res["heldout(没看)"]["avg"],
            "global_baseline": gb,
            "margin_focus": gb - res["focus(看过)"]["avg"],
            "margin_heldout": gb - res["heldout(没看)"]["avg"],
            "focus_pairs": res["focus(看过)"]["pairs"],
            "heldout_pairs": res["heldout(没看)"]["pairs"]}


def main():
    t0 = time.time()
    c2i, i2c, V = load_vocab()
    PIXEL, npx = load_pixels(c2i, i2c, V)
    STRUCT, nst = load_struct(c2i, i2c, V)
    print(f"[Data] vocab={V} 像素={npx} 结构={nst}")
    PIXEL_t, STRUCT_t = torch.from_numpy(PIXEL).to(DEVICE), torch.from_numpy(STRUCT).to(DEVICE)

    fc = sorted({c for pr in FOCUS_PAIRS for c in pr})
    fcp_local = [(fc.index(a), fc.index(b)) for a, b in FOCUS_PAIRS if a in fc and b in fc]
    focus_px = PIXEL_t[[c2i[c] for c in fc]]
    focus_st = STRUCT_t[[c2i[c] for c in fc]]
    print(f"[Focus] 焦点字 {len(fc)} 个, 训练监督对 {len(fcp_local)} 对; 留出测试对 {len(HELD_OUT_PAIRS)} 对")

    results = {}
    encR = PixelEncoder().to(DEVICE)
    results["R_纯随机_像素"] = train_arm("R", encR, PIXEL_t, STRUCT_t, c2i, i2c, focus_px, focus_st, fcp_local)
    torch.save(encR.state_dict(), os.path.join(BASE_DIR, "enc_R_pixel.pt"))

    encS = StructEncoder().to(DEVICE)
    results["S_预先结构化_结构码"] = train_arm("S", encS, PIXEL_t, STRUCT_t, c2i, i2c, focus_px, focus_st, fcp_local)
    torch.save(encS.state_dict(), os.path.join(BASE_DIR, "enc_S_struct.pt"))

    encF = FusionEncoder().to(DEVICE)
    results["F_融合头_像素+结构"] = train_arm("F", encF, PIXEL_t, STRUCT_t, c2i, i2c, focus_px, focus_st, fcp_local)
    torch.save(encF.state_dict(), os.path.join(BASE_DIR, "enc_F_fusion.pt"))

    # 参考
    ref = json.load(open(os.path.join(BASE_DIR, "fuse_struct_v2_result.json"), encoding="utf-8"))
    p64, s26 = ref["reps"]["P_像素64"], ref["reps"]["S26_新结构(去拼音)"]
    results["P64_现状LM基线"] = {"focus_avg_cos": p64["focus_avg_cos"], "heldout_avg_cos": None,
                                 "global_baseline": p64["global_baseline"],
                                 "margin_focus": p64["global_baseline"] - p64["focus_avg_cos"],
                                 "margin_heldout": None, "focus_pairs": p64["focus_pairs"], "heldout_pairs": {}}
    results["S26_原始结构码_未训练"] = {"focus_avg_cos": s26["focus_avg_cos"], "heldout_avg_cos": None,
                                        "global_baseline": s26["global_baseline"],
                                        "margin_focus": s26["global_baseline"] - s26["focus_avg_cos"],
                                        "margin_heldout": None, "focus_pairs": s26["focus_pairs"], "heldout_pairs": {}}

    print("\n" + "=" * 80)
    print("纯字形判别力对照（focus/heldout越低越好；负cos=硬骨头被推到反向=极强分离）")
    print("=" * 80)
    print(f"{'模型':<22}{'focus':>8}{'heldout':>9}{'baseline':>10}{'裕度F':>8}{'裕度H':>8}")
    print("-" * 67)
    for k in ["P64_现状LM基线", "S26_原始结构码_未训练", "R_纯随机_像素",
             "S_预先结构化_结构码", "F_融合头_像素+结构"]:
        r = results[k]
        hf = f"{r['heldout_avg_cos']:+.4f}" if r["heldout_avg_cos"] is not None else "  N/A"
        mf = f"{r['margin_focus']:+.3f}"
        mh = f"{r['margin_heldout']:+.3f}" if r["margin_heldout"] is not None else "  N/A"
        tag = "  ←参考" if ("基线" in k or "未训练" in k) else ""
        print(f"{k:<22}{r['focus_avg_cos']:>+8.4f}{hf:>9}{r['global_baseline']:>10.4f}{mf:>8}{mh:>8}{tag}")

    out = {"note": "纯字形对比损失(留出版): 纯随机像素 vs 预先结构化 vs 融合头",
           "hyper": dict(BATCH=BATCH, EPOCHS=EPOCHS, LR=LR, TEMP=TEMP,
                         FOCUS_MARGIN=FOCUS_MARGIN, FOCUS_LAMBDA=FOCUS_LAMBDA),
           "reps": results, "elapsed_sec": round(time.time() - t0, 1)}
    op = os.path.join(BASE_DIR, "glyph_struct_vs_random_result.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[Save] {op}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
