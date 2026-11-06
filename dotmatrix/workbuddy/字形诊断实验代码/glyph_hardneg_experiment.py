# -*- coding: utf-8 -*-
"""
glyph_hardneg_experiment.py — 足量硬负版: 预先结构化 vs 纯随机
==============================================================
与上一版的区别: 把"15对手工硬骨头"换成从6726字自动挖出的近形字对:
  - 443 对 训练硬负 (像素L2<11, 模型真正难分的那批)
  - 111 对 留出测试 (训练时从未出现过的近形字, 测泛化)

仍甩开语义LM, 只练字形区分力。三臂随机初始化, 唯一区别是输入:
  R 纯随机像素 / S 预先结构化(26维结构码) / F 融合头(像素+结构)
目标: 对比损失(SimCLR) + 近形字硬负分离(443对)。

判别力: 训练对avg / 留出对avg 近形字余弦(越低越好); baseline=随机字对余弦; 裕度=baseline-avg。
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
PIX_CACHE = os.path.join(BASE_DIR, "PIXEL_cache.npy")
MINED = os.path.join(BASE_DIR, "near_pairs_mined.json")
DEVICE = torch.device("cpu")

BATCH = 256
EPOCHS = 40
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
    c2i = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    return c2i, {v: k for k, v in c2i.items()}, len(c2i)


def load_pixels(V):
    if os.path.exists(PIX_CACHE):
        return np.load(PIX_CACHE).astype(np.float32)
    sys.path.insert(0, BASE_DIR)
    from hzk32_reader import HZK32Reader
    reader = HZK32Reader(HZK32_PATH)
    gm = np.zeros((V, 1024), np.float32)
    for idx in range(V):
        ch = i2c.get(idx, "")
        if ch and ch != "<UNK>":
            try: gm[idx] = reader.get_glyph_vector(ch).astype(np.float32)
            except (ValueError, IOError): pass
    np.save(PIX_CACHE, gm)
    return gm


def load_struct(c2i, i2c, V):
    sj = json.load(open(STRUCT_JSON, encoding="utf-8"))
    nopy26 = sj["nopy26"]
    S = np.zeros((V, 26), np.float32)
    for idx in range(V):
        ch = i2c.get(idx, "")
        if ch in nopy26:
            arr = np.asarray(nopy26[ch], np.float32)
            if arr.shape[0] >= 26: S[idx] = arr[:26]
    return S


def load_mined(c2i):
    d = json.load(open(MINED, encoding="utf-8"))
    tr = [(c2i[p["a"]], c2i[p["b"]]) for p in d["train"] if p["a"] in c2i and p["b"] in c2i]
    ho = [(c2i[p["a"]], c2i[p["b"]]) for p in d["heldout"] if p["a"] in c2i and p["b"] in c2i]
    return tr, ho


# ---------------- 模型 ----------------
class PixelEncoder(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


class StructEncoder(nn.Module):
    def __init__(self, in_dim=26, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


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
                if len(gb) == 2 and gb[0] >= 0xB0: top.append(ch)
            except UnicodeEncodeError: continue
        if len(top) >= n: break
    vecs = l2norm(np.stack([E[c2i[c]] for c in top], 0))
    sm = vecs @ vecs.T
    iu = np.triu_indices(len(top), k=1)
    return float(sm[iu].mean())


def pair_avg_cos(E, pairs_global, c2i=None):
    fps = [cosine_sim(E[a], E[b]) for a, b in pairs_global]
    return float(np.mean(fps))


def evaluate(encoder, arm, PIXEL, STRUCT, c2i, train_pairs, heldout_pairs):
    with torch.no_grad():
        if arm == "R":
            E = encoder(torch.from_numpy(PIXEL)).numpy().astype(np.float64)
        elif arm == "S":
            E = encoder(torch.from_numpy(STRUCT)).numpy().astype(np.float64)
        else:
            E = encoder(torch.from_numpy(PIXEL), torch.from_numpy(STRUCT)).numpy().astype(np.float64)
    fa = pair_avg_cos(E, train_pairs, c2i)
    ha = pair_avg_cos(E, heldout_pairs, c2i)
    gb = global_baseline(E, c2i, {v: k for k, v in c2i.items()})
    return fa, ha, gb


def train_arm(arm, encoder, PIXEL_t, STRUCT_t, c2i, train_pairs, heldout_pairs,
              focus_chars_global, train_pairs_local):
    opt = torch.optim.Adam(encoder.parameters(), lr=LR)
    N = PIXEL_t.shape[0]
    fc_t = torch.tensor(focus_chars_global, dtype=torch.long, device=DEVICE)
    print(f"\n=== 训练 Arm {arm} ===")
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
            loss = Lc
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        # 硬负分离阶段: 对443训练对显式推开
        if arm == "R":
            fe = encoder(aug_pixel(PIXEL_t[fc_t]))
        elif arm == "S":
            fe = encoder(aug_struct(STRUCT_t[fc_t]))
        else:
            fe = encoder(aug_pixel(PIXEL_t[fc_t]), aug_struct(STRUCT_t[fc_t]))
        Lf = focus_loss(fe, train_pairs_local)
        opt.zero_grad(); Lf.backward(); opt.step()

        fa, ha, gb = evaluate(encoder, arm, PIXEL_t.numpy(), STRUCT_t.numpy(), c2i, train_pairs, heldout_pairs)
        print(f"  ep{ep+1:02d}  Lc={tot/nb:.3f}  Lf={Lf.item():.3f}  train_avg={fa:+.4f}  heldout_avg={ha:+.4f}  baseline={gb:.4f}")
    fa, ha, gb = evaluate(encoder, arm, PIXEL_t.numpy(), STRUCT_t.numpy(), c2i, train_pairs, heldout_pairs)
    return {"train_avg_cos": fa, "heldout_avg_cos": ha, "global_baseline": gb,
            "margin_train": gb - fa, "margin_heldout": gb - ha}


def main():
    t0 = time.time()
    c2i, i2c, V = load_vocab()
    PIXEL = load_pixels(V)
    STRUCT = load_struct(c2i, i2c, V)
    train_pairs, heldout_pairs = load_mined(c2i)
    print(f"[Data] vocab={V} 训练硬负对={len(train_pairs)} 留出对={len(heldout_pairs)}")

    PIXEL_t, STRUCT_t = torch.from_numpy(PIXEL).to(DEVICE), torch.from_numpy(STRUCT).to(DEVICE)
    focus_chars_global = sorted({c for pr in train_pairs for c in pr})
    loc = {c: k for k, c in enumerate(focus_chars_global)}
    train_pairs_local = [(loc[a], loc[b]) for a, b in train_pairs]
    print(f"[Focus] 训练对涉及 {len(focus_chars_global)} 个字; 局部对 {len(train_pairs_local)}")

    results = {}
    encR = PixelEncoder().to(DEVICE)
    results["R_纯随机_像素"] = train_arm("R", encR, PIXEL_t, STRUCT_t, c2i, train_pairs, heldout_pairs, focus_chars_global, train_pairs_local)
    torch.save(encR.state_dict(), os.path.join(BASE_DIR, "enc_R_pixel.pt"))

    encS = StructEncoder().to(DEVICE)
    results["S_预先结构化_结构码"] = train_arm("S", encS, PIXEL_t, STRUCT_t, c2i, train_pairs, heldout_pairs, focus_chars_global, train_pairs_local)
    torch.save(encS.state_dict(), os.path.join(BASE_DIR, "enc_S_struct.pt"))

    encF = FusionEncoder().to(DEVICE)
    results["F_融合头_像素+结构"] = train_arm("F", encF, PIXEL_t, STRUCT_t, c2i, train_pairs, heldout_pairs, focus_chars_global, train_pairs_local)
    torch.save(encF.state_dict(), os.path.join(BASE_DIR, "enc_F_fusion.pt"))

    # 参考基线
    ref = json.load(open(os.path.join(BASE_DIR, "fuse_struct_v2_result.json"), encoding="utf-8"))
    p64 = ref["reps"]["P_像素64"]
    results["P64_现状LM基线"] = {"train_avg_cos": p64["focus_avg_cos"], "heldout_avg_cos": None,
                                 "global_baseline": p64["global_baseline"],
                                 "margin_train": p64["global_baseline"] - p64["focus_avg_cos"],
                                 "margin_heldout": None}

    print("\n" + "=" * 82)
    print("足量硬负对照: 纯随机像素 vs 预先结构化 vs 融合头 (train/heldout越低越好)")
    print("=" * 82)
    print(f"{'模型':<22}{'train':>8}{'heldout':>9}{'baseline':>10}{'裕度T':>8}{'裕度H':>8}")
    print("-" * 71)
    for k in ["P64_现状LM基线", "R_纯随机_像素", "S_预先结构化_结构码", "F_融合头_像素+结构"]:
        r = results[k]
        tr = f"{r['train_avg_cos']:>+8.4f}"
        ho = f"{r['heldout_avg_cos']:+.4f}" if r["heldout_avg_cos"] is not None else "  N/A"
        mt = f"{r['margin_train']:+.3f}"
        mh = f"{r['margin_heldout']:+.3f}" if r["margin_heldout"] is not None else "  N/A"
        tag = "  ←参考" if "基线" in k else ""
        print(f"{k:<22}{tr}{ho:>9}{r['global_baseline']:>10.4f}{mt:>8}{mh:>8}{tag}")

    out = {"note": "足量硬负(443训练/111留出)纯字形对照: 纯随机 vs 预先结构化 vs 融合头",
           "hyper": dict(BATCH=BATCH, EPOCHS=EPOCHS, LR=LR, TEMP=TEMP, FOCUS_MARGIN=FOCUS_MARGIN, FOCUS_LAMBDA=FOCUS_LAMBDA),
           "n_train_pairs": len(train_pairs), "n_heldout_pairs": len(heldout_pairs),
           "reps": results, "elapsed_sec": round(time.time() - t0, 1)}
    op = os.path.join(BASE_DIR, "glyph_hardneg_result.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[Save] {op}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
