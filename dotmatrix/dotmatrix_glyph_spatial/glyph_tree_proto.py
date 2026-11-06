# 组合树原型 (composition-tree glyph discriminator)
# 层级: 部首(radical) -> 结构(structure) -> 细节(detail, 训练的小MLP)
# sim_tree(a,b):
#   radical 不同 -> 0 (不同根, 必为不同字)
#   radical 同, structure 不同 -> 0.5 * relu(cos_det)
#   radical 同, structure 同 -> relu(cos_det)
# 同时导出 emb_T(64维): 部首/结构/细节 分占子空间, 部首主导 -> 可直接塞进现有app
import os, json, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)

c2i = json.load(open(os.path.join(BASE, "char_to_idx_full.json"), encoding="utf-8"))
i2c = {v: k for k, v in c2i.items()}
struct = json.load(open(os.path.join(BASE, "struct_tree_fields.json"), encoding="utf-8"))
pairs = json.load(open(os.path.join(BASE, "near_pairs_mined.json"), encoding="utf-8"))
emb_prev = json.load(open(os.path.join(BASE, "glyph_embeddings.json"), encoding="utf-8"))

# ---- 组装每字特征 (只取词表内有结构码的字) ----
vocab = [i2c[i] for i in range(len(c2i)) if i2c[i] in struct]
idx_of = {ch: k for k, ch in enumerate(vocab)}          # 局部下标
n = len(vocab)
radical = np.array([struct[ch]["radical_id"] for ch in vocab], dtype=np.int64)
structure = np.array([struct[ch]["structure"] for ch in vocab], dtype=np.int64)
detail = np.array([struct[ch]["four_corner"] + struct[ch]["five_stroke"] +
                   struct[ch]["stroke_order"] + [struct[ch]["total_strokes"]]
                   for ch in vocab], dtype=np.float32)   # 20维
print(f"[data] 词表内有结构码 {n} 字")

# 部首/结构 固定正交子空间 (近似, 高维随机近正交)
rng = np.random.default_rng(0)
radicals = sorted(set(radical.tolist()))
strs = sorted(set(structure.tolist()))
rad_vec = {r: rng.standard_normal(24) for r in radicals}
str_vec = {s: rng.standard_normal(8) for s in strs}
for d in rad_vec.values():
    d /= np.linalg.norm(d)
for d in str_vec.values():
    d /= np.linalg.norm(d)

# ---- 细节支路: 小MLP, 对比损失训练 ----
class DetailMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32))
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

model = DetailMLP().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
detail_t = torch.tensor(detail, device=DEVICE)

# 硬负对 (局部下标)
hard_local = []
for p in pairs["train"]:
    a, b = p["a"], p["b"]
    if a in idx_of and b in idx_of:
        hard_local.append((idx_of[a], idx_of[b]))
hard_local = torch.tensor(hard_local, dtype=torch.long, device=DEVICE) if hard_local else None
print(f"[train] 硬负对(局部) {len(hard_local) if hard_local is not None else 0}")

# 自增强: 每字两视图
aug = detail + rng.standard_normal(detail.shape).astype(np.float32) * 0.1
aug2 = detail + rng.standard_normal(detail.shape).astype(np.float32) * 0.1
aug_t = torch.tensor(np.concatenate([aug, aug2], 0), device=DEVICE)
N = n
bs = 256
for ep in range(25):
    perm = torch.randperm(2 * N, device=DEVICE)
    tot = 0.0; cnt = 0
    model.train()
    for s in range(0, 2 * N, bs):
        idx = perm[s:s + bs]
        v = model(aug_t[idx])
        # InfoNCE: 同一下标的正负视图配对
        # 把 2N 拆成两半: 前半=view1, 后半=view2, 同 i 配对
        lab = idx % N
        # 构造正负: 对每个样本, 正=同 lab 的另一半, 负=同 batch 其它
        sim = v @ v.T / 0.1
        pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
        # 去掉自身
        eye = torch.eye(len(idx), device=DEVICE)
        pos_mask = pos_mask - eye
        neg_mask = 1 - pos_mask - eye
        pos = (sim * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
        neg = torch.logsumexp(sim * neg_mask - (1 - neg_mask) * 1e4, dim=1)
        loss = -(pos - neg).mean()
        if hard_local is not None:
            ha = model(detail_t[hard_local[:, 0]])
            hb = model(detail_t[hard_local[:, 1]])
            hc = (ha * hb).sum(1)
            loss = loss + F.relu(0.5 - hc).mean() * 0.5
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); cnt += 1
    if ep % 5 == 0 or ep == 24:
        print(f"  ep{ep:02d} loss={tot/cnt:.4f}")

model.eval()
emb_det = model(detail_t).detach().cpu().numpy()          # (n,32)
print(f"[detail] 训练完成, emb_det {emb_det.shape}")

# ---- 组合 emb_T (64维, 部首主导) ----
WR, WS, WD = 1.506, 0.949, 0.632   # 使 部首占 ~0.85 范数 -> 不同部首 cos<=~0.15
emb_T = np.zeros((n, 64), dtype=np.float32)
emb_T[:, :24] = np.stack([WR * rad_vec[r] for r in radical])[:]
emb_T[:, 24:32] = np.stack([WS * str_vec[s] for s in structure])
emb_T[:, 32:] = WD * emb_det
# 归一化
emb_T = emb_T / np.linalg.norm(emb_T, axis=1, keepdims=True)

# ---- 规则版 sim_tree (精确三层) ----
def cos_det(i, j):
    return float(emb_det[i] @ emb_det[j])

def sim_tree_rule(i, j):
    if radical[i] != radical[j]:
        return 0.0
    if structure[i] != structure[j]:
        return 0.5 * max(cos_det(i, j), 0.0)
    return max(cos_det(i, j), 0.0)

# ---- 评估 ----
def cos_mode(mode, a, b):
    ea = emb_prev["embed"][mode].get(a); eb = emb_prev["embed"][mode].get(b)
    if ea is None or eb is None: return None
    a_ = np.array(ea); b_ = np.array(eb)
    return float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-8))

def t_cos(a, b):
    ia, ib = idx_of.get(a), idx_of.get(b)
    if ia is None or ib is None: return None
    return float(emb_T[ia] @ emb_T[ib])

def s_rule(a, b):
    ia, ib = idx_of.get(a), idx_of.get(b)
    if ia is None or ib is None: return None
    return sim_tree_rule(ia, ib)

print("\n=== 留出近形字对 (111对, 越低越好) ===")
held = pairs["heldout"]
def mean_mode(fn):
    vs = [fn(p["a"], p["b"]) for p in held]
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)), len(vs)
for name, fn in [("S结构化", lambda a,b: cos_mode("S",a,b)),
                 ("R纯像素", lambda a,b: cos_mode("R",a,b)),
                 ("F融合", lambda a,b: cos_mode("F",a,b)),
                 ("T规则树", s_rule),
                 ("T嵌入", t_cos)]:
    m, c = mean_mode(fn)
    print(f"  {name:<10} 均值={m:+.4f}  (n={c})")

print("\n=== spot-check 对照表 ===")
spot = [("风","冈"),("上","月"),("日","曰"),("王","玉"),("后","右"),
        ("末","未"),("干","千"),("风","凤"),("田","由"),("刀","力")]
rows = []
for a, b in spot:
    s = cos_mode("S", a, b); r = cos_mode("R", a, b); t = s_rule(a, b)
    te = t_cos(a, b)
    rows.append({"pair": f"{a}/{b}", "S": s, "R": r, "T_rule": t, "T_emb": te})
    print(f"  {a}/{b:<3} S={s*100:5.1f}%  R={r*100:5.1f}%  T规则={t*100:5.1f}%  T嵌入={te*100:5.1f}%")

# 保存
out = {
    "spot": rows,
    "heldout_mean": {m: mean_mode(fn)[0] for m, fn in
                     [("S", lambda a,b: cos_mode("S",a,b)),("R", lambda a,b: cos_mode("R",a,b)),
                      ("F", lambda a,b: cos_mode("F",a,b)),("T_rule", s_rule),("T_emb", t_cos)]},
    "n_tree_chars": n,
}
json.dump(out, open(os.path.join(BASE, "glyph_tree_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# 存 tree_data.json: 每字 {radical_id, structure, emb_det(32)} -> app 用规则精确计算 T
tree_data = {vocab[i]: {"rad": int(radical[i]), "str": int(structure[i]),
                        "det": [round(float(x), 4) for x in emb_det[i]]}
             for i in range(n)}
json.dump(tree_data, open(os.path.join(BASE, "tree_data.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print(f"\n[done] 结果存 glyph_tree_result.json, tree_data.json ({n} 字, 供app精确计算T)")
