# -*- coding: utf-8 -*-
"""
P2 过渡判别器探针 (Transitional Discriminator Probe) — 修正版
============================================================
关键认知(诊断发现): 形近字在原始特征X上的余弦=0.51(随机对=-0.02), 即"形近字
特征天然相似"是事实。字形判别力 G2 不是"让编码器余弦变低", 而是"编码器保留足够
结构信息, 使字形头 A 能从 Z_a,Z_b 算出 G2(≈0.015, 低=分得开)"。

故正确指标: 头 A 预测的 G2 在留出近形集上是否仍低。这正是 advice 双头原意:
  编码器 E(X->Z) -> 双头
    头 A: 从 (Z_a,Z_b) 预测字形相似度 G2 (冻结/极低LR, 守住判别力)
    头 B: 从 Z 预测语义(声母, 本地可用的语义代理信号)
  解耦: Z 拆 Zg(头A)/Zs(头B)+正交约束, 防语义头把编码器拉偏。

变体:
  V1 Frozen : 编码器=规则G2(头A即G2公式), 只训语义头B. 字形天然保全.
  V2 Merged : 编码器可训(低LR), A+B 多任务. 测字形头A的G2是否被语义拉高.
  V3 Disent : 编码器可训, Zg/Zs拆分+正交. 测解耦是否护住头A的G2.

字形成锚点(advice 5.1): 规则G2留出均值=0.0152 | 随机基线=0.0000.
"""
import os, json, math
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
NPL = r'D:/desktop/学习结果小文件/neuroglyph34_project'
np.random.seed(0)

def load(name): return json.load(open(os.path.join(BASE, name), encoding='utf-8'))
SF   = load('stroke_features.json')
S32  = load(os.path.join(NPL, 'struct32_aligned.json'))
NO   = S32['nopy26']; ALL32 = S32['all32']
TREE = load('tree_data.json')
PAIRS= load('near_pairs_mined.json')
NSHENG = 28

# ---------------- 规则 G2 (字形锚点) ----------------
def r_cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb+1e-12)
def treeSim(a, b):
    ta, tb = TREE.get(a), TREE.get(b)
    if not ta or not tb: return None
    if ta['rad'] != tb['rad']: return 0.0
    det = max(0.0, r_cos(ta['det'], tb['det']))
    return 0.5*det if ta['str'] != tb['str'] else det
def vSim(a, b):
    fa, fb = SF.get(a), SF.get(b)
    if not fa or not fb: return None
    nA, nB = len(fa), len(fb); base = 1 - abs(nA-nB)/max(nA, nB); m = min(nA, nB)
    if m == 0: return 0.0
    gs = []
    for i in range(m):
        sa, sb = fa[i], fb[i]
        lsim = 1 - abs(sa['fl']-sb['fl']); rsim = 1 - abs(sa['rank']-sb['rank'])/max(nA, nB)
        dsim = (sa['sin']*sb['sin']+sa['cos']*sb['cos']+1)/2
        pd = math.hypot(sa['x']-sb['x'], sa['y']-sb['y']); psim = max(0.0, 1-pd/0.4)
        gs.append(0.28*lsim+0.50*rsim+0.12*dsim+0.10*psim)
    return base*(sum(gs)/m)
def relSim2(a, b):
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
    La = [f['L'] for f in fa][:m]; Lb = [f['L'] for f in fb][:m]
    maxL = max(La+Lb)+1e-9; prof = 1.0
    for i in range(m): prof *= max(0.0, 1-abs(La[i]-Lb[i])/maxL)
    ra = [f['rank'] for f in fa][:m]; rb = [f['rank'] for f in fb][:m]
    mt = tot = 0
    for i in range(m):
        for j in range(i+1, m):
            sa = 1 if ra[i]<ra[j] else (0 if ra[i]>ra[j] else 2)
            sb = 1 if rb[i]<rb[j] else (0 if rb[i]>rb[j] else 2)
            tot += 1; mt += (1 if sa==sb else 0)
    rankord = mt/tot if tot else 1.0
    return cnt_f*type_match*prof*rankord
def gSim(a, b):
    t, v, r = treeSim(a, b), vSim(a, b), relSim2(a, b)
    if t is None or v is None or r is None: return None
    return t*v*r

# ---------------- 冻结特征 X (43维) ----------------
def vgeom(c):
    s = SF.get(c); n = len(s) if s else 0
    if n == 0: return [0.0]*17
    Ls=[f['L'] for f in s]; rk=[f['rank'] for f in s]; xs=[f['x'] for f in s]; ys=[f['y'] for f in s]
    si=[f['sin'] for f in s]; co=[f['cos'] for f in s]
    aspect = (max(xs)-min(xs)+1e-6)/(max(ys)-min(ys)+1e-6)
    t5L=(Ls[:5]+[0.0]*5)[:5]; t5r=(rk[:5]+[0.0]*5)[:5]
    return [n/20.0, aspect, float(np.mean(Ls)), float(np.mean(si)), float(np.mean(co)),
            float(np.mean(xs)), float(np.mean(ys))] + t5L + t5r

vocab = [c for c in NO if c in SF and c in TREE and c in ALL32 and '一'<=c<='鿿']
Xb = np.array([NO[c] + vgeom(c) for c in vocab], dtype=float)
Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6)
cidx = {c:i for i,c in enumerate(vocab)}
def sheng_onehot(c):
    idx = int(ALL32[c][26]) if 26 < len(ALL32[c]) else 0
    v = np.zeros(NSHENG); v[idx] = 1.0; return v
Y = np.array([sheng_onehot(c) for c in vocab], dtype=float)
print(f'vocab={len(vocab)} | 语义目标=声母({NSHENG}类)')

held = [p for p in PAIRS['heldout'] if p['a'] in cidx and p['b'] in cidx]
trn  = [p for p in PAIRS['train']  if p['a'] in cidx and p['b'] in cidx]
def rule_g2(pairs):
    vs = [gSim(p['a'], p['b']) for p in pairs]; vs=[v for v in vs if v is not None]
    return float(np.mean(vs))
G2_HELD = rule_g2(held)
rng = np.random.default_rng(0)
rand_pairs = [{'a': vocab[i], 'b': vocab[j]} for i,j in rng.integers(0,len(vocab),(len(held),2)) if i!=j]
RAND_BASE = rule_g2(rand_pairs)
print(f'字形成锚点: 规则G2(留出)={G2_HELD:.4f} | 随机基线={RAND_BASE:.4f}')

n_test = int(len(vocab)*0.2)
perm = rng.permutation(len(vocab))
test_idx, train_idx = perm[:n_test], perm[n_test:]
Xtr, Ytr = Xb[train_idx], Y[train_idx]
Xte, Yte = Xb[test_idx], Y[test_idx]

# ---------------- 工具 ----------------
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def softmax(z): e=z-z.max(-1,keepdims=True); e=np.exp(e); return e/e.sum(-1,keepdims=True)

# 头 A: 从 (Za,Zb) 预测 G2. phi = [Za, Zb, (Za-Zb)^2] -> 线性 -> sigmoid
def headA_fwd(Za, Zb, Wa, ba):
    feat = np.concatenate([Za, Zb, (Za-Zb)**2], -1)
    return float(sigmoid(feat @ Wa + ba)[0])
def headA_grad(feat, out, g2, Wa):
    """返回 dL/dWa, dL/dba, dL/dfeat(96,); L=(out-g2)^2 ; out,g2 标量"""
    dl_dout = 2*(out-g2)
    dout_du = out*(1-out)
    du = dl_dout*dout_du
    dWa = du*feat[:,None]
    dba = du
    dfeat = du*Wa.ravel()
    return dWa, dba, dfeat

def enc_full(W, X): return X@W
def enc_split(Wg, Ws, X): return X@Wg, X@Ws

def sem_metric(enc_fn, Wb):
    pte = softmax(enc_fn(Xte)@Wb)
    top1 = float(np.mean(pte.argmax(-1) == Yte.argmax(-1)))
    cosv = float(np.mean(np.sum(Yte*pte, -1)))   # 分布吻合度(而非余弦, 避免one-hot余弦歧义)
    return top1, cosv

def g2_pred_mean(enc_fn, Wa, ba, pairs):
    """头A在给定对上的预测G2均值(字形保全度: 越低越分得开)."""
    vs = []
    for p in pairs:
        Za = enc_fn(Xb[[cidx[p['a']]]])[0]; Zb = enc_fn(Xb[[cidx[p['b']]]])[0]
        vs.append(float(headA_fwd(Za, Zb, Wa, ba)))
    return float(np.mean(vs))

# ---------------- 预训练 编码器+头A (字形) ----------------
def pretrain_glyph(K=32, epochs=300, lr_e=0.01, lr_a=0.02):
    D = Xb.shape[1]
    W = np.random.randn(D, K)*0.1
    Wa = np.random.randn(3*K, 1)*0.1; ba = np.zeros(1)
    pidx = [(cidx[p['a']], cidx[p['b']], gSim(p['a'],p['b'])) for p in trn]
    pidx = [(a,b,g) for a,b,g in pidx if g is not None]
    for ep in range(epochs):
        dW=np.zeros_like(W); dWa=np.zeros_like(Wa); dba=np.zeros_like(ba)
        for a,b,g in pidx:
            Za=Xb[a]@W; Zb=Xb[b]@W
            feat=np.concatenate([Za,Zb,(Za-Zb)**2])
            out=float(sigmoid(feat@Wa+ba)[0])
            dWa_i, dba_i, dfeat = headA_grad(feat, out, g, Wa)
            dWa+=dWa_i; dba+=dba_i
            dZa = dfeat[:K]      + dfeat[2*K:]*2*(Za-Zb)
            dZb = dfeat[K:2*K]   - dfeat[2*K:]*2*(Za-Zb)
            dW += np.outer(Xb[a], dZa) + np.outer(Xb[b], dZb)
        W  -= lr_e*(dW/len(pidx) + 1e-4*W)
        Wa -= lr_a*(dWa/len(pidx)); ba -= lr_a*(dba/len(pidx))
    return W, Wa, ba

# 在 *冻结* 编码器 W(列宽=Kf) 上, 单独把字形头A(V3用)拟合到G2. 供V3拆列起点基准.
def fit_headA_static(W, Kf, epochs=300, lr_a=0.02):
    Wa = np.random.randn(3*Kf, 1)*0.1; ba = np.zeros(1)
    pidx=[(cidx[p['a']],cidx[p['b']],gSim(p['a'],p['b'])) for p in trn]
    pidx=[(a,b,g) for a,b,g in pidx if g is not None]
    for ep in range(epochs):
        dWa=np.zeros_like(Wa); dba=np.zeros_like(ba)
        for a,b,g in pidx:
            Za=Xb[a]@W; Zb=Xb[b]@W
            feat=np.concatenate([Za,Zb,(Za-Zb)**2])
            out=float(sigmoid(feat@Wa+ba)[0])
            dWa_i,dba_i,dfeat=headA_grad(feat, out, g, Wa)
            dWa+=dWa_i; dba+=dba_i
        Wa -= lr_a*(dWa/len(pidx)); ba -= lr_a*(dba/len(pidx))
    return Wa, ba

# ---------------- V2: 合并(编码器可训, A+B多任务) ----------------
def train_merged(W, Wa, ba, epochs=100, lr_e=0.003, lr_a=0.01, lr_b=0.05, wb=None):
    N,D = Xb.shape; K=W.shape[1]; Kc=Y.shape[1]
    if wb is None: wb=np.random.randn(K,Kc)*0.1
    pidx=[(cidx[p['a']],cidx[p['b']],gSim(p['a'],p['b'])) for p in trn]
    pidx=[(a,b,g) for a,b,g in pidx if g is not None]
    for ep in range(epochs):
        # 语义头 B (全vocab, Z=X@W)
        Z=Xb@W; p=softmax(Z@wb); dL=(p-Y)/N
        wb -= lr_b*(Z.T@dL); dZ=dL@wb.T
        W -= lr_e*(Xb.T@dZ + 1e-4*W)
        # 字形头 A
        dW=np.zeros_like(W); dWa=np.zeros_like(Wa); dba=np.zeros_like(ba)
        for a,b,g in pidx:
            Za=Xb[a]@W; Zb=Xb[b]@W
            feat=np.concatenate([Za,Zb,(Za-Zb)**2])
            out=float(sigmoid(feat@Wa+ba)[0])
            dWa_i,dba_i,dfeat=headA_grad(feat, out, g, Wa)
            dWa+=dWa_i; dba+=dba_i
            dZa=dfeat[:K]+dfeat[2*K:]*2*(Za-Zb); dZb=dfeat[K:2*K]-dfeat[2*K:]*2*(Za-Zb)
            dW+=np.outer(Xb[a],dZa)+np.outer(Xb[b],dZb)
        W -= lr_e*(dW/len(pidx)+1e-4*W)
        Wa -= lr_a*(dWa/len(pidx)); ba -= lr_a*(dba/len(pidx))
    return W, Wa, ba, wb

# ---------------- V3: 解耦(Zg/Zs拆分+正交) ----------------
def train_disentangled(Wg, Ws, Wa, ba, epochs=100, lr_e=0.003, lr_a=0.01, lr_b=0.05, mu=0.3):
    N,D=Xb.shape; K=32; Kh=K//2; Kc=Y.shape[1]
    wb=np.random.randn(Kh,Kc)*0.1
    pidx=[(cidx[p['a']],cidx[p['b']],gSim(p['a'],p['b'])) for p in trn]
    pidx=[(a,b,g) for a,b,g in pidx if g is not None]
    for ep in range(epochs):
        # 语义头 B (Zs)
        Zs=Xb@Ws; p=softmax(Zs@wb); dL=(p-Y)/N
        wb -= lr_b*(Zs.T@dL); dZs=dL@wb.T
        Ws -= lr_e*(Xb.T@dZs + 1e-4*Ws)
        # 字形头 A (Zg)
        dWg=np.zeros_like(Wg); dWa=np.zeros_like(Wa); dba=np.zeros_like(ba)
        for a,b,g in pidx:
            Zga=Xb[a]@Wg; Zgb=Xb[b]@Wg
            feat=np.concatenate([Zga,Zgb,(Zga-Zgb)**2])
            out=float(sigmoid(feat@Wa+ba)[0])
            dWa_i,dba_i,dfeat=headA_grad(feat, out, g, Wa)
            dWa+=dWa_i; dba+=dba_i
            dZga=dfeat[:Kh]+dfeat[2*Kh:]*2*(Zga-Zgb); dZgb=dfeat[Kh:2*Kh]-dfeat[2*Kh:]*2*(Zga-Zgb)
            dWg+=np.outer(Xb[a],dZga)+np.outer(Xb[b],dZgb)
        Wg -= lr_e*(dWg/len(pidx)+1e-4*Wg)
        Wa -= lr_a*(dWa/len(pidx)); ba -= lr_a*(dba/len(pidx))
        # 正交约束(归一化Z, 防溢出): 推开 Zg 与 Zs 子空间
        Zg_n=(Xb@Wg); Zs_n=(Xb@Ws)
        Zg_n/=np.linalg.norm(Zg_n,axis=1,keepdims=True)+1e-8
        Zs_n/=np.linalg.norm(Zs_n,axis=1,keepdims=True)+1e-8
        G=Zg_n.T@Zs_n
        Lorth=mu*np.sum(G*G)
        dZg_n=2*mu*(Zs_n@G.T)/N; dZs_n=2*mu*(Zg_n@G.T)/N
        Wg -= lr_e*(Xb.T@dZg_n); Ws -= lr_e*(Xb.T@dZs_n)
    return Wg, Ws, Wa, ba, wb

# ================= 跑三变体 =================
print('\n========== V1 Frozen (字形=规则G2, 只训语义头B) ==========')
Wb1 = np.random.randn(Xb.shape[1], NSHENG)*0.1
for _ in range(150):
    Z=Xb; p=softmax(Z@Wb1); dL=(p-Y)/len(vocab); Wb1 -= 0.05*(Z.T@dL)
top1_v1, cosv_v1 = sem_metric(lambda X: X, Wb1)
print(f'  语义头 top1=声母 {top1_v1*100:.1f}% | 分布吻合={cosv_v1:.3f}')
print(f'  字形: 规则G2(留出)={G2_HELD:.4f} (冻结, 天然保全)')

print('\n========== 预训练 编码器+头A (V2/V3 字形基线) ==========')
Wpre, Wapre, bapre = pretrain_glyph(K=32, epochs=300, lr_e=0.01, lr_a=0.02)
base_g2 = g2_pred_mean(lambda X: X@Wpre, Wapre, bapre, held)
print(f'  预训练后 头A预测G2(留出)={base_g2:.4f}  (规则锚={G2_HELD:.4f})')

print('\n========== V2 Merged (编码器可训, A+B多任务) ==========')
W2, Wa2, ba2, Wb2 = train_merged(Wpre.copy(), Wapre.copy(), bapre.copy(), epochs=100,
                                 lr_e=0.003, lr_a=0.01, lr_b=0.05)
g2_v2 = g2_pred_mean(lambda X: X@W2, Wa2, ba2, held)
top1_v2, cosv_v2 = sem_metric(lambda X: X@W2, Wb2)
print(f'  语义头 top1=声母 {top1_v2*100:.1f}% | 分布吻合={cosv_v2:.3f}')
print(f'  字形(头A G2): 基线 {base_g2:.4f} -> 训后 {g2_v2:.4f}  (Δ={g2_v2-base_g2:+.4f}, 越大越退化)')

print('\n========== V3 Disentangled (Zg/Zs拆分+正交, 同起点) ==========')
Wg0, Ws0 = Wpre[:, :16].copy(), np.random.randn(Xb.shape[1], 16)*0.1
Wav3, bav3 = fit_headA_static(Wg0, Kf=16)            # 在冻结拆列Zg上独立拟合字形头A
base_g2_v3 = g2_pred_mean(lambda X: X@Wg0, Wav3, bav3, held)
Wg3, Ws3, Wa3, ba3, Wb3 = train_disentangled(Wg0.copy(), Ws0.copy(), Wav3.copy(), bav3.copy(),
                                              epochs=100, lr_e=0.003, lr_a=0.01, lr_b=0.05, mu=0.3)
g2_v3 = g2_pred_mean(lambda X: X@Wg3, Wa3, ba3, held)
top1_v3, cosv_v3 = sem_metric(lambda X: X@Ws3, Wb3)
print(f'  语义头 top1=声母 {top1_v3*100:.1f}% | 分布吻合={cosv_v3:.3f}')
print(f'  字形(头A Zg G2): 拆列基线 {base_g2_v3:.4f} -> 训后 {g2_v3:.4f}  (Δ={g2_v3-base_g2_v3:+.4f})')

# ================= 汇总 =================
summary = {
    'anchor': {'rule_G2_heldout': G2_HELD, 'random_baseline': RAND_BASE,
               'pretrain_headA_g2_heldout': base_g2},
    'sem_target': '声母(拼音首字母,28类); 注: 本地无HowNet/义原, 此语音代理替代义类',
    'V1_frozen': {'glyph_G2': G2_HELD, 'sem_top1': top1_v1, 'sem_distmatch': cosv_v1},
    'V2_merged': {'glyph_baseline': base_g2, 'glyph_after': g2_v2, 'glyph_delta': g2_v2-base_g2,
                  'sem_top1': top1_v2, 'sem_distmatch': cosv_v2},
    'V3_disentangled': {'glyph_baseline': base_g2_v3, 'glyph_after': g2_v3, 'glyph_delta': g2_v3-base_g2_v3,
                        'sem_top1': top1_v3, 'sem_distmatch': cosv_v3},
}
outp = os.path.join(BASE, 'glyph_transition_probe_result.json')
json.dump(summary, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('\n========== 结论速览 ==========')
print(f'  字形成锚点: 规则G2(留出)={G2_HELD:.4f} | 随机基线={RAND_BASE:.4f}')
print(f'  V1 冻结   : 字形保全({G2_HELD:.4f}), 语义(top1声母)={top1_v1*100:.1f}%')
print(f'  V2 合并   : 字形Δ={g2_v2-base_g2:+.4f} {"⚠退化" if g2_v2-base_g2>0.005 else "✓保全"}, 语义={top1_v2*100:.1f}%')
print(f'  V3 解耦   : 字形Δ={g2_v3-base_g2_v3:+.4f} {"⚠退化" if g2_v3-base_g2_v3>0.005 else "✓保全"}, 语义={top1_v3*100:.1f}%')
print(f'  [存] {outp}')
