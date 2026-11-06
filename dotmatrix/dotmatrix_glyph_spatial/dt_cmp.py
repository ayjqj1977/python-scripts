# -*- coding: utf-8 -*-
"""
Q2 验证：几何特征(det/s32)贡献甚微，是不是因为逻辑回归太浅、抓不到非线性组合？
================================================================================
在同一套 C_full 特征(283维)上，对「物(B)类内部 18 中类」任务对比：
  1) 多类 softmax 逻辑回归(LR, 已用)  —— 浅线性
  2) 浅决策树 DecisionTree(max_depth=4~6, 可解释) —— 能抓特征交互/非线性
看两件事：
  (a) DT 准确率是否显著高于 LR  → 若是，说明"浅"确实埋没了信号
  (b) DT 的特征重要性里，几何块(s32+det+笔画+结构)占比 vs 部首块占比
      对比 LR 的 |权重| 块占比 → 几何特征是否在非线性模型里"说话"
"""
import json, os, random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

BASE  = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
BASE2 = r"D:/desktop/学习结果小文件/neuroglyph34_project"

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

tree = load(os.path.join(BASE, "tree_data.json"))
sf   = load(os.path.join(BASE, "stroke_features.json"))
no   = load(os.path.join(BASE2, "struct32_aligned.json"))["nopy26"]
cilin = load(os.path.join(BASE, "cilin_index.json"))["char_codes"]

samples = []
rad_vocab, str_vocab = {}, {}
for ch, cc in cilin.items():
    if cc["src"] == "none": continue
    t = tree.get(ch); s = sf.get(ch); v = no.get(ch)
    if not (t and s and v): continue
    rad_vocab.setdefault(t["rad"], len(rad_vocab))
    str_vocab.setdefault(t["str"], len(str_vocab))
    samples.append((ch, cc["codes"][0][0], cc["codes"][0][0:2], t["rad"], t["str"],
                    len(s), np.array(v, float), np.array(t["det"], float)))

n_rad = len(rad_vocab); n_str = len(str_vocab)
s32 = np.array([s[6] for s in samples], float); det = np.array([s[7] for s in samples], float)
scnt = np.array([s[5] for s in samples], float)
rad_oh = np.zeros((len(samples), n_rad)); str_oh = np.zeros((len(samples), n_str))
for i, s in enumerate(samples):
    rad_oh[i, rad_vocab[s[3]]] = 1; str_oh[i, str_vocab[s[4]]] = 1
def z(X): X=np.asarray(X,float); return (X-X.mean(0))/(X.std(0)+1e-9)
X = np.hstack([z(rad_oh), z(str_oh), z(scnt).reshape(-1,1), z(s32), z(det)])  # (N,283)

# 块边界
i_rad_end=n_rad; i_str_end=n_rad+n_str; i_sc=i_str_end; i_s32=i_sc+1; i_det=i_s32+26; i_end=i_det+32
RAD=slice(0,i_rad_end); GEOM=slice(i_str_end,i_end)  # 几何 = 结构+笔画+s32+det

# 取 B 类子集
idx=[i for i,s in enumerate(samples) if s[1]=="B"]
mids=sorted(set(samples[i][2] for i in idx)); mi={m:k for k,m in enumerate(mids)}
y=np.array([mi[samples[i][2]] for i in idx]); Xb=X[idx]

def split(n,seed):
    r=random.Random(seed); ix=list(range(n)); r.shuffle(ix); nt=int(.2*n); return ix[:nt],ix[nt:]

# LR (复用同参数)
def softmax(Z): Z-=Z.max(1,keepdims=True); e=np.exp(Z); return e/e.sum(1,keepdims=True)
def train_lr(Xtr,ytr,nc,iters=3000,lr=0.02,reg=0.5):
    N,D=Xtr.shape; W=np.zeros((D,nc)); pr=np.bincount(ytr,minlength=nc).astype(float)
    b=np.log(pr/pr.sum()+1e-9); mW=np.zeros_like(W); mb=np.zeros_like(b)
    Y=np.zeros((N,nc)); Y[np.arange(N),ytr]=1
    for t in range(1,iters+1):
        P=softmax(Xtr@W+b); gW=(Xtr.T@(P-Y))/N+reg*W; gb=(P-Y).sum(0)/N
        mW=0.9*mW+0.1*gW; mb=0.9*mb+0.1*gb
        vW=0.999*mW*mW+0.001*(gW**2); vb=0.999*mb*mb+0.001*(gb**2)
        mWh=mW/(1-0.9**t); mbh=mb/(1-0.9**t); vWh=vW/(1-0.999**t); vbh=vb/(1-0.999**t)
        W-=lr*mWh/(np.sqrt(vWh)+1e-8); b-=lr*mbh/(np.sqrt(vbh)+1e-8)
    return W,b
def lr_acc(Xte,yte,W,b): return float((softmax(Xte@W+b).argmax(1)==yte).mean())
def block_share(imp,sl): 
    tot=imp.sum(); return float(imp[sl].sum()/tot) if tot>0 else 0.0

print("物(B)类内部 18 中类任务：LR vs 浅决策树  (5-seed 按字80/20)")
lr_accs=[]; dt4=[]; dt6=[]
lr_rad=[]; lr_geom=[]; dt4_rad=[]; dt4_geom=[]
for sd in (0,1,2,3,4):
    te,tr=split(len(idx),sd)
    # LR
    W,b=train_lr(Xb[tr],y[tr],len(mids)); lr_accs.append(lr_acc(Xb[te],y[te],W,b))
    lr_imp=np.mean(np.abs(W),axis=1)
    lr_rad.append(block_share(lr_imp,RAD)); lr_geom.append(block_share(lr_imp,GEOM))
    # DT depth4
    dt=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,random_state=sd)
    dt.fit(Xb[tr],y[tr]); dt4.append(float(dt.score(Xb[te],y[te])))
    dt4_rad.append(block_share(dt.feature_importances_,RAD)); dt4_geom.append(block_share(dt.feature_importances_,GEOM))
    # DT depth6
    dt6c=DecisionTreeClassifier(max_depth=6,min_samples_leaf=15,random_state=sd)
    dt6c.fit(Xb[tr],y[tr]); dt6.append(float(dt6c.score(Xb[te],y[te])))

lr_m=np.mean(lr_accs); dt4_m=np.mean(dt4); dt6_m=np.mean(dt6)
mc=np.bincount(y[split(len(idx),0)[1]]).argmax(); base=float((y[split(len(idx),0)[0]]==mc).mean())
print(f"  基线(猜最多中类)      : {base*100:.1f}%")
print(f"  LR(线性, 浅)          : {lr_m*100:.1f}%  部首块{np.mean(lr_rad)*100:.1f}% 几何块{np.mean(lr_geom)*100:.1f}%")
print(f"  DT(max_depth=4)       : {dt4_m*100:.1f}%  部首块{np.mean(dt4_rad)*100:.1f}% 几何块{np.mean(dt4_geom)*100:.1f}%")
print(f"  DT(max_depth=6)       : {dt6_m*100:.1f}%")
print(f"  DT4 - LR = {(dt4_m-lr_m)*100:+.1f}pp   → {'DT显著更强, 浅LR埋没信号' if dt4_m-lr_m>0.03 else 'DT未胜过LR, 几何确实弱'}")

out={"task":"B_midclass","baseline":base,
     "LR":{"acc":lr_m,"rad_share":float(np.mean(lr_rad)),"geom_share":float(np.mean(lr_geom))},
     "DT4":{"acc":dt4_m,"rad_share":float(np.mean(dt4_rad)),"geom_share":float(np.mean(dt4_geom))},
     "DT6":{"acc":dt6_m},
     "conclusion":"DT未显著胜过LR则几何特征确实弱(非模型太浅)"}
json.dump(out,open(os.path.join(BASE,"dt_cmp_result.json"),"w",encoding="utf-8"),
          ensure_ascii=False,indent=1)
print("\n[OK] -> dt_cmp_result.json")
