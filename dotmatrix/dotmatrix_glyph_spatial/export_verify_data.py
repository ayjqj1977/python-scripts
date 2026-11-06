# -*- coding: utf-8 -*-
"""导出验证用字形数据 + Python 官方 G2 基准值"""
import json, math, os
BASE="D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
TREE=json.load(open(os.path.join(BASE,"tree_data.json"),encoding="utf-8"))
SF=json.load(open(os.path.join(BASE,"stroke_features.json"),encoding="utf-8"))
NO=json.load(open(os.path.join("D:/desktop/学习结果小文件/neuroglyph34_project","struct32_aligned.json"),encoding="utf-8"))["nopy26"]

def cosine(a,b):
    d=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(y*y for y in b))
    return d/(na*nb+1e-12)
def treeSim(a,b):
    ta,tb=TREE.get(a),TREE.get(b)
    if not ta or not tb:return None
    if ta["rad"]!=tb["rad"]:return 0.0
    det=max(0.0,cosine(ta["det"],tb["det"]))
    return 0.5*det if ta["str"]!=tb["str"] else det
def vSim(a,b):
    fa,fb=SF.get(a),SF.get(b)
    if not fa or not fb:return None
    nA,nB=len(fa),len(fb);base=1-abs(nA-nB)/max(nA,nB);m=min(nA,nB)
    if m==0:return 0.0
    gs=[]
    for i in range(m):
        sa,sb=fa[i],fb[i]
        lsim=1-abs(sa["fl"]-sb["fl"])
        rsim=1-abs(sa["rank"]-sb["rank"])/max(nA,nB)
        dsim=(sa["sin"]*sb["sin"]+sa["cos"]*sb["cos"]+1)/2
        pd=math.hypot(sa["x"]-sb["x"],sa["y"]-sb["y"])
        psim=max(0.0,1-pd/0.4)
        gs.append(0.28*lsim+0.50*rsim+0.12*dsim+0.10*psim)
    return base*(sum(gs)/m)
def relSim2(a,b):
    fa,fb=SF.get(a),SF.get(b)
    if not fa or not fb:return None
    nA,nB=len(fa),len(fb);m=min(nA,nB)
    if m==0:return 0.0
    cnt_f=1.0-0.6*abs(nA-nB)/max(nA,nB,1)
    v,v2=NO.get(a),NO.get(b)
    if v and v2:
        cap=min(int(v[9]),10);cap2=min(int(v2[9]),10)
        ta=[int(x) for x in v[16:16+cap]];tb=[int(x) for x in v2[16:16+cap2]]
        mm=min(len(ta),len(tb));type_match=sum(1 for i in range(mm) if ta[i]==tb[i])/mm
    else:type_match=0.0
    La=[f["L"] for f in fa][:m];Lb=[f["L"] for f in fb][:m]
    maxL=max(La+Lb)+1e-9;prof=1.0
    for i in range(m):prof*=max(0.0,1-abs(La[i]-Lb[i])/maxL)
    ra=[f["rank"] for f in fa][:m];rb=[f["rank"] for f in fb][:m]
    mt=tot=0
    for i in range(m):
        for j in range(i+1,m):
            sa=1 if ra[i]<ra[j] else (0 if ra[i]>ra[j] else 2)
            sb=1 if rb[i]<rb[j] else (0 if rb[i]>rb[j] else 2)
            tot+=1;mt+=1 if sa==sb else 0
    rankord=mt/tot if tot else 1.0
    return cnt_f*type_match*prof*rankord
def gSim(a,b):
    t,v,r=treeSim(a,b),vSim(a,b),relSim2(a,b)
    if t is None or v is None or r is None:return None
    return t*v*r

words=["牛","羊","猫","狗","末","未","树","江","河"]
chars={}
for c in words:
    t=TREE.get(c);sf=SF.get(c);s32=NO.get(c)
    chars[c]={"rad":t.get("rad"),"str":t.get("str"),"det":t.get("det"),"strokes":sf,"s32":s32}
pairs={}
for i in range(len(words)):
    for j in range(i+1,len(words)):
        a,b=words[i],words[j]
        g=gSim(a,b)
        pairs[f"{a}|{b}"]=round(g,6) if g is not None else None
json.dump({"chars":chars,"pairs":pairs,"words":words},
          open(os.path.join(BASE,"verify_g2_data.json"),"w",encoding="utf-8"),ensure_ascii=False)
print("expected pairs (Python 官方):")
for k,v in pairs.items():print("  ",k,v)
