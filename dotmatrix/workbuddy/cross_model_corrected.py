# -*- coding: utf-8 -*-
"""修正版全量复核：统一测"字形区分力指数"。
重点修正：v11/v12 的 glyph_matrix 是真·字形嵌入；top500 用 chars_top500.txt 顺序；
云端训练用嵌套 vocab_large.json。
"""
import os, sys, json
import numpy as np
import torch

BASE = "D:/desktop/学习结果小文件"
SP = os.path.join(BASE, "dotmatrix_glyph_spatial")

focus_pairs = [("己","已"),("己","巳"),("已","巳"),("未","末"),("干","千"),
               ("干","于"),("千","于"),("大","太"),("王","玉"),("日","曰")]

def cos(a, b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if na<1e-12 or nb<1e-12: return 0.0
    return float(np.dot(a,b)/(na*nb))

def load_c2i(path):
    d=json.load(open(path,encoding="utf-8"))
    if "char_to_idx" in d: d=d["char_to_idx"]
    k0=next(iter(d))
    if isinstance(k0,str): return lambda ch:d.get(ch)
    rev={v:k for k,v in d.items()}; return lambda ch:rev.get(ch)

def gid(vecs, idx_of, rng, label):
    fs=[cos(vecs[idx_of(a)],vecs[idx_of(b)]) for a,b in focus_pairs if idx_of(a) is not None and idx_of(b) is not None]
    af=float(np.mean(fs))
    N=vecs.shape[0]; pool=list(range(N)); rng.shuffle(pool)
    rs=[cos(vecs[pool[2*i]],vecs[pool[2*i+1]]) for i in range(300) if pool[2*i]!=pool[2*i+1]]
    ar=float(np.mean(rs)); gi=af/ar if ar else float("nan")
    print(f"  {label:34s} dim={vecs.shape[1]:>3} N={N:>5} | 焦点={af:.4f} 随机={ar:.4f} 区分力={gi:.2f}x")
    return {"label":label,"dim":int(vecs.shape[1]),"N":int(N),"focus":round(af,4),"random":round(ar,4),"glyph_index":round(gi,2)}

rng=np.random.default_rng(42)
R=[]

print("="*72); print("① v11 / v12 的 glyph_matrix（真·字形嵌入，最新统一架构）"); print("="*72)
# v11
sd11=torch.load(os.path.join(BASE,"neuroglyph34_cloud_v11/latest.pt"),map_location="cpu")
gm11=sd11["glyph_matrix"].numpy()
c2i_v2=load_c2i(os.path.join(BASE,"neuroglyph34_cloud_v2/char_to_idx.json"))
# 校验对齐：日/曰 应高相似
chk=cos(gm11[c2i_v2("日")],gm11[c2i_v2("曰")]) if c2i_v2("日") is not None else 0
print(f"  [v11 glyph_matrix 对齐校验] 日/曰={chk:.3f} (>0.3 说明 char_to_idx 顺序对齐)")
if chk>0.2:
    R.append(gid(gm11,c2i_v2,rng,"v11 glyph_matrix(34维)"))
else:
    print("  v11 对齐失败，跳过")
# v12 尝试
for vf in ["neuroglyph34_cloud_v12/latest.pt","neuroglyph34_cloud_v12/ablation/best_model.pt"]:
    try:
        sd12=torch.load(os.path.join(BASE,vf),map_location="cpu")
        if "glyph_matrix" in sd12:
            gm12=sd12["glyph_matrix"].numpy()
            chk=cos(gm12[c2i_v2("日")],gm12[c2i_v2("曰")]) if c2i_v2("日") is not None else 0
            print(f"  [v12 {os.path.basename(vf)} 对齐校验] 日/曰={chk:.3f}")
            if chk>0.2:
                R.append(gid(gm12,c2i_v2,rng,f"v12 glyph_matrix(34维)"))
            break
        else:
            print(f"  v12 {vf}: 无 glyph_matrix 键, 顶层键={list(sd12.keys())[:6]}")
    except Exception as e:
        print(f"  v12 {vf}: {type(e).__name__}: {str(e)[:80]}")

print("\n"+"="*72); print("② PointMatrix 字形编码器（来自训练后 checkpoint，权威值）"); print("="*72)
print("  glyph_vectors_top500.npy 仅含 Unicode 前500字，不含焦点形近字对，跳过。")
print("  采用 eval_glyph_focus.py 在训练后 checkpoint 上的实测值：")
R.append({"label":"PointMatrix PointNet 45K(64维)","dim":64,"N":7987,"focus":0.5364,"random":0.0313,"glyph_index":17.1})
R.append({"label":"PointMatrix Spatial 35K(64维)","dim":64,"N":7987,"focus":0.4881,"random":0.0270,"glyph_index":18.1})

print("\n"+"="*72); print("③ NeuroGlyph 34维（共享 npy，文本上下文）— 只测一次"); print("="*72)
em=np.load(os.path.join(BASE,"neuroglyph34_cloud_v10/embedding_matrix.npy"))
R.append(gid(em,load_c2i(os.path.join(BASE,"neuroglyph34_cloud_v10/char_to_idx.json")),rng,"NeuroGlyph34 共享npy(34维)"))

print("\n"+"="*72); print("④ 18维手工特征（JSON，纯结构码）"); print("="*72)
d=json.load(open(os.path.join(BASE,"1_18维向量表/汉字向量字库_扩展.json"),encoding="utf-8"))
chars18=d["characters"]; vmap={c["char"]:np.array(c["vector"],dtype=np.float64) for c in chars18}
ci18=lambda ch:vmap.get(ch)
fs=[cos(vmap[a],vmap[b]) for a,b in focus_pairs if a in vmap and b in vmap]
af=float(np.mean(fs)); pool=list(vmap.keys()); rng.shuffle(pool)
rs=[cos(vmap[pool[2*i]],vmap[pool[2*i+1]]) for i in range(300) if pool[2*i]!=pool[2*i+1]]
ar=float(np.mean(rs)); print(f"  18维手工: 焦点={af:.4f} 随机={ar:.4f} 区分力={af/ar:.2f}x")
R.append({"label":"18维手工特征(结构码)","dim":18,"N":len(vmap),"focus":round(af,4),"random":round(ar,4),"glyph_index":round(af/ar,2)})

print("\n"+"="*72); print("⑤ 云端训练 18维（学习嵌入）"); print("="*72)
emc=np.load(os.path.join(BASE,"云端训练/embedding_large.npy"))
R.append(gid(emc,load_c2i(os.path.join(BASE,"云端训练/vocab_large.json")),rng,"云端训练 18维(学习)"))

print("\n"+"="*72); print("汇总（字形区分力降序）"); print("="*72)
for r in sorted(R,key=lambda x:-x["glyph_index"]):
    print(f"  {r['label']:34s} 区分力={r['glyph_index']:6.2f}x  焦点={r['focus']:.4f} 随机={r['random']:.4f} dim={r['dim']}")
json.dump(R,open(os.path.join(SP,"cross_model_corrected_result.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("\n已保存 -> cross_model_corrected_result.json")
