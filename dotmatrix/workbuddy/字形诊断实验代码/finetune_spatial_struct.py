# -*- coding: utf-8 -*-
"""
finetune_spatial_struct.py — 端到端结构监督微调（本地可跑，无需句子语料）
================================================================================
证明：让梯度流回 point_encoder，用“四角号码预测 + 形近字对比损失”微调，
能否恢复被 LM 目标丢掉的字形判别力。

做法：
  - 以 50K ckpt 初始化，冻结 Transformer/LM 主体（本实验不训 LM）；
  - 仅训练 point_encoder(字形编码器) + 四角头，目标：
      L = 四角号码交叉熵  +  λ·形近字三元组对比损失
  - 四角标签来自 18维字表 vector[0:5]（逐字任务，不需要语料）；
  - 对比损失的形近字对来自 FOCUS_PAIRS / EXTRA_GROUPS。
跑完重抽字形嵌入，看重点字对余弦是否比基线(0.3653)显著下降。

输出: WorkBuddy/eval_spatial_struct_result.json
      WorkBuddy/glyph_embedding_viz_struct.html (基线 vs 结构微调)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step050000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
V18_PATH = "D:/desktop/学习结果小文件/1_18维向量表/汉字向量字库_扩展.json"
OUT_JSON = "D:/desktop/学习结果小文件/WorkBuddy/eval_spatial_struct_result.json"
OUT_HTML = "D:/desktop/学习结果小文件/WorkBuddy/glyph_embedding_viz_struct.html"
PLOTLY_JS = "D:/desktop/学习结果小文件/个人归档/plotly.min.js"
DEVICE = torch.device("cpu")

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"), ("未", "末"), ("干", "千"),
    ("干", "于"), ("千", "于"), ("大", "太"), ("王", "玉"), ("日", "曰"),
    ("人", "入"), ("土", "士"), ("天", "夫"), ("力", "刀"), ("乌", "鸟"),
    ("风", "凤"), ("王", "主"), ("田", "由"), ("由", "甲"),
]
EXTRA_GROUPS = [
    ["人", "入", "八"], ["土", "士"], ["天", "夫"], ["日", "目", "白"],
    ["田", "由", "甲", "申"], ["王", "玉", "主"], ["力", "刀"], ["乌", "鸟"],
    ["风", "凤"], ["干", "千", "于"],
]
LAMBDA_CON = 0.3
EPOCHS = 25
LR = 1e-3


def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    V = len(char_to_idx)

    model = PointMatrixSpatialModel(vocab_size=V)
    reader = HZK32Reader(HZK32_PATH)
    gm = np.zeros((V, 1024), np.float32)
    for idx in range(V):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                gm[idx] = reader.get_glyph_vector(ch).astype(np.float32)
            except (ValueError, IOError):
                pass
    model.load_glyph_matrix(gm)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(DEVICE)

    # 冻结 Transformer / LM / semantic（本实验只训字形编码器）
    for p in model.transformer.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False
    for p in model.input_proj.parameters():
        p.requires_grad = False
    for p in model.semantic_embed.parameters():
        p.requires_grad = False
    # 四角头
    head_fc = nn.Sequential(
        nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 5 * 26)
    ).to(DEVICE)
    trainable = list(model.point_encoder.parameters()) + list(head_fc.parameters())
    opt = torch.optim.Adam(trainable, lr=LR)

    # 四角标签
    v18 = json.load(open(V18_PATH, encoding="utf-8"))["characters"]
    fc_map = {}
    for it in v18:
        code = it["vector"][0:5]
        if all(c != 25 for c in code):
            fc_map[it["char"]] = code
    rows, codes = [], []
    for ch, code in fc_map.items():
        if ch in char_to_idx and gm[char_to_idx[ch]].sum() > 0:
            rows.append(char_to_idx[ch])
            codes.append(code)
    rows = np.array(rows)
    codes = np.array(codes, np.int64)
    print(f"[数据] 四角标签有效字 {len(rows)}")

    # 形近字对索引（对比损失）
    conf_idx = {}
    for a, b in FOCUS_PAIRS:
        conf_idx.setdefault(char_to_idx[a], set()).add(char_to_idx[b])
        conf_idx.setdefault(char_to_idx[b], set()).add(char_to_idx[a])
    for grp in EXTRA_GROUPS:
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i != j:
                    conf_idx.setdefault(char_to_idx[grp[i]], set()).add(char_to_idx[grp[j]])

    Xg = torch.from_numpy(gm[rows]).to(DEVICE).float()
    Yc = torch.from_numpy(codes).to(DEVICE).long()
    rng = np.random.default_rng(1)
    all_pairs = [(char_to_idx[a], char_to_idx[b]) for a, b in FOCUS_PAIRS
                 if a in char_to_idx and b in char_to_idx]
    for grp in EXTRA_GROUPS:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                all_pairs.append((char_to_idx[grp[i]], char_to_idx[grp[j]]))

    # 基线（微调前）嵌入
    model.point_encoder.eval()
    with torch.no_grad():
        base_emb = model.point_encoder(
            torch.from_numpy(gm).to(DEVICE).float()
        ).cpu().numpy().astype(np.float64)
    model.point_encoder.train()

    print("=" * 56)
    print("端到端结构监督微调（只训 point_encoder + 四角头）")
    print("=" * 56)
    for ep in range(EPOCHS):
        opt.zero_grad()
        feat = model.point_encoder(Xg)   # (N,64)
        logits = head_fc(feat).reshape(len(rows), 5, 26)
        loss_fc = sum(F.cross_entropy(logits[:, d], Yc[:, d]) for d in range(5)) / 5

        # 对比损失：采样三元组
        loss_con = torch.tensor(0.0, device=DEVICE)
        ntri = min(256, len(all_pairs))
        sel = rng.choice(len(all_pairs), ntri, replace=False)
        anc, pos, neg = [], [], []
        for pi in sel:
            a, b = all_pairs[pi]
            cand = [c for c in rows if c not in conf_idx.get(a, set()) and c != a]
            if not cand:
                continue
            p = rng.choice(cand)
            anc.append(a); neg.append(b); pos.append(int(p))
        if anc:
            ai = torch.tensor(anc); pi2 = torch.tensor(pos); ni = torch.tensor(neg)
            # 用全量 emb 取向量（微调中需梯度）
            allfeat = model.point_encoder(
                torch.from_numpy(gm).to(DEVICE).float())
            anc_n = F.normalize(allfeat[ai], dim=-1)
            pos_n = F.normalize(allfeat[pi2], dim=-1)
            neg_n = F.normalize(allfeat[ni], dim=-1)
            loss_con = F.triplet_margin_loss(anc_n, pos_n, neg_n, margin=0.3, p=2)
        loss = loss_fc + LAMBDA_CON * loss_con
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:2d}: loss={loss.item():.4f} "
                  f"(fc={loss_fc.item():.4f}, con={loss_con.item():.4f})")

    # 微调后嵌入
    model.point_encoder.eval()
    with torch.no_grad():
        new_emb = model.point_encoder(
            torch.from_numpy(gm).to(DEVICE).float()
        ).cpu().numpy().astype(np.float64)

    # 评估重点字对
    base_d, new_d = {}, {}
    for a, b in FOCUS_PAIRS:
        if a in char_to_idx and b in char_to_idx:
            ia, ib = char_to_idx[a], char_to_idx[b]
            base_d[f"{a}/{b}"] = cosine_sim(base_emb[ia], base_emb[ib])
            new_d[f"{a}/{b}"] = cosine_sim(new_emb[ia], new_emb[ib])
    base_avg = float(np.mean(list(base_d.values())))
    new_avg = float(np.mean(list(new_d.values())))
    print(f"\n[结果] 重点字对平均余弦: 基线 {base_avg:.4f} → 结构微调 {new_avg:.4f} "
          f"(Δ {(new_avg-base_avg):+.4f})")

    result = {
        "baseline_avg_cos": base_avg,
        "struct_finetune_avg_cos": new_avg,
        "lambda_contrastive": LAMBDA_CON,
        "epochs": EPOCHS,
        "fourcorner_label_n": int(len(rows)),
        "focus_pairs": {k: {"baseline": base_d[k], "struct": new_d[k]} for k in base_d},
    }
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[Save] {OUT_JSON}")

    build_viz(gm, base_emb, new_emb, base_d, new_d, char_to_idx, idx_to_char)
    print("\n若结构微调后平均余弦明显低于基线 → 端到端结构监督有效，"
          "下一步应在云端对完整模型做同类重训（编码器+LM主体一起，带四角/对比辅助损失）。")


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def build_viz(gm, base_emb, new_emb, base_d, new_d, char_to_idx, idx_to_char):
    valid = gm.sum(1) > 0
    idxs = np.where(valid)[0]
    chars = [idx_to_char[i] for i in idxs]
    pb = pca2(base_emb[idxs]); pn = pca2(new_emb[idxs])
    HIGHLIGHT = sorted(set(c for p in FOCUS_PAIRS for c in p))
    char_to_row = {c: k for k, c in enumerate(chars)}
    hl_b = {c: [round(float(pb[char_to_row[c], 0]), 3), round(float(pb[char_to_row[c], 1]), 3)]
            for c in HIGHLIGHT if c in char_to_row}
    hl_n = {c: [round(float(pn[char_to_row[c], 0]), 3), round(float(pn[char_to_row[c], 1]), 3)]
            for c in HIGHLIGHT if c in char_to_row}
    data = {
        "chars": chars,
        "base": [[round(float(x), 3), round(float(y), 3)] for x, y in pb],
        "head": [[round(float(x), 3), round(float(y), 3)] for x, y in pn],
        "highlight_base": hl_b, "highlight_head": hl_n,
        "pairs": [{"a": k.split("/")[0], "b": k.split("/")[1],
                   "base": round(base_d[k], 4), "head": round(new_d[k], 4)} for k in base_d],
    }
    with open(PLOTLY_JS, "r", encoding="utf-8") as f:
        plotly_src = f.read()
    rows = ""
    for k in base_d:
        bv, nv = base_d[k], new_d[k]
        color = "#27ae60" if nv < bv else "#c0392b"
        rows += (f"<tr><td>{k}</td><td>{bv:.4f}</td><td>{nv:.4f}</td>"
                 f"<td style='color:{color};font-weight:bold'>{(nv-bv):+.4f}</td></tr>")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>基线 vs 结构监督微调</title><script>{plotly_src}</script>
<style>body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:0;background:#fafafa;color:#222}}
header{{padding:16px 24px;background:#fff;border-bottom:1px solid #eee}}
h1{{margin:0;font-size:20px}}.sub{{color:#666;font-size:13px;margin-top:4px}}
.wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:16px 24px}}
.card{{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px;flex:1;min-width:420px}}
.card h2{{font-size:15px;margin:8px 4px}}
table{{border-collapse:collapse;width:auto;font-size:13px;margin:8px 24px}}
th,td{{border:1px solid #eee;padding:4px 8px;text-align:center}}th{{background:#f5f5f5}}
.note{{padding:8px 24px;color:#555;font-size:13px;line-height:1.6}}</style></head>
<body><header><h1>基线（50K 冻结） vs 端到端结构监督微调</h1>
<div class="sub">灰点=全部汉字；橙点=硬骨头；连线=重点字对。绿表行=微调后余弦下降（字形被拉开）。</div></header>
<div class="wrap">
<div class="card"><h2>左：基线 50K 嵌入</h2><div id="pB" style="height:560px"></div></div>
<div class="card"><h2>右：结构监督微调后</h2><div id="pA" style="height:560px"></div></div>
</div>
<div class="note">这是<b>端到端</b>微调：梯度回流到 point_encoder（字形编码器），目标 = 四角号码预测 + 形近字对比损失。
若右图硬骨头明显散开、表中余弦大幅下降，则证明<b>结构监督能恢复字形判别力</b>，应在云端对完整模型做同类重训。</div>
<h2 style="margin:16px 24px 0">重点字对余弦：基线 → 结构微调</h2>
<table><tr><th>字对</th><th>基线</th><th>结构微调</th><th>Δ</th></tr>{rows}</table>
<script>
const D={json.dumps(data,ensure_ascii=False)};
function mk(div,key,hlKey){{
  const xs=D[key].map(p=>p[0]),ys=D[key].map(p=>p[1]),txt=D.chars;
  const tr={{x:xs,y:ys,mode:'markers',type:'scatter',marker:{{size:4,color:'#bbb',opacity:.5}},text:txt,hoverinfo:'text'}};
  const hx=[],hy=[],ht=[];
  Object.keys(D[hlKey]).forEach(c=>{{const co=D[hlKey][c];hx.push(co[0]);hy.push(co[1]);ht.push(c);}});
  const ht2={{x:hx,y:hy,mode:'markers+text',type:'scatter',marker:{{size:11,color:'#e67e22'}},
    text:ht,textposition:'top center',textfont:{{size:13,color:'#c0392b'}},hoverinfo:'text'}};
  const lines=[];
  D.pairs.forEach(p=>{{const ca=D[hlKey][p.a],cb=D[hlKey][p.b];
    if(ca&&cb){{const col=p.head<p.base?'#27ae60':'#c0392b';
      lines.push({{x:[ca[0],cb[0]],y:[ca[1],cb[1]],mode:'lines',line:{{color:col,width:2,opacity:.7}},hoverinfo:'skip',showlegend:false}});}}}});
  Plotly.newPlot(div,[tr,ht2].concat(lines),
    {{margin:{{t:10,l:40,r:10,b:40}},showlegend:false,xaxis:{{title:'PC1'}},yaxis:{{title:'PC2'}},hovermode:'closest'}},
    {{displayModeBar:true,responsive:true}});
}}
mk('pB','base','highlight_base'); mk('pA','head','highlight_head');
</script></body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Save] {OUT_HTML}")


if __name__ == "__main__":
    main()
