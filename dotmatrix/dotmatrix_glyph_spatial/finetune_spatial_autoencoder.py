# -*- coding: utf-8 -*-
"""
finetune_spatial_autoencoder.py — 字形自编码器微调（本地可跑，无需语料）
==========================================================================
针对已证根因（50K 嵌入丢掉了像素级判别信息）的治本实验：
  让 point_encoder(64维) + decoder 重建 HZK32 点阵(32x32)，
  迫使编码器保留每个像素差异（含王/玉 的那个点）。
目标: L = BCE(重建点阵, 原点阵)。
若微调后重点字对余弦显著低于基线(0.3653) → 像素重建是正确监督信号，
下一步应在云端对完整模型做"LM + 像素重建"联合训练。

输出: WorkBuddy/eval_spatial_ae_result.json
      WorkBuddy/glyph_embedding_viz_ae.html
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
OUT_JSON = "D:/desktop/学习结果小文件/WorkBuddy/eval_spatial_ae_result.json"
OUT_HTML = "D:/desktop/学习结果小文件/WorkBuddy/glyph_embedding_viz_ae.html"
PLOTLY_JS = "D:/desktop/学习结果小文件/个人归档/plotly.min.js"
DEVICE = torch.device("cpu")

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"), ("未", "末"), ("干", "千"),
    ("干", "于"), ("千", "于"), ("大", "太"), ("王", "玉"), ("日", "曰"),
    ("人", "入"), ("土", "士"), ("天", "夫"), ("力", "刀"), ("乌", "鸟"),
    ("风", "凤"), ("王", "主"), ("田", "由"), ("由", "甲"),
]
EPOCHS = 40
LR = 1e-3


def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class GlyphDecoder(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.fc = nn.Linear(in_dim, 4 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.GELU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.fc(x).reshape(-1, 4, 8, 8)
        return self.deconv(h)  # (N,1,32,32)


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

    for p in model.transformer.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False
    for p in model.input_proj.parameters():
        p.requires_grad = False
    for p in model.semantic_embed.parameters():
        p.requires_grad = False

    decoder = GlyphDecoder().to(DEVICE)
    opt = torch.optim.Adam(
        list(model.point_encoder.parameters()) + list(decoder.parameters()), lr=LR)

    valid = gm.sum(1) > 0
    idxs = np.where(valid)[0]
    Xglyph = torch.from_numpy(gm[idxs]).to(DEVICE).float()        # (N,1024)
    Ybitmap = torch.from_numpy(gm[idxs]).to(DEVICE).float().reshape(len(idxs), 1, 32, 32)

    # 基线嵌入
    model.point_encoder.eval()
    with torch.no_grad():
        base_emb = model.point_encoder(torch.from_numpy(gm).to(DEVICE).float()
                                       ).cpu().numpy().astype(np.float64)
    model.point_encoder.train()

    print("=" * 56)
    print("字形自编码器微调（encoder+decoder 重建点阵）")
    print("=" * 56)
    for ep in range(EPOCHS):
        opt.zero_grad()
        emb = model.point_encoder(Xglyph)               # (N,64)
        recon = decoder(emb)                            # (N,1,32,32)
        loss = F.binary_cross_entropy(recon, Ybitmap)
        loss.backward()
        opt.step()
        if ep % 8 == 0 or ep == EPOCHS - 1:
            # 重建精度（像素级）
            with torch.no_grad():
                acc = ((recon > 0.5).float() == (Ybitmap > 0.5).float()).float().mean().item()
            print(f"  epoch {ep:2d}: bce={loss.item():.4f}  像素重建准确率={acc:.3f}")

    model.point_encoder.eval()
    with torch.no_grad():
        new_emb = model.point_encoder(torch.from_numpy(gm).to(DEVICE).float()
                                      ).cpu().numpy().astype(np.float64)

    base_d, new_d = {}, {}
    for a, b in FOCUS_PAIRS:
        if a in char_to_idx and b in char_to_idx:
            ia, ib = char_to_idx[a], char_to_idx[b]
            base_d[f"{a}/{b}"] = cosine_sim(base_emb[ia], base_emb[ib])
            new_d[f"{a}/{b}"] = cosine_sim(new_emb[ia], new_emb[ib])
    base_avg = float(np.mean(list(base_d.values())))
    new_avg = float(np.mean(list(new_d.values())))
    print(f"\n[结果] 重点字对平均余弦: 基线 {base_avg:.4f} → 自编码器 {new_avg:.4f} "
          f"(Δ {(new_avg-base_avg):+.4f})")

    result = {
        "baseline_avg_cos": base_avg,
        "autoencoder_avg_cos": new_avg,
        "epochs": EPOCHS,
        "focus_pairs": {k: {"baseline": base_d[k], "ae": new_d[k]} for k in base_d},
    }
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[Save] {OUT_JSON}")

    build_viz(gm, base_emb, new_emb, base_d, new_d, char_to_idx, idx_to_char)
    print("\n若自编码器后平均余弦明显低于基线 → 像素重建是正确监督信号，"
          "下一步云端做 LM + 像素重建 联合训练。")


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
<title>基线 vs 字形自编码器</title><script>{plotly_src}</script>
<style>body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:0;background:#fafafa;color:#222}}
header{{padding:16px 24px;background:#fff;border-bottom:1px solid #eee}}
h1{{margin:0;font-size:20px}}.sub{{color:#666;font-size:13px;margin-top:4px}}
.wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:16px 24px}}
.card{{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px;flex:1;min-width:420px}}
.card h2{{font-size:15px;margin:8px 4px}}
table{{border-collapse:collapse;width:auto;font-size:13px;margin:8px 24px}}
th,td{{border:1px solid #eee;padding:4px 8px;text-align:center}}th{{background:#f5f5f5}}
.note{{padding:8px 24px;color:#555;font-size:13px;line-height:1.6}}</style></head>
<body><header><h1>基线（50K 冻结） vs 字形自编码器微调</h1>
<div class="sub">灰点=全部汉字；橙点=硬骨头；连线=重点字对。绿表行=微调后余弦下降（字形被拉开）。</div></header>
<div class="wrap">
<div class="card"><h2>左：基线 50K 嵌入</h2><div id="pB" style="height:560px"></div></div>
<div class="card"><h2>右：自编码器微调后</h2><div id="pA" style="height:560px"></div></div>
</div>
<div class="note">自编码器目标 = 从 64维嵌入重建 32×32 点阵。迫使编码器保留每个像素差异（含王/玉的那个点）。
若右图硬骨头散开、表中余弦大幅下降 → <b>像素重建是正确监督信号</b>。</div>
<h2 style="margin:16px 24px 0">重点字对余弦：基线 → 自编码器</h2>
<table><tr><th>字对</th><th>基线</th><th>自编码器</th><th>Δ</th></tr>{rows}</table>
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
