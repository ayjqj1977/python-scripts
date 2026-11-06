# -*- coding: utf-8 -*-
"""
viz_glyph_embedding.py — 可视化 50K Spatial 字形嵌入空间
========================================================
两个对照图（同一批字，两种表示，都 PCA→2D）：
  A. 学到的字形嵌入 (64维, point_encoder 输出)
  B. 原始 HZK32 点阵像素 (1024维, 未训练)
重点：看硬骨头字对（王/玉、风/凤、日/曰…）在 A 里是否比在 B 里更糊。
     若 B 里分得开、A 里糊成一团 → 证明编码器丢掉了字形判别信息。
输出：D:/desktop/学习结果小文件/WorkBuddy/glyph_embedding_viz_50k.html
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step050000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
OUT_HTML = "D:/desktop/学习结果小文件/WorkBuddy/glyph_embedding_viz_50k.html"
PLOTLY_JS = "D:/desktop/学习结果小文件/个人归档/plotly.min.js"
DEVICE = torch.device("cpu")

# 硬骨头字对（来自 eval 脚本）
FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"), ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"), ("王", "玉"), ("日", "曰"),
    ("人", "入"), ("土", "士"), ("天", "夫"),
    ("力", "刀"), ("乌", "鸟"), ("风", "凤"),
    ("王", "主"), ("田", "由"), ("由", "甲"),
]
HIGHLIGHT = sorted(set(c for p in FOCUS_PAIRS for c in p))


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main():
    with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    model = PointMatrixSpatialModel(vocab_size=vocab_size)
    reader = HZK32Reader(HZK32_PATH)
    glyph_matrix = np.zeros((vocab_size, 1024), dtype=np.float32)
    loaded = 0
    for idx in range(vocab_size):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                glyph_matrix[idx] = reader.get_glyph_vector(ch).astype(np.float32)
                loaded += 1
            except (ValueError, IOError):
                pass
    model.load_glyph_matrix(glyph_matrix)
    print(f"[Load] 点阵矩阵 {loaded}/{vocab_size}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    print(f"[Load] Checkpoint: {os.path.basename(CKPT_PATH)}")

    # --- 一次性取出全部字的 64维嵌入 ---
    with torch.no_grad():
        g = torch.from_numpy(glyph_matrix).to(DEVICE)
        g = g.reshape(vocab_size, 1, 32, 32)
        conv = model.point_encoder.conv(g)            # (V,4,32,32)
        conv = conv.reshape(vocab_size, -1)
        learned = model.point_encoder.proj(conv).cpu().numpy().astype(np.float64)  # (V,64)

    # 只保留有点阵的字（去掉 <UNK>/ 未加载），减少噪声点
    valid = (glyph_matrix.sum(1) > 0)
    idxs = np.where(valid)[0]
    chars = [idx_to_char[i] for i in idxs]
    learned_v = learned[idxs]
    raw_v = glyph_matrix[idxs].astype(np.float64)

    proj_learned = pca2(learned_v)
    proj_raw = pca2(raw_v)

    # 计算硬骨头对在两种投影下的欧氏距离（越小越糊）
    def dist2(proj, a, b):
        if a in char_to_idx and b in char_to_idx:
            ia, ib = char_to_idx[a], char_to_idx[b]
            if valid[ia] and valid[ib]:
                xa, xb = proj[list(idxs).index(ia)], proj[list(idxs).index(ib)]
                return float(np.linalg.norm(xa - xb))
        return None

    pair_stats = []
    for a, b in FOCUS_PAIRS:
        dl = dist2(proj_learned, a, b)
        dr = dist2(proj_raw, a, b)
        pair_stats.append({"a": a, "b": b,
                           "learned_dist": round(dl, 4) if dl else None,
                           "raw_dist": round(dr, 4) if dr else None})

    # 把 highlight 字的坐标抽出来用于标注
    def coords(proj, ch):
        i = char_to_idx.get(ch)
        if i is None or not valid[i]:
            return None
        pos = list(idxs).index(i)
        return [round(float(proj[pos, 0]), 3), round(float(proj[pos, 1]), 3)]

    highlight_learned = {c: coords(proj_learned, c) for c in HIGHLIGHT}
    highlight_raw = {c: coords(proj_raw, c) for c in HIGHLIGHT}

    data = {
        "chars": chars,
        "learned": [[round(float(x), 3), round(float(y), 3)] for x, y in proj_learned],
        "raw": [[round(float(x), 3), round(float(y), 3)] for x, y in proj_raw],
        "highlight": HIGHLIGHT,
        "highlight_learned": highlight_learned,
        "highlight_raw": highlight_raw,
        "pairs": pair_stats,
    }

    # --- 组装 HTML（内联 plotly，离线可用）---
    with open(PLOTLY_JS, "r", encoding="utf-8") as f:
        plotly_src = f.read()
    html = build_html(data, plotly_src)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Save] HTML: {OUT_HTML}")

    # 控制台也打印硬骨头距离对比
    print("\n硬骨头字对：学到的嵌入距离 vs 原始像素距离（越小越糊）")
    print(f"{'字对':<8}{'learned':>10}{'raw':>10}{'比值(L/R)':>12}")
    print("-" * 42)
    for p in pair_stats:
        if p["learned_dist"] and p["raw_dist"]:
            r = p["learned_dist"] / p["raw_dist"]
            print(f"{p['a']}/{p['b']:<6}{p['learned_dist']:>10.4f}{p['raw_dist']:>10.4f}{r:>12.2f}x")


def build_html(data, plotly_src):
    payload = json.dumps(data, ensure_ascii=False)
    # 比值表
    rows = ""
    for p in data["pairs"]:
        if p["learned_dist"] and p["raw_dist"]:
            r = p["learned_dist"] / p["raw_dist"]
            color = "#c0392b" if r < 1 else "#27ae60"
            rows += (f"<tr><td>{p['a']}/{p['b']}</td>"
                     f"<td>{p['learned_dist']:.4f}</td>"
                     f"<td>{p['raw_dist']:.4f}</td>"
                     f"<td style='color:{color};font-weight:bold'>{r:.2f}x</td></tr>")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Spatial 50K 字形嵌入空间可视化</title>
<script>{plotly_src}</script>
<style>
 body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:0;background:#fafafa;color:#222}}
 header{{padding:16px 24px;background:#fff;border-bottom:1px solid #eee}}
 h1{{margin:0;font-size:20px}}
 .sub{{color:#666;font-size:13px;margin-top:4px}}
 .wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:16px 24px}}
 .card{{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px;flex:1;min-width:420px}}
 .card h2{{font-size:15px;margin:8px 4px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
 th,td{{border:1px solid #eee;padding:4px 8px;text-align:center}}
 th{{background:#f5f5f5}}
 .note{{padding:8px 24px;color:#555;font-size:13px;line-height:1.6}}
 .legend{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
</style></head>
<body>
<header>
  <h1>Spatial 50K 字形嵌入空间 · 病灶诊断</h1>
  <div class="sub">左：模型学到的 64维字形嵌入(PCA→2D)　|　右：原始 HZK32 点阵像素(PCA→2D)　|　红=形近字对连线上距离更短(更糊)</div>
</header>
<div class="wrap">
  <div class="card"><h2>A. 学到的字形嵌入（point_encoder 输出）</h2><div id="plotA" style="height:560px"></div></div>
  <div class="card"><h2>B. 原始点阵像素（未训练，对照）</h2><div id="plotB" style="height:560px"></div></div>
</div>
<div class="note">
  <b>怎么看：</b> 灰点是全部有效汉字。彩色大点是硬骨头字（王/玉、风/凤、日/曰…），带连线的字对是关键对比。
  <span class="legend" style="background:#c0392b"></span>红线=该对在「学到的嵌入」里比「原始像素」里更近（信息被丢掉，糊了）
  <span class="legend" style="background:#27ae60"></span>绿线=反而更开（学到了判别）。
  <br>若大量形近字对在 A 中距离明显短于 B，说明<b>编码器在训练后把像素里本就存在的字形判别信息压没了</b>——这正是「加训练步数没用」的根因。
</div>
<h2 style="margin:16px 24px 0">硬骨头字对距离对比（2D 欧氏距离，越小越糊）</h2>
<table style="margin:8px 24px;width:auto">
  <tr><th>字对</th><th>学到的嵌入</th><th>原始像素</th><th>比值 L/R</th></tr>
  {rows}
</table>
<script>
const D = {payload};
function mkPlot(divId, key, hlKey){{
  const xs=D[key].map(p=>p[0]), ys=D[key].map(p=>p[1]);
  const txt=D.chars;
  const trace={{x:xs,y:ys,mode:'markers',type:'scatter',
    marker:{{size:4,color:'#bbb',opacity:0.5}},
    text:txt,hoverinfo:'text',name:'汉字'}};
  const hl=D.highlight;
  const hx=[],hy=[],ht=[];
  hl.forEach(c=>{{const co=D[hlKey][c];if(co){{hx.push(co[0]);hy.push(co[1]);ht.push(c);}}}});
  const htrace={{x:hx,y:hy,mode:'markers+text',type:'scatter',
    marker:{{size:11,color:'#e67e22',line:{{color:'#fff',width:1}}}},
    text:ht,textposition:'top center',textfont:{{size:13,color:'#c0392b'}},
    hoverinfo:'text',name:'硬骨头'}};
  // 连线
  const lines=[];
  D.pairs.forEach(p=>{{
    const ca=D[hlKey][p.a], cb=D[hlKey][p.b];
    if(ca&&cb){{
      const ratio = (p.learned_dist&&p.raw_dist)? p.learned_dist/p.raw_dist : 1;
      const col = ratio<1? '#c0392b':'#27ae60';
      lines.push({{x:[ca[0],cb[0]],y:[ca[1],cb[1]],mode:'lines',
        line:{{color:col,width:2,opacity:0.7}},hoverinfo:'skip',showlegend:false}});
    }}
  }});
  const data=[trace,htrace].concat(lines);
  const layout={{margin:{{t:10,l:40,r:10,b:40}},showlegend:false,
    xaxis:{{title:'PC1',zeroline:false}},yaxis:{{title:'PC2',zeroline:false}},
    hovermode:'closest'}};
  Plotly.newPlot(divId,data,layout,{{displayModeBar:true,responsive:true}});
}}
mkPlot('plotA','learned','highlight_learned');
mkPlot('plotB','raw','highlight_raw');
</script>
</body></html>"""


if __name__ == "__main__":
    main()
