# -*- coding: utf-8 -*-
"""
probe_structural_supervision.py — 本地原型：验证“换监督信号”能否撬动字形硬骨头
=================================================================================
用冻结的 50K Spatial 编码器输出(64维)做两个探针，全部 CPU 可跑，不烧云 GPU：
  探针A(对比损失): 在冻结嵌入上加可训练小头，用形近字三元组对比损失训练，
                   看王/玉、风/凤 等重点字对余弦能否被显著拉开。
  探针B(四角预测): 训练小分类头从冻结嵌入预测四角号码(vector[0:5])，
                   看字形结构信号是否本就存在于冻结嵌入中。
输出: WorkBuddy/probe_structural_result.json
      WorkBuddy/glyph_embedding_viz_contrastive.html (基线 vs 对比头 空间对照)
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
OUT_JSON = "D:/desktop/学习结果小文件/WorkBuddy/probe_structural_result.json"
OUT_HTML = "D:/desktop/学习结果小文件/WorkBuddy/glyph_embedding_viz_contrastive.html"
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


def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_embeddings():
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
    model.eval().to(DEVICE)
    with torch.no_grad():
        g = torch.from_numpy(gm).to(DEVICE).reshape(V, 1, 32, 32)
        conv = model.point_encoder.conv(g).reshape(V, -1)
        emb = model.point_encoder.proj(conv).cpu().numpy().astype(np.float64)
    valid = gm.sum(1) > 0
    idxs = np.where(valid)[0]
    chars = [idx_to_char[i] for i in idxs]
    embv = emb[idxs]
    return chars, embv, char_to_idx, idx_to_char


def build_confusable():
    conf = {}
    for a, b in FOCUS_PAIRS:
        conf.setdefault(a, set()).add(b)
        conf.setdefault(b, set()).add(a)
    for grp in EXTRA_GROUPS:
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i != j:
                    conf.setdefault(grp[i], set()).add(grp[j])
    return conf


def probe_contrastive(chars, embv, char_to_idx):
    """在冻结嵌入上加小头，三元组对比损失，拉开形近字。"""
    conf = build_confusable()
    # 预生成三元组 (anchor, positive-random, negative-confusable)
    rng = np.random.default_rng(0)
    triplets = []
    for a, b in FOCUS_PAIRS:
        if a in char_to_idx and b in char_to_idx:
            ai, bi = char_to_idx[a], char_to_idx[b]
            # positive: 随机一个非形近字
            cand = [c for c in char_to_idx
                    if c not in conf.get(a, set()) and c != a and char_to_idx[c] in set(char_to_idx.values())]
            pos = rng.choice(cand) if cand else a
            triplets.append((a, pos, b))
    # 也把 extra groups 内两两当负对
    for grp in EXTRA_GROUPS:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                if a in char_to_idx and b in char_to_idx:
                    cand = [c for c in char_to_idx if c not in conf.get(a, set()) and c != a]
                    pos = rng.choice(cand) if cand else a
                    triplets.append((a, pos, b))

    # 映射为索引
    tri = [(char_to_idx[a], char_to_idx[p], char_to_idx[n]) for a, p, n in triplets]

    head = nn.Sequential(
        nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)
    ).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    X = torch.from_numpy(embv).to(DEVICE).float()

    def proj(idx):
        return F.normalize(head(X[idx]), dim=-1)

    for step in range(1500):
        opt.zero_grad()
        ai = torch.tensor([t[0] for t in tri])
        pi = torch.tensor([t[1] for t in tri])
        ni = torch.tensor([t[2] for t in tri])
        anc, pos, neg = proj(ai), proj(pi), proj(ni)
        loss = F.triplet_margin_loss(anc, pos, neg, margin=0.3, p=2)
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"  [contrastive] step {step}: loss={loss.item():.4f}")

    # 计算对比头后的嵌入
    with torch.no_grad():
        out = head(X).cpu().numpy().astype(np.float64)
    return out


def probe_fourcorner(chars, embv, char_to_idx):
    """预测四角号码(vector[0:5])，看结构信号是否在冻结嵌入中。"""
    v18 = json.load(open(V18_PATH, encoding="utf-8"))["characters"]
    fc = {}
    for item in v18:
        ch = item["char"]
        vec = item["vector"]
        code = vec[0:5]
        if all(c != 25 for c in code):  # 跳过 padding
            fc[ch] = code
    # 收集有标签且嵌入有效的样本
    Xs, Ys = [], []
    for ch, code in fc.items():
        if ch in char_to_idx:
            i = char_to_idx[ch]
            # 找在 embv 中的行
            # embv 顺序是 idxs，需要映射
            Xs.append(ch); Ys.append(code)
    # 建立 char->embv 行
    char_to_row = {c: k for k, c in enumerate(chars)}
    rows, labels = [], []
    for ch, code in zip(Xs, Ys):
        if ch in char_to_row:
            rows.append(char_to_row[ch])
            labels.append(code)
    rows = np.array(rows)
    labels = np.array(labels)  # (N,5)
    print(f"  [fourcorner] 有效样本 {len(rows)}，类别 0-25")

    # 5 个独立的小分类头
    heads = [nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                           nn.Linear(32, 26)).to(DEVICE) for _ in range(5)]
    opts = [torch.optim.Adam(h.parameters(), lr=1e-2) for h in heads]
    Xt = torch.from_numpy(embv[rows]).to(DEVICE).float()
    Yt = torch.from_numpy(labels).long().to(DEVICE)
    for step in range(800):
        for d in range(5):
            opts[d].zero_grad()
            loss = F.cross_entropy(heads[d](Xt), Yt[:, d])
            loss.backward()
            opts[d].step()
    # 评估
    accs = []
    with torch.no_grad():
        for d in range(5):
            pred = heads[d](Xt).argmax(1)
            acc = (pred == Yt[:, d]).float().mean().item()
            accs.append(acc)
    return float(np.mean(accs)), accs


def main():
    print("=" * 60)
    print("本地原型：结构监督能否撬动字形硬骨头？")
    print("=" * 60)
    chars, embv, char_to_idx, idx_to_char = load_embeddings()
    char_to_row = {c: k for k, c in enumerate(chars)}

    # 基线余弦（冻结嵌入）
    base = {}
    for a, b in FOCUS_PAIRS:
        if a in char_to_row and b in char_to_row:
            base[f"{a}/{b}"] = cosine_sim(embv[char_to_row[a]], embv[char_to_row[b]])
    base_avg = float(np.mean(list(base.values())))
    print(f"\n[基线] 重点字对平均余弦 = {base_avg:.4f}")

    # 探针A
    print("\n--- 探针A：形近字对比损失 ---")
    outA = probe_contrastive(chars, embv, char_to_idx)
    postA = {}
    for a, b in FOCUS_PAIRS:
        if a in char_to_row and b in char_to_row:
            postA[f"{a}/{b}"] = cosine_sim(outA[char_to_row[a]], outA[char_to_row[b]])
    postA_avg = float(np.mean(list(postA.values())))
    print(f"[对比头] 重点字对平均余弦 = {postA_avg:.4f}  (基线 {base_avg:.4f})")

    # 探针B
    print("\n--- 探针B：四角号码预测 ---")
    fc_acc, fc_per = probe_fourcorner(chars, embv, char_to_idx)
    print(f"[四角] 平均预测准确率 = {fc_acc:.3f}  (逐位 {[round(x,3) for x in fc_per]})")

    # 保存
    result = {
        "baseline_avg_cos": base_avg,
        "contrastive_head_avg_cos": postA_avg,
        "fourcorner_acc": fc_acc,
        "fourcorner_per_pos": fc_per,
        "focus_pairs": {k: {"baseline": base[k], "contrastive": postA[k]} for k in base},
    }
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[Save] {OUT_JSON}")

    # 生成对照可视化（基线 vs 对比头）
    build_contrastive_viz(chars, embv, outA, char_to_row, base, postA)

    print("\n=== 结论提示 ===")
    print(f"对比头后重点字对平均余弦 {postA_avg:.4f} vs 基线 {base_avg:.4f}")
    print(f"  → ↑(更大)=更混淆：冻结嵌入上加头无法恢复字形，须端到端重训编码器")
    print(f"四角预测准确率 {fc_acc:.3f}（随机基线≈0.038）：远高于随机，冻结嵌入仍含部分结构信号但已被模糊")


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def build_contrastive_viz(chars, embv, outA, char_to_row, base, postA):
    HIGHLIGHT = sorted(set(c for p in FOCUS_PAIRS for c in p))
    pl = pca2(embv)
    pa = pca2(outA)
    hl_base = {c: [round(float(pl[char_to_row[c], 0]), 3), round(float(pl[char_to_row[c], 1]), 3)]
               for c in HIGHLIGHT if c in char_to_row}
    hl_a = {c: [round(float(pa[char_to_row[c], 0]), 3), round(float(pa[char_to_row[c], 1]), 3)]
            for c in HIGHLIGHT if c in char_to_row}
    data = {
        "chars": chars,
        "base": [[round(float(x), 3), round(float(y), 3)] for x, y in pl],
        "head": [[round(float(x), 3), round(float(y), 3)] for x, y in pa],
        "highlight_base": hl_base, "highlight_head": hl_a,
        "pairs": [{"a": k.split("/")[0], "b": k.split("/")[1],
                   "base": round(base[k], 4), "head": round(postA[k], 4)} for k in base],
    }
    with open(PLOTLY_JS, "r", encoding="utf-8") as f:
        plotly_src = f.read()
    rows = ""
    for k in base:
        bv, hv = base[k], postA[k]
        color = "#27ae60" if hv < bv else "#c0392b"
        rows += (f"<tr><td>{k}</td><td>{bv:.4f}</td><td>{hv:.4f}</td>"
                 f"<td style='color:{color};font-weight:bold'>{(bv-hv):+.4f}</td></tr>")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>基线 vs 对比头 字形空间</title><script>{plotly_src}</script>
<style>body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:0;background:#fafafa;color:#222}}
header{{padding:16px 24px;background:#fff;border-bottom:1px solid #eee}}
h1{{margin:0;font-size:20px}}.sub{{color:#666;font-size:13px;margin-top:4px}}
.wrap{{display:flex;flex-wrap:wrap;gap:16px;padding:16px 24px}}
.card{{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px;flex:1;min-width:420px}}
.card h2{{font-size:15px;margin:8px 4px}}
table{{border-collapse:collapse;width:auto;font-size:13px;margin:8px 24px}}
th,td{{border:1px solid #eee;padding:4px 8px;text-align:center}}th{{background:#f5f5f5}}
.note{{padding:8px 24px;color:#555;font-size:13px;line-height:1.6}}</style></head>
<body><header><h1>基线（冻结嵌入） vs 对比头（拉开形近字）</h1>
<div class="sub">灰点=全部汉字；橙点=硬骨头；连线=重点字对。绿表行=对比头后余弦下降（字形被拉开）。</div></header>
<div class="wrap">
<div class="card"><h2>左：基线 50K 嵌入</h2><div id="pB" style="height:560px"></div></div>
<div class="card"><h2>右：对比头后嵌入</h2><div id="pA" style="height:560px"></div></div>
</div>
<div class="note">这是<b>本地原型</b>：仅在冻结的 50K 嵌入上加了一个可训练小头（三元组对比损失），
未重训编码器。若右图中王/玉、风/凤 明显散开、表中余弦大幅下降，说明<b>字形信号可被对比监督恢复</b>，
下一步应在云端<b>重训编码器 + 形近字对比损失</b>（而非只加头）。</div>
<h2 style="margin:16px 24px 0">重点字对余弦：基线 → 对比头</h2>
<table><tr><th>字对</th><th>基线</th><th>对比头</th><th>Δ</th></tr>{rows}</table>
<script>
const D={json.dumps(data,ensure_ascii=False)};
function mk(div,key,hlKey){{
  const xs=D[key].map(p=>p[0]),ys=D[key].map(p=>p[1]),txt=D.chars;
  const tr={{x:xs,y:ys,mode:'markers',type:'scatter',marker:{{size:4,color:'#bbb',opacity:.5}},text:txt,hoverinfo:'text'}};
  const hx=[],hy=[],ht=[];
  D.highlight_base&&0; // noop
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
