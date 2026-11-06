# -*- coding: utf-8 -*-
"""
tiny 字符语言模型（本地 CPU / numpy，零依赖）
=========================================
用户选定路线 C：本地训 tiny 模型，不花云钱，产出"涌现语意"。

模型 = learned-embedding bigram 字符 LM：
  P(下字=c2 | 本字=c1) = softmax( E[c1] · E[c2]ᵀ )
  E 是共享字 embedding 矩阵 (V × D)，通过最大化下一字似然训出来。
  → 输入一字即输出下一字概率分布（用户说的"语言模型"）；
  → 训出的 E 就是语义向量：用法相近的字在向量空间里靠近（涌现语意 + 泛化）。

数据：10 域中文 KG CSV，每字段抽 CJK 字符序列，取有序相邻对 (c1->c2)。
训练：Adam，全 softmax（V 仅数千，CPU 可承受）。

输出：
  embeddings.npy (V×D float32)  —— 语义向量
  vocab.json                  —— 字→idx
  results.json               —— 下一字 top-k 准确率 / 语义邻字样本 / 与计数基线对比
  char_lm_graph.html         —— 离线交互：中心字→embedding 语义邻字(放射图)
"""
import os, csv, json, time, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
KG_DIR = r"D:/desktop/学习结果小文件/neuroglyph34_project/knowledge_graph"
OUT_DIR = BASE

# ---------- 超参 ----------
MAX_ROWS_PER_FILE = 300_000   # 每域最多读多少行（控制子集规模，保证跨域覆盖 + 快速）
MAX_PAIRS         = 8_000_000 # 全局最多保留多少有序对
VOCAB_THRESH      = 30        # 字出现次数阈值（入词表）
D                 = 48        # embedding 维度
EPOCHS            = 4
BATCH             = 2048
LR                = 0.05
SEED              = 20260712
rng = np.random.default_rng(SEED)

CJK_LO = 0x4E00   # 中日韩统一表意文字起点
CJK_SPAN = 0x9FFF - 0x4E00 + 1   # ≈ 20992

def is_cjk(ch):
    return 0x4E00 <= ord(ch) <= 0x9FFF

# ---------- 1. 抽数据 ----------
def build_pairs():
    prev_ord = np.empty(MAX_PAIRS, dtype=np.int32)
    next_ord = np.empty(MAX_PAIRS, dtype=np.int32)
    n = 0
    files = sorted(f for f in os.listdir(KG_DIR) if f.endswith('.csv'))
    print(f"[data] 扫描 {len(files)} 域 CSV (每域≤{MAX_ROWS_PER_FILE}行, 全局≤{MAX_PAIRS}对)...", flush=True)
    for fi, fname in enumerate(files, 1):
        path = os.path.join(KG_DIR, fname)
        rows = 0
        with open(path, encoding='utf-8', errors='replace', newline='') as fh:
            r = csv.reader(fh)
            for row in r:
                rows += 1
                if rows > MAX_ROWS_PER_FILE:
                    break
                for field in row:
                    if not field:
                        continue
                    seq = [c for c in field if is_cjk(c)]
                    for i in range(len(seq) - 1):
                        if n >= MAX_PAIRS:
                            break
                        prev_ord[n] = ord(seq[i]); next_ord[n] = ord(seq[i+1]); n += 1
                if n >= MAX_PAIRS:
                    break
        print(f"  [{fi}/{len(files)}] {fname}  已采 {n:,} 对", flush=True)
        if n >= MAX_PAIRS:
            break
    prev_ord = prev_ord[:n]; next_ord = next_ord[:n]
    print(f"[data] 采集有序对 {n:,}", flush=True)

    # 词表：出现频次 >= VOCAB_THRESH
    all_ord = np.concatenate([prev_ord, next_ord])
    freq = np.bincount(all_ord - CJK_LO, minlength=CJK_SPAN + 1)
    vocab_ords = CJK_LO + np.where(freq >= VOCAB_THRESH)[0]
    V = len(vocab_ords)
    print(f"[data] 词表大小 V = {V:,} (阈值≥{VOCAB_THRESH})", flush=True)
    idx_lookup = np.full(CJK_SPAN + 1, -1, dtype=np.int16)
    idx_lookup[vocab_ords - CJK_LO] = np.arange(V, dtype=np.int16)
    present = idx_lookup >= 0
    valid = present[prev_ord - CJK_LO] & present[next_ord - CJK_LO]
    p = prev_ord[valid]; nx = next_ord[valid]
    prev_idx = idx_lookup[p - CJK_LO].astype(np.int32)
    next_idx = idx_lookup[nx - CJK_LO].astype(np.int32)
    print(f"[data] 入表有效对 {len(prev_idx):,}", flush=True)
    vocab = [chr(o) for o in vocab_ords]
    return prev_idx, next_idx, vocab, V

# ---------- 2. 训练 ----------
def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def train(prev_idx, next_idx, V):
    N = len(prev_idx)
    perm = rng.permutation(N)
    prev_idx = prev_idx[perm].astype(np.int32)
    next_idx = next_idx[perm].astype(np.int32)
    # 训练/留出
    n_test = N // 10
    te_p, te_n = prev_idx[:n_test], next_idx[:n_test]
    tr_p, tr_n = prev_idx[n_test:], next_idx[n_test:]
    print(f"[train] 训练 {len(tr_p):,} / 留出 {len(te_p):,}", flush=True)

    E = (rng.standard_normal((V, D)).astype(np.float32) * 0.1)
    m = np.zeros_like(E); v = np.zeros_like(E)
    t = 0
    steps_per_epoch = (len(tr_p) + BATCH - 1) // BATCH
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        loss_ep = 0.0
        for s in range(0, len(tr_p), BATCH):
            b = min(BATCH, len(tr_p) - s)
            pi = tr_p[s:s+b]; ni = tr_n[s:s+b]
            Pem = E[pi]                      # (b,D)
            scores = Pem @ E.T              # (b,V)
            P = softmax_rows(scores)        # (b,V)
            # 交叉熵
            corr = P[np.arange(b), ni]
            loss = -np.mean(np.log(corr + 1e-8))
            loss_ep += loss * b
            # 梯度
            dlog = P.copy(); dlog[np.arange(b), ni] -= 1.0; dlog /= b
            dE_cols = dlog.T @ Pem           # (V,D)  —— 下一字侧
            dPem = dlog @ E                  # (b,D)  —— 本字侧
            dE_rows = np.zeros_like(E)
            np.add.at(dE_rows, pi, dPem)     # 散射到本字行
            g = dE_cols + dE_rows            # (V,D)
            # Adam
            t += 1
            m[:] = 0.9*m + 0.1*g
            v[:] = 0.999*v + 0.001*(g*g)
            mhat = m / (1 - 0.9**t)
            vhat = v / (1 - 0.999**t)
            E -= LR * mhat / (np.sqrt(vhat) + 1e-8)
        loss_ep /= len(tr_p)
        # 留出 top-k
        acc1, acc5 = eval_topk(E, te_p, te_n)
        el = time.time() - t0
        print(f"[train] epoch {ep}/{EPOCHS}  loss={loss_ep:.4f}  留出 top1={acc1*100:.2f}% top5={acc5*100:.2f}%  ({el/60:.1f}min)", flush=True)
    return E, (te_p, te_n)

def eval_topk(E, te_p, te_n, k=5):
    b = 4096
    hit1 = hit5 = 0; tot = 0
    for s in range(0, len(te_p), b):
        pi = te_p[s:s+b]; ni = te_n[s:s+b]
        scores = E[pi] @ E.T
        top = np.argsort(-scores, axis=1)[:, :k]
        for j, tgt in enumerate(ni):
            if tgt == top[j, 0]: hit1 += 1
            if tgt in top[j]: hit5 += 1
            tot += 1
    return hit1/tot, hit5/tot

# ---------- 3. 计数基线（同任务，ML 估计） ----------
def count_baseline(prev_idx, next_idx, V, te_p, te_n):
    # 对每个本字，argmax count 表 作为预测
    cnt = np.zeros((V, V), dtype=np.int64)
    np.add.at(cnt, (prev_idx, next_idx), 1)
    pred = np.argmax(cnt, axis=1)   # (V,)
    hit1 = sum(1 for j in range(len(te_p)) if pred[te_p[j]] == te_n[j])
    # top5
    top = np.argsort(-cnt, axis=1)[:, :5]
    hit5 = sum(1 for j in range(len(te_p)) if te_n[j] in top[te_p[j]])
    return hit1/len(te_p), hit5/len(te_p)

# ---------- 4. 语义邻字（embedding 余弦） ----------
def semantic_neighbors(E, vocab, demo, topk=10):
    V = E.shape[0]
    Et = E / np.linalg.norm(E, axis=1, keepdims=True)
    res = {}
    for c in demo:
        if c not in vocab:
            continue
        i = vocab.index(c)
        sim = Et @ Et[i]
        order = np.argsort(-sim)
        nb = [(vocab[o], round(float(sim[o]), 3)) for o in order[:topk+1] if o != i]
        res[c] = nb[:topk]
    return res

# ---------- 5. 可视化（离线放射图） ----------
def build_viz(E, vocab, neighbors, out_html):
    V = E.shape[0]
    Et = E / np.linalg.norm(E, axis=1, keepdims=True)
    # 演示字
    demo_pool = list("长牛明江中水大国学木火山金土日月天山风人心王子女时年文书人法市生力小高")
    demos = [c for c in demo_pool if c in vocab][:14]
    data = {}
    for c in demos:
        i = vocab.index(c)
        sim = Et @ Et[i]
        order = np.argsort(-sim)
        nb = [[vocab[o], round(float(sim[o]), 3)] for o in order[:11] if o != i][:10]
        data[c] = nb
    data_js = json.dumps(data, ensure_ascii=False)
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>字语义向量 · embedding 邻字（本地训练 tiny LM）</title>
<style>
:root{--bg:#f6f8fa;--card:#fff;--ink:#1f2933;--sub:#687586;--line:#e3e8ee;--accent:#7c3aed;--good:#0b6e57}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;line-height:1.6;padding:24px}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.lead{color:var(--sub);font-size:14px;margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.io{display:flex;align-items:center;gap:14px;justify-content:center;margin:6px 0 12px}
.box{width:90px;height:90px;font-size:50px;text-align:center;border:2px solid var(--line);border-radius:12px;outline:none;color:var(--ink);background:#fbfdff;font-family:inherit}
.box:focus{border-color:var(--accent)}
.btn{background:var(--accent);color:#fff;border:none;border-radius:10px;padding:11px 22px;font-size:15px;cursor:pointer}
.presets{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:6px}
.chip{background:#f1ebfa;color:var(--accent);border:1px solid #e3d4f5;border-radius:999px;padding:5px 13px;font-size:14px;cursor:pointer}
.chip:hover{background:#e7dbf4}
.note{font-size:13px;color:var(--sub);background:#f7f4fc;border-left:3px solid var(--accent);padding:8px 12px;border-radius:6px;margin-top:12px}
.foot{color:var(--sub);font-size:12px;text-align:center;margin-top:6px}
svg{display:block;margin:0 auto}
.tlabel{font-size:13px;fill:var(--ink)}
</style></head><body><div class="wrap">
<h1>字语义向量 · embedding 邻字</h1>
<p class="lead">这是 <b>本地 CPU 训出来的 tiny 字符语言模型</b> 学出的 <b>字 embedding</b>：输入一字→下一字概率，同时把"用法相近"的字压进向量空间。下面每个中心字周围的字，是它 embedding 余弦最近的字——即模型<b>自己涌现出的语义邻域</b>（不是计数表，是训练出的分布式表示）。</p>
<div class="card">
  <div class="io"><input id="a" class="box" maxlength="1" value="牛"/><button class="btn" onclick="run()">语义邻字是谁？</button></div>
  <div class="presets" id="presets"></div>
</div>
<div class="card res" id="result" style="display:none">
  <div class="big" id="headline"></div>
  <div id="graph"></div>
  <div class="note" id="note"></div>
</div>
<div class="foot" id="foot"></div>
</div>
<script>
const NB = __DATA__;
const META = {demos: __DEMOS__};
function run(){
  const c=document.getElementById("a").value.trim();
  const res=document.getElementById("result");
  const note=document.getElementById("note");
  const headline=document.getElementById("headline");
  if(!c){alert("请输入一个字");return;}
  const top=NB[c];
  if(!top){res.style.display="block";headline.textContent="";document.getElementById("graph").innerHTML="";
    note.innerHTML='<span style="color:#c0392b;font-weight:600">字库未收录</span>：该字在训练词表外。试常见字（牛/江/明/中/水…）。';return;}
  res.style.display="block";
  headline.innerHTML='「'+c+'」的 <b>语义向量邻字</b>（embedding 余弦最近，模型涌现的语义邻域）：';
  const W=560,H=380,cx=W/2,cy=H/2,R=140,n=top.length;
  let s='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
  for(let i=0;i<n;i++){
    const ang=-Math.PI/2+i*2*Math.PI/n;
    const x=cx+R*Math.cos(ang),y=cy+R*Math.sin(ang);
    const w=Math.max(1.2,top[i][1]*40);
    s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+x.toFixed(1)+'" y2="'+y.toFixed(1)+'" stroke="#7c3aed" stroke-width="'+w.toFixed(2)+'" stroke-opacity="0.55"/>';
    s+='<text x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" class="tlabel" text-anchor="middle" dy="4">'+top[i][0]+'</text>';
    s+='<text x="'+((cx+x)/2).toFixed(1)+'" y="'+((cy+y)/2).toFixed(1)+'" font-size="10" fill="#7c3aed" text-anchor="middle" dy="-2">'+(top[i][1]*100).toFixed(0)+'%</text>';
  }
  s+='<circle cx="'+cx+'" cy="'+cy+'" r="30" fill="#7c3aed"/>';
  s+='<text x="'+cx+'" y="'+cy+'" class="tlabel" fill="#fff" text-anchor="middle" dy="6" style="font-size:22px;font-weight:700">'+c+'</text></svg>';
  document.getElementById("graph").innerHTML=s;
  note.innerHTML='这些字不是"和'+c+'共现次数最多"的计数结果，而是模型把'+c+'压进向量空间后，<b>几何上最接近</b>的字——同一个向量空间里，语义/用法相近的字自然聚拢。这就是"涌现语意"。';
}
const pc=document.getElementById("presets");
META.demos.forEach(d=>{const s=document.createElement("span");s.className="chip";s.textContent=d;
  s.onclick=()=>{document.getElementById("a").value=d;run();};pc.appendChild(s);});
document.getElementById("foot").textContent="tiny 字符LM · embedding 维度 "+__DIM__+" · 本地CPU训练 · 零云钱 · 双击即用";
run();
</script></body></html>"""
    html = html.replace("__DATA__", data_js).replace("__DEMOS__", json.dumps(demos, ensure_ascii=False)).replace("__DIM__", str(E.shape[1]))
    open(out_html, 'w', encoding='utf-8').write(html)
    return demos

# ---------- main ----------
def main():
    prev_idx, next_idx, vocab, V = build_pairs()
    E, held = train(prev_idx, next_idx, V)
    te_p, te_n = held
    acc1, acc5 = eval_topk(E, te_p, te_n)
    base1, base5 = count_baseline(prev_idx, next_idx, V, te_p, te_n)
    demo = list("长牛明江中水大国学木火山金土日月天山风人心王子女时年文书人法市生力小高")
    neigh = semantic_neighbors(E, vocab, demo, topk=10)
    demos = build_viz(E, vocab, neigh, os.path.join(OUT_DIR, "char_lm_graph.html"))
    # 保存
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), E.astype(np.float32))
    json.dump(vocab, open(os.path.join(OUT_DIR, "vocab.json"), 'w', encoding='utf-8'), ensure_ascii=False)
    results = {
        "V": V, "D": D, "epochs": EPOCHS, "subset_rows_per_file": MAX_ROWS_PER_FILE,
        "trained_nextchar_top1": round(acc1, 4), "trained_nextchar_top5": round(acc5, 4),
        "count_baseline_top1": round(base1, 4), "count_baseline_top5": round(base5, 4),
        "semantic_neighbors_sample": {k: v for k, v in list(neigh.items())[:16]},
        "demo_chars": demos,
    }
    json.dump(results, open(os.path.join(OUT_DIR, "results.json"), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print("\n===== 结果 =====", flush=True)
    print(f"下一字预测  训练模型 top1={acc1*100:.2f}% top5={acc5*100:.2f}%", flush=True)
    print(f"下一字预测  计数基线 top1={base1*100:.2f}% top5={base5*100:.2f}%", flush=True)
    print("语义邻字样本(embedding余弦):", flush=True)
    for c in demo:
        if c in neigh:
            print(f"  {c} -> " + "  ".join(f"{x[0]}({x[1]:.2f})" for x in neigh[c][:8]), flush=True)
    print(f"\n[done] embeddings.npy / vocab.json / results.json / char_lm_graph.html", flush=True)

if __name__ == '__main__':
    main()
