# -*- coding: utf-8 -*-
"""把全量训练后的判别模型(model_for_web.json)内嵌进一个离线 HTML。
用户输入任意一字 -> 模型(softmax 逻辑回归, 真·训练出来)实时预测其同义词词林语义大类,
并展示"哪些字形维度主导了这个判断" —— 是真预测, 不是查词林表。"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
M = json.load(open(os.path.join(BASE, "model_for_web.json"), encoding="utf-8"))

# 为了让前端 JS 直接看到维度名, 这里拼一个 feat_names 顺序(与训练端 C_full 一致)
n_rad = M["n_rad"]; n_str = M["n_str"]
feat_names = (["部首%d" % r for r in range(n_rad)]
              + ["结构%d" % k for k in range(n_str)]
              + ["笔画数"]
              + ["s32_%d" % i for i in range(26)]
              + ["det_%d" % i for i in range(32)])

HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>字形→语义 可训练判别器</title>
<style>
  :root{--bg:#fafafa;--card:#fff;--ink:#222;--mut:#777;--line:#e3e3e3;--a:#c0392b;--b:#2471a3;--ok:#1e8449}
  *{box-sizing:border-box} body{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
    background:var(--bg);color:var(--ink);padding:18px}
  h1{font-size:19px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin-bottom:14px;line-height:1.6}
  .ctrl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
  input.in{font-size:18px;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;width:130px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
  .card h2{font-size:15px;margin:0 0 10px}
  .big{font-size:54px;font-weight:700;text-align:center;margin:6px 0}
  .pred{text-align:center;font-size:26px;font-weight:700;margin:6px 0;color:var(--b)}
  .probs{font-size:13px}
  .prow{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #eee}
  .prow .lab{font-weight:600}
  .bar{height:8px;background:var(--b);border-radius:4px;margin-top:2px}
  .feat{font-size:13px;padding:3px 0}
  .feat b{color:var(--a)}
  .caveat{font-size:12px;color:var(--mut);line-height:1.6;background:#fff8e9;border:1px solid #f0e0b0;
    border-radius:10px;padding:10px 12px;margin-top:10px}
  .notfound{color:var(--a);font-weight:600}
</style></head><body>
<h1>字形 → 语义 可训练判别器</h1>
<div class="sub">输入任意一字，下面这个<b>真·训练出来的模型</b>（softmax 逻辑回归判别头，本地 CPU 训出，非查表、非死公式）
实时预测它属于《同义词词林》哪个语义大类，并展示"哪些字形维度主导了这个判断"。<br>
模型输入全是你手工设计的结构化字形特征（部首 / 结构 / 笔画数 / 26维矢量树 s32 / 32维树 det），只有"怎么组合下判断"是学的。</div>
<div class="ctrl">
  <label>输入字 <input class="in" id="inA" list="cl" placeholder="打字或选字" autocomplete="off"></label>
  <datalist id="cl"></datalist>
  <button class="q" id="rand">随机抽一个</button>
</div>
<div class="card">
  <h2>模型预测（训练产物，非查表）</h2>
  <div class="big" id="big">—</div>
  <div class="pred" id="pred">请输入一个字</div>
  <div class="probs" id="probs"></div>
</div>
<div class="card">
  <h2>这个判断靠哪些字形维度？（可解读性）</h2>
  <div id="feats" class="probs"></div>
  <div class="caveat" id="caveat"></div>
</div>
<script>
const MODEL = __MODEL__;
const FEAT_NAMES = __FEATNAMES__;
const AGG={"Bh":"生物","Bi":"生物","Bj":"生物","Bl":"生物","Be":"自然物","Bf":"自然物","Bg":"自然物","Bm":"自然物","Bd":"自然物","Ba":"人工物","Bb":"人工物","Bc":"人工物","Bk":"人工物","Bn":"人工物","Bo":"人工物","Bp":"人工物","Bq":"人工物","Br":"人工物"};
const dl=document.getElementById("cl");
Object.keys(MODEL.char_comp).forEach(c=>dl.appendChild(new Option(c,c)));
const inA=document.getElementById("inA");

function z(x,meanArr,stdArr,i){return (x-meanArr[i])/(stdArr[i]+1e-9);}
function featOf(ch){
  const c=MODEL.char_comp[ch]; if(!c)return null;
  const f=[];
  // 部首 onehot -> 标准化
  for(let i=0;i<MODEL.n_rad;i++) f.push(z(i===c.r?1:0, MODEL.rad_ms[0], MODEL.rad_ms[1], i));
  for(let i=0;i<MODEL.n_str;i++) f.push(z(i===c.st?1:0, MODEL.str_ms[0], MODEL.str_ms[1], i));
  f.push(z(c.n, MODEL.scnt_ms[0], MODEL.scnt_ms[1], 0));
  for(let i=0;i<26;i++) f.push(z(c.s32[i], MODEL.s32_ms[0], MODEL.s32_ms[1], i));
  for(let i=0;i<32;i++) f.push(z(c.det[i], MODEL.det_ms[0], MODEL.det_ms[1], i));
  return f;
}
function softmax(a){let m=Math.max.apply(null,a),s=0;const e=a.map(x=>Math.exp(x-m));e.forEach(v=>s+=v);return e.map(v=>v/s);}
function predict(ch){
  const f=featOf(ch); if(!f)return null;
  const W=MODEL.W, b=MODEL.b, L=MODEL.labels;
  const z=new Array(L.length).fill(0);
  for(let j=0;j<L.length;j++){let s=b[j];for(let i=0;i<f.length;i++)s+=f[i]*W[i][j];z[j]=s;}
  const p=softmax(z);
  // 按概率降序
  const order=L.map((lab,j)=>({lab,j,p:p[j]})).sort((a,b)=>b.p-a.p);
  // 若属"物"类，追加物类内部中类判别(信号最强，最能展示真学习)
  let sub=null;
  if(MODEL.char_comp[ch] && MODEL.char_comp[ch].lab==="B" && MODEL.D){
    const WD=MODEL.D.W, bD=MODEL.D.b, ML=MODEL.D.mid_labels;
    const z2=new Array(ML.length).fill(0);
    for(let j=0;j<ML.length;j++){let s=bD[j];for(let i=0;i<f.length;i++)s+=f[i]*WD[i][j];z2[j]=s;}
    const p2=softmax(z2);
    sub=ML.map((m,j)=>({m,j,p:p2[j]})).sort((a,b)=>b.p-a.p);
  }
  return {order, sub};
}
function render(){
  const ch=inA.value.trim();
  if(!ch){document.getElementById("pred").textContent="请输入一个字";return;}
  if(!MODEL.char_comp[ch]){document.getElementById("big").textContent=ch;
    document.getElementById("pred").innerHTML='<span class="notfound">该字无字形特征或词林标签，换一个</span>';return;}
  document.getElementById("big").textContent=ch;
  const pr=predict(ch); if(!pr){document.getElementById("pred").innerHTML='<span class="notfound">无数据</span>';return;}
  const ord=pr.order, top=ord[0];
  document.getElementById("pred").textContent=
    top.lab+" "+MODEL.da_lei[top.lab]+"  (置信度 "+(top.p*100).toFixed(1)+"%)";
  // 概率条
  let html="";
  ord.slice(0,5).forEach(o=>{
    const pct=(o.p*100).toFixed(1);
    html+=`<div class="prow"><span class="lab">${o.lab} ${MODEL.da_lei[o.lab]}</span><span>${pct}%</span></div>`;
    html+=`<div class="bar" style="width:${pct}%"></div>`;
  });
  document.getElementById("probs").innerHTML=html;
  // 主导维度 (该类权重绝对值最大的前5)
  const ci=top.j, W=MODEL.W; const wcol=W.map(row=>row[ci]);
  const idx=wcol.map((w,i)=>({i,w:Math.abs(w)})).sort((a,b)=>b.w-a.w).slice(0,5);
  let fh="<div style='font-weight:600;margin:6px 0 4px'>主导判断的字形维度</div>";
  idx.forEach(o=>{
    const nm=FEAT_NAMES[o.i]; const sign=W[o.i][ci]>=0?"推高":"压低";
    fh+=`<div class="feat">${nm} ：权重 ${W[o.i][ci].toFixed(2)}（${sign}该类概率）</div>`;
  });
  // 物类内部中类判别(信号最强) — 追加在主导维度之后, 不覆盖
  let cav;
  if(pr.sub){
    const D=MODEL.D; const s=pr.sub[0]; const ex=D.mid_example[s.m]||"";
    fh+=`<div class="pred" style="color:var(--ok);font-size:20px;margin:12px 0 4px">`
         +`物类内部中类：${s.m} ${ex?"("+ex+")":""}  (置信度 ${(s.p*100).toFixed(1)}%)</div>`;
    fh+=`<div class="probs">`;
    pr.sub.slice(0,4).forEach(o=>{
      const pct=(o.p*100).toFixed(1); const ex2=D.mid_example[o.m]||"";
      fh+=`<div class="prow"><span class="lab">${o.m} ${ex2}</span><span>${pct}%</span></div>`;
      fh+=`<div class="bar" style="width:${pct}%;background:var(--ok)"></div>`;
    });
    fh+=`</div>`;
    cav=`说明：上方是 11 大类判别（模型在此几乎只能默认"物"，这是实验已证实的负结论）；`
      +`下方是<b>物类内部的 18 中类判别</b>——这是实验中字形（主要是部首/形旁）唯一显出信号的地方：`
      +`测试准确率从基线 14.9% 拉到 44.5%（+29.6pp）。但单字预测会有噪声（18 类细分任务本就难），`
      +`重点看模型在"动物/植物/自然物"大方向上的倾向，而非精确到某一中类。`
      +`两层都是<b>训练出来</b>的判别头（softmax 逻辑回归），不是查《同义词词林》表、也不是 G2 几何公式。`;
  } else {
    cav=`说明：这是<b>训练出来</b>的判别头（softmax 逻辑回归，权重由梯度下降从 5858 字学出），不是查表、也不是 G2 几何公式。`
      +`在 11 大类粒度上模型几乎只能默认"物"（实验已证实字形不携带粗粒度语义类别），故非"物"类的字这里预测不可靠；`
      +`"物"类字会额外显示内部中类判别（那里模型真能区分动物/植物/自然物）。`;
  }
  document.getElementById("feats").innerHTML=fh;
  document.getElementById("caveat").innerHTML=cav;
}
inA.addEventListener("input",render);
document.getElementById("rand").onclick=()=>{
  const ks=Object.keys(MODEL.char_comp); inA.value=ks[Math.floor(Math.random()*ks.length)]; render();};
render();
</script></body></html>"""

html = HTML.replace("__MODEL__", json.dumps(M, ensure_ascii=False, separators=(",", ":")))
html = html.replace("__FEATNAMES__", json.dumps(feat_names, ensure_ascii=False))
out = os.path.join(BASE, "glyph_semantic_predictor.html")
open(out, "w", encoding="utf-8").write(html)
print("[OK] 预测 demo ->", out, " 大小 %.1f MB" % (os.path.getsize(out)/1e6))
