# -*- coding: utf-8 -*-
"""
字·形 vs 义 对比台 (v2, 词林权威语义版)
=====================================
相比 v1（右侧用自搭 10 域 tf-idf，受齐夫律影响弱）：
  - 语义：换用《同义词词林（扩展版）》(哈工大 IR-Lab) 权威分类。
          单字 -> 分类码映射见 cilin_index.json（71% 独立词条 + 17% 复合词推断 = 88.5% 覆盖）。
          相似度用田久乐/朱新华(2010) 公式（基于分类码前缀共享），前端 JS 实时复现。
  - 字形：仍用官方 G2 backbone（树 det×矢量×R2），前端实时复现，任意两字当场算。
  - UI：input + datalist 任意字输入。

纯可视化，零训练，离线，双击即用。
输出: char_compare.html
"""
import os, json, math
BASE = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"

# ---------- 语义: 同义词词林索引 ----------
cilin = json.load(open(os.path.join(BASE, "cilin_index.json"), encoding="utf-8"))
char_codes = cilin["char_codes"]
targets = list(char_codes.keys())

# ---------- 字形: 全量特征内嵌 ----------
TREE = json.load(open(os.path.join(BASE, "tree_data.json"), encoding="utf-8"))
SF = json.load(open(os.path.join(BASE, "stroke_features.json"), encoding="utf-8"))
NO = json.load(open(os.path.join("D:/desktop/学习结果小文件/neuroglyph34_project",
                                  "struct32_aligned.json"), encoding="utf-8"))["nopy26"]

chars = {}
for c in targets:
    t = TREE.get(c); sf = SF.get(c); s32 = NO.get(c)
    if not (t and sf and s32):
        continue
    chars[c] = {"rad": t.get("rad"), "str": t.get("str"), "det": t.get("det"),
                "strokes": sf, "s32": s32}

data = {
    "targets": targets,
    "chars": chars,
    "cilin": {"da_lei": cilin["da_lei"], "char_codes": cilin["char_codes"],
              "branch_n": cilin["branch_n"], "code_words": cilin["code_words"]},
    "meta": {
        "note": "v2 词林版：语义用《同义词词林(扩展版)》(哈工大 IR-Lab) 权威分类 + 田久乐/朱新华相似度；"
                "字形用官方 G2 backbone(前端实时复现)。纯可视化,零训练.",
    },
}

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>字·形 vs 义 对比台（词林权威语义版）</title>
<style>
  :root{ --bg:#fafafa; --card:#fff; --ink:#222; --mut:#777; --line:#e3e3e3;
         --a:#c0392b; --b:#2471a3; --ok:#1e8449; --warn:#b9770e; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
       background:var(--bg);color:var(--ink);padding:18px;}
  h1{font-size:19px;margin:0 0 4px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:14px;line-height:1.6}
  .ctrl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
  input.in{font-size:16px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;width:120px}
  button.q{font-size:13px;padding:5px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}
  button.q:hover{background:#f0f0f0}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:760px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
  .card h2{font-size:15px;margin:0 0 10px;color:var(--ink)}
  .bigpair{display:flex;gap:22px;justify-content:center;margin:6px 0 12px}
  .big{font-size:46px;font-weight:700;line-height:1}
  .rows{font-size:14px}
  .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px dashed #eee}
  .row span:last-child{font-weight:600}
  .same{color:var(--ok)} .diff{color:var(--a)}
  .g2{text-align:center;font-size:22px;font-weight:700;margin:10px 0 2px}
  .g2sub{text-align:center;color:var(--mut);font-size:12px}
  .verdict{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:14px;font-size:15px;line-height:1.6}
  .verdict b{color:var(--ink)}
  .caveat{margin-top:12px;font-size:12px;color:var(--mut);line-height:1.6;background:#fff8e9;border:1px solid #f0e0b0;border-radius:10px;padding:10px 12px}
  .bad{color:var(--a);font-weight:600}
  .semcard{margin:8px 0;padding:8px 10px;border:1px solid #eee;border-radius:8px;background:#fcfcfc}
  .semhead{font-size:18px;font-weight:700;margin-bottom:4px}
  .semtag{font-size:11px;color:#fff;background:var(--b);border-radius:4px;padding:1px 6px;margin-left:6px;vertical-align:2px}
  .semln{font-size:13px;padding:2px 0}
  .semln code{background:#eef;padding:1px 5px;border-radius:4px;font-family:ui-monospace,monospace;color:#224}
  .semwords{color:var(--mut)}
  .mut{color:var(--mut);font-size:12px}
  .top2{font-size:12px;color:var(--mut);text-align:center;margin-top:6px;line-height:1.7}
</style>
</head>
<body>
<h1>字·形 vs 义 对比台 <span style="font-size:13px;color:var(--mut)">（词林权威语义版 v2）</span></h1>
<div class="sub">左边看「字长什么样」（官方 G2 backbone，已定稿，前端实时复现）· 右边看「字归哪类」（<b>《同义词词林·扩展版》哈工大 IR-Lab 权威分类</b> + 田久乐/朱新华相似度）。<br>
<b>直接在两个框里打字或下拉选字</b>，任意两个字当场对比。纯可视化，零训练。</div>

<div class="ctrl">
  <label>字 A <input class="in" id="inA" list="charList" placeholder="输入/选字A" autocomplete="off"></label>
  <label>字 B <input class="in" id="inB" list="charList" placeholder="输入/选字B" autocomplete="off"></label>
  <datalist id="charList"></datalist>
  <span style="color:var(--mut);font-size:13px">快速对：</span>
  <button class="q" data-a="牛" data-b="羊">牛 / 羊</button>
  <button class="q" data-a="猫" data-b="狗">猫 / 狗</button>
  <button class="q" data-a="猫" data-b="树">猫 / 树</button>
  <button class="q" data-a="末" data-b="未">末 / 未</button>
  <button class="q" data-a="江" data-b="河">江 / 河</button>
  <span style="color:var(--mut);font-size:13px;margin-left:auto">布局：</span>
  <button class="q" id="mBoth">并排</button>
  <button class="q" id="mGlyph">只看字形</button>
  <button class="q" id="mSem">只看语义</button>
</div>

<div class="grid">
  <div class="card" id="cardGlyph">
    <h2>字形（G2 官方 backbone，实时复现）</h2>
    <div class="bigpair"><div class="big" id="bigA">牛</div><div class="big" id="bigB">羊</div></div>
    <div class="rows" id="glyphRows"></div>
    <div class="g2" id="g2val">—</div>
    <div class="g2sub">字形相似度 G2（越低 = 两个字形分得越开；≥20% 视为难分，末/未=15.8% 是典型「形近」）</div>
  </div>
  <div class="card" id="cardSem">
    <h2>语义（同义词词林·扩展版，权威分类）</h2>
    <div class="semcard" id="semA"></div>
    <div class="semcard" id="semB"></div>
    <div class="g2" id="cilinSim">—</div>
    <div class="g2sub">词林语义相似度（田久乐/朱新华公式，0~1，越高=语义越近）</div>
    <div class="top2" id="cilinNote"></div>
  </div>
</div>

<div class="verdict" id="verdict">—</div>
<div class="caveat" id="caveat"></div>

<script>
const DATA = __DATA__;
const dl=document.getElementById("charList");
DATA.targets.forEach(c=>dl.appendChild(new Option(c,c)));
const inA=document.getElementById("inA"), inB=document.getElementById("inB");
inA.value="牛"; inB.value="羊";

// ===== 字形 G2 官方 backbone (JS 复现 Python 版) =====
function cosine(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-12);}
function treeSim(a,b){const ta=DATA.chars[a],tb=DATA.chars[b];if(!ta||!tb)return null;
  if(ta.rad!==tb.rad)return 0;
  const det=Math.max(0,cosine(ta.det,tb.det));
  return ta.str===tb.str?det:0.5*det;}
function vSim(a,b){const fa=DATA.chars[a].strokes,fb=DATA.chars[b].strokes;
  if(!fa||!fb)return null;const nA=fa.length,nB=fb.length;
  const base=1-Math.abs(nA-nB)/Math.max(nA,nB);const m=Math.min(nA,nB);
  if(m===0)return 0;const gs=[];
  for(let i=0;i<m;i++){const sa=fa[i],sb=fb[i];
    const lsim=1-Math.abs(sa.fl-sb.fl);
    const rsim=1-Math.abs(sa.rank-sb.rank)/Math.max(nA,nB);
    const dsim=(sa.sin*sb.sin+sa.cos*sb.cos+1)/2;
    const pd=Math.hypot(sa.x-sb.x,sa.y-sb.y);
    const psim=Math.max(0,1-pd/0.4);
    gs.push(0.28*lsim+0.50*rsim+0.12*dsim+0.10*psim);}
  let s=0;for(const x of gs)s+=x;return base*(s/m);}
function relSim2(a,b){const fa=DATA.chars[a].strokes,fb=DATA.chars[b].strokes;
  if(!fa||!fb)return null;const nA=fa.length,nB=fb.length;const m=Math.min(nA,nB);
  if(m===0)return 0;
  const cnt_f=1-0.6*Math.abs(nA-nB)/Math.max(nA,nB,1);
  const va=DATA.chars[a].s32,vb=DATA.chars[b].s32;let type_match=0;
  if(va&&vb){const cap=Math.min(va[9]|0,10),cap2=Math.min(vb[9]|0,10);
    const ta=[],tb=[];for(let i=0;i<cap;i++)ta.push(va[16+i]);for(let i=0;i<cap2;i++)tb.push(vb[16+i]);
    const mm=Math.min(ta.length,tb.length);let tm=0;for(let i=0;i<mm;i++)if(ta[i]===tb[i])tm++;
    type_match= tm/mm;}
  const La=[],Lb=[];for(let i=0;i<m;i++){La.push(fa[i].L);Lb.push(fb[i].L);}
  const maxL=Math.max.apply(null,La.concat(Lb))+1e-9;let prof=1;
  for(let i=0;i<m;i++)prof*=Math.max(0,1-Math.abs(La[i]-Lb[i])/maxL);
  const ra=[],rb=[];for(let i=0;i<m;i++){ra.push(fa[i].rank);rb.push(fb[i].rank);}
  let mt=0,tot=0;
  for(let i=0;i<m;i++)for(let j=i+1;j<m;j++){
    const sa=ra[i]<ra[j]?1:(ra[i]>ra[j]?0:2);
    const sb=rb[i]<rb[j]?1:(rb[i]>rb[j]?0:2);
    tot++;if(sa===sb)mt++;}
  const rankord=tot?mt/tot:1;
  return cnt_f*type_match*prof*rankord;}
function gSim(a,b){const t=treeSim(a,b),v=vSim(a,b),r=relSim2(a,b);
  if(t===null||v===null||r===null)return null;return t*v*r;}
function radLit(rad){return (typeof rad==="string"&&rad.length===1)?rad:("#"+rad);}

// ===== 语义：同义词词林（扩展版）田久乐/朱新华相似度 =====
function newLayer(c){return [c[0],c[1],c.substring(2,4),c[4],c.substring(5,7),c[7]];}
function commonCode(c1,c2){let L=Math.min(c1.length,c2.length),i=0;
  while(i<L&&c1[i]===c2[i])i++; if(i===3||i===6)return c1.substring(0,i-1); return c1.substring(0,i);}
function getK(l1,l2){for(let i=0;i<5;i++){
    if((i===2||i===4)&&l1[i]!==l2[i])return Math.abs(parseInt(l1[i])-parseInt(l2[i]));
    else if(l1[i]!==l2[i])return Math.abs(l1[i].charCodeAt(0)-l2[i].charCodeAt(0));} return 0;}
const PARAM={a:0.65,b:0.8,c:0.9,d:0.96,e:0.5,f:0.1};
const LT={1:1,2:2,4:3,5:4};
function simCilin(c1,c2){
  if(c1===c2)return 1;
  const l1=newLayer(c1),l2=newLayer(c2);
  const cs=commonCode(c1,c2),cl=cs.length;
  if(cl===0)return PARAM.f;                       // 放宽 @：去掉原实现把 '@' 判 0.1 的全局惩罚
  if(cl>=7){const r1=c1[7],r2=c2[7]; if(r1===r2)return r1==='='?1:(r1==='#'?PARAM.e:0.7); return 0.7;}
  const layer=LT[cl];
  const coef=[null,PARAM.a,PARAM.b,PARAM.c,PARAM.d][layer];
  const n=DATA.cilin.branch_n[cs]||1;
  const k=getK(l1,l2);
  let v=coef*Math.cos(n*Math.PI/180)*(n-k+1)/n;
  return Math.min(1,Math.max(0,v));
}
function codesOf(c){const cc=DATA.cilin.char_codes[c];return cc?cc.codes:[];}
function charSim(a,b){const A=codesOf(a),B=codesOf(b);
  if(!A.length||!B.length)return null;
  let best=0,bc1=null,bc2=null;
  for(const x of A)for(const y of B){const s=simCilin(x,y);if(s>best){best=s;bc1=x;bc2=y;}}
  return {sim:best,c1:bc1,c2:bc2};}
function describeShared(c1,c2){
  const cl=commonCode(c1,c2).length;
  if(cl>=7)return "近同义（同一原子词群）";
  if(cl===5)return "共享到【词群】层";
  if(cl===4)return "共享到【小类】层";
  if(cl===2)return "共享【大类+中类】";
  if(cl===1)return "仅共享【大类】";
  return "无共同分类（完全不同语义域）";}

function renderSem(c, elId){
  const cc=DATA.cilin.char_codes[c], el=document.getElementById(elId);
  if(!cc||cc.codes.length===0){el.innerHTML=`<div class="semhead">${c} <span class="bad">词林无数据</span></div>`;return;}
  let html=`<div class="semhead">${c} <span class="semtag">${cc.src==='exact'?'独立词条':cc.src==='compound'?'复合词推断':'无'}</span></div>`;
  cc.codes.slice(0,4).forEach(code=>{
    const da=DATA.cilin.da_lei[code[0]]||code[0];
    const words=(DATA.cilin.code_words[code]||[]).slice(0,14).join("、");
    html+=`<div class="semln"><b>${da}</b> <code>${code}</code> ${words?`<span class="semwords">${words}</span>`:""}</div>`;
  });
  if(cc.codes.length>4)html+=`<div class="semln mut">…另有 ${cc.codes.length-4} 个义项</div>`;
  el.innerHTML=html;
}

function render(){
  const a=inA.value.trim(), b=inB.value.trim();
  if(!a||!b){document.getElementById("verdict").textContent="请输入两个字";return;}
  if(!DATA.chars[a]||!DATA.chars[b]){
    document.getElementById("verdict").innerHTML='<span class="bad">有字不在数据集中，换个字试试</span>';return;}
  document.getElementById("bigA").textContent=a;
  document.getElementById("bigB").textContent=b;
  const ca=DATA.chars[a], cb=DATA.chars[b];
  const radSame=ca.rad===cb.rad;
  const rows=[
    ["部首", `${radLit(ca.rad)}${radSame?' <span class="same">(同)</span>':' <span class="diff">(异)</span>'}`,
              `${radLit(cb.rad)}${radSame?' <span class="same">(同)</span>':' <span class="diff">(异)</span>'}`],
    ["笔画数", `${ca.strokes.length}`, `${cb.strokes.length}`],
    ["结构类型", `${ca.str}`, `${cb.str}`],
  ];
  document.getElementById("glyphRows").innerHTML=rows.map(r=>
    `<div class="row"><span>${r[0]}</span><span>${r[1]}</span><span>${r[2]}</span></div>`).join("");
  const g=gSim(a,b);
  document.getElementById("g2val").textContent = g===null?"缺字形数据":(g*100).toFixed(1)+"%";

  renderSem(a,"semA"); renderSem(b,"semB");
  const sm=charSim(a,b);
  document.getElementById("cilinSim").textContent = sm? (sm.sim*100).toFixed(1)+"%" : "无语义数据";
  document.getElementById("cilinNote").innerHTML = sm
    ? `最佳匹配码对 <code>${sm.c1}</code> ↔ <code>${sm.c2}</code><br>${describeShared(sm.c1,sm.c2)}`
    : "至少一字在词林中无数据";

  // 结论（规则，非学习）
  const glyphLike = (g!==null && g>=0.10);
  const band = sm? (sm.sim>=0.65?"义近": sm.sim>=0.30?"义中等":"义远") : "义未知";
  const gv = g===null?"?":(g*100).toFixed(1);
  const sv = sm? (sm.sim*100).toFixed(0) : "?";
  let txt;
  if(band==="义未知") txt="至少一字词林无数据，仅字形可比。";
  else if(glyphLike && band==="义近") txt=`两字<b>字形像</b>（G2 ${gv}%），且<b>语义也近</b>（词林 ${sv}%）——形近义近。`;
  else if(glyphLike && band==="义远") txt=`两字<b>字形像</b>（G2 ${gv}%），但<b>语义几乎无关</b>（词林 ${sv}%）——典型「形近义异」易混字！`;
  else if(glyphLike && band==="义中等") txt=`两字<b>字形像</b>（G2 ${gv}%），语义中等相关（同属一大类，如"物"）。`;
  else if(!glyphLike && band==="义近") txt=`两字<b>字形不像</b>（G2 ${gv}%），但<b>语义近</b>（词林 ${sv}%）——义近而形无关。`;
  else if(!glyphLike && band==="义远") txt=`两字<b>字形不像</b>（G2 ${gv}%），语义也远（词林 ${sv}%）——基本无关。`;
  else txt=`两字字形不像（G2 ${gv}%），语义中等相关（词林 ${sv}%）。`;
  document.getElementById("verdict").innerHTML="结论（规则生成，非学习）： "+txt;
  document.getElementById("caveat").innerHTML=
    "诚实边界：① 语义来自《同义词词林(扩展版)》(哈工大 IR-Lab) 公开资源，是<b>人为预先结构化、可解读、零训练</b>的权威分类，非模型输出。"+
    "② 词林是「词」级词典，单字覆盖 88.5%（71% 独立词条 + 17% 复合词推断），生僻/结构字标「无数据」。"+
    "③ 相似度用田久乐/朱新华公式（按分类码前缀共享），绝对值偏中等、且对 @ 自闭环词做了放宽（否则猫这类词对所有词都=0.1）。"+
    "④ 字形 G2 由前端实时复现官方 backbone；底部结论是 if-then 规则拼的，不训练、不推理。";
}
function setMode(m){
  document.getElementById("cardGlyph").style.display = (m==="glyph") ? "none" : "";
  document.getElementById("cardSem").style.display   = (m==="sem")   ? "none" : "";
  document.getElementById("mBoth").style.background = (m==="both")  ? "#e8f0fe" : "#fff";
  document.getElementById("mGlyph").style.background = (m==="glyph") ? "#e8f0fe" : "#fff";
  document.getElementById("mSem").style.background   = (m==="sem")   ? "#e8f0fe" : "#fff";
}
document.getElementById("mBoth").onclick = ()=>setMode("both");
document.getElementById("mGlyph").onclick = ()=>setMode("glyph");
document.getElementById("mSem").onclick  = ()=>setMode("sem");
inA.addEventListener("input",render);
inB.addEventListener("input",render);
document.querySelectorAll("button.q").forEach(btn=>btn.onclick=()=>{inA.value=btn.dataset.a;inB.value=btn.dataset.b;render();});
render();
</script>
</body>
</html>
"""
html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
out = os.path.join(BASE, "char_compare.html")
open(out, "w", encoding="utf-8").write(html)
print("done ->", out)
print("targets:", len(targets), " chars with full glyph:", len(chars),
      " cilin codes used:", len(data["cilin"]["char_codes"]))
