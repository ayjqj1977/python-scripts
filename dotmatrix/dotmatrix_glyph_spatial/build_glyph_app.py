# -*- coding: utf-8 -*-
"""build_glyph_app.py — 嵌入 glyph_embeddings.json + tree_data.json + stroke_features.json + 笔顺类型,
生成自包含 HTML 应用(六种看字方式: 组合G / 组合C / 组合树T / 矢量笔画V / 结构化S / 纯像素R)
G = 组合树 × 矢量 × 笔画级关系, 是字形判别当前最强。"""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = json.loads(open(os.path.join(BASE, "glyph_embeddings.json"), encoding="utf-8").read())
TREE = open(os.path.join(BASE, "tree_data.json"), encoding="utf-8").read()
STROKE = open(os.path.join(BASE, "stroke_features.json"), encoding="utf-8").read()

# 笔顺类型序列 (来自 xlsx, 6 类): {char: [t0,t1,...]}
NO = json.load(open(os.path.join("D:/desktop/学习结果小文件/neuroglyph34_project",
                                  "struct32_aligned.json"), encoding="utf-8"))["nopy26"]
ST = json.loads(STROKE)
STYPES = {}
for ch in ST:
    v = NO.get(ch)
    if v is None:
        continue
    n = int(v[9])
    STYPES[ch] = [int(x) for x in v[16:16+n]]
STYPES_STR = json.dumps(STYPES, ensure_ascii=False)

# 去掉失败的融合头F
if "F" in DATA["embed"]:
    del DATA["embed"]["F"]
if "F" in DATA["meta"]["covered"]:
    del DATA["meta"]["covered"]["F"]
DATA_STR = json.dumps(DATA, ensure_ascii=False)

ref_pairs = [["王","玉"],["干","千"],["大","太"],["日","曰"],["土","士"],
             ["人","入"],["己","已"],["日","月"],["木","水"],["天","夫"],
             ["风","冈"],["上","月"],["后","右"],["末","未"],["田","由"],["由","甲"],["刀","力"]]

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>字形判别小应用 · 两个字有多像</title>
<style>
  :root{
    --bg:#f6f8fa; --card:#ffffff; --ink:#1f2933; --sub:#687586;
    --line:#e3e8ee; --accent:#0e7c86; --accent2:#b4530a; --good:#0e7c86; --bad:#c0392b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;
       line-height:1.6;padding:24px}
  .wrap{max-width:880px;margin:0 auto}
  h1{font-size:22px;margin:0 0 4px}
  .lead{color:var(--sub);font-size:14px;margin:0 0 18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
  .io{display:flex;align-items:center;gap:14px;justify-content:center;margin:6px 0 14px}
  .box{width:96px;height:96px;font-size:54px;text-align:center;border:2px solid var(--line);
       border-radius:12px;outline:none;color:var(--ink);background:#fbfdff;font-family:inherit}
  .box:focus{border-color:var(--accent)}
  .vs{color:var(--sub);font-size:20px;font-weight:700}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:10px;padding:11px 26px;
       font-size:15px;cursor:pointer;font-family:inherit}
  .btn:hover{opacity:.9}
  .presets{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:6px}
  .chip{background:#eef6f7;color:var(--accent);border:1px solid #cfe9ec;border-radius:999px;
        padding:5px 13px;font-size:13px;cursor:pointer}
  .chip:hover{background:#e0eff1}
  .res{margin-top:8px}
  .big{font-size:42px;font-weight:800;text-align:center;margin:6px 0 2px}
  .verdict{text-align:center;font-size:15px;color:var(--sub);margin-bottom:14px}
  .barwrap{margin:10px 0}
  .barrow{display:flex;align-items:center;gap:10px;margin:7px 0}
  .barlabel{width:190px;font-size:13px;color:var(--ink);flex:none}
  .bartrack{flex:1;height:20px;background:#eef1f4;border-radius:6px;overflow:hidden}
  .barfill{height:100%;border-radius:6px;transition:width .35s}
  .barval{width:74px;text-align:right;font-size:13px;font-variant-numeric:tabular-nums;flex:none}
  .note{font-size:13px;color:var(--sub);background:#f3f7f8;border-left:3px solid var(--accent);
        padding:8px 12px;border-radius:6px;margin-top:10px}
  .warn{color:var(--bad);font-weight:600}
  details{margin-top:4px}
  summary{cursor:pointer;color:var(--accent);font-size:14px;font-weight:600}
  .term{font-size:14px;margin:8px 0}
  .term b{color:var(--ink)}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
  th,td{border:1px solid var(--line);padding:6px 8px;text-align:center}
  th{background:#f0f4f6;color:var(--sub);font-weight:600}
  .foot{color:var(--sub);font-size:12px;text-align:center;margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <h1>字形判别小应用</h1>
  <p class="lead">输入任意两个字，看模型认为它们「有多像」。六种「看字方式」并排对比：
  <b>组合(G·推荐)</b>=组合树×矢量×<b>笔画级关系</b>、<b>组合(C)</b>、<b>组合树(T)</b>、<b>矢量(V)</b>、
  <b>结构化(S)</b>、<b>纯像素(R)</b>。G 越低=分得越开（最好）。</p>

  <div class="card">
    <div class="io">
      <input id="a" class="box" maxlength="1" value="末"/>
      <span class="vs">vs</span>
      <input id="b" class="box" maxlength="1" value="未"/>
    </div>
    <div style="text-align:center">
      <button class="btn" onclick="run()">对比这两个字</button>
    </div>
    <div class="presets" id="presets"></div>
  </div>

  <div class="card res" id="result" style="display:none">
    <div class="big" id="bigPct">—</div>
    <div class="verdict" id="verdict">—</div>
    <div class="barwrap" id="bars"></div>
    <div class="note" id="note"></div>
  </div>

  <div class="card">
    <h3 style="margin:0 0 4px;font-size:15px">参考基准（已提前算好）</h3>
    <p style="font-size:13px;color:var(--sub);margin:0 0 6px">
      G/C 越低=分得越开（最好）；纯像素(R)常虚高=把不同的字误认成同一个。点行可填入上方。</p>
    <table id="reftab"><thead><tr><th>字对</th><th>组合(G)</th><th>组合(C)</th><th>树(T)</th><th>矢量(V)</th><th>结构化(S)</th><th>纯像素(R)</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="card">
    <details>
      <summary>看不懂那些术语？点开用大白话解释</summary>
      <div style="margin-top:8px">
        <div class="term"><b>笔画级关系编码</b>：把“哪一笔最长、两笔谁长谁短”这种<b>离散关系</b>显式编码。末/未 只是两横长短顺序相反，普通相似度分不出，关系编码能直接点破。</div>
        <div class="term"><b>组合(G) = 组合树 × 矢量 × 关系</b>：三条路互补——部首/结构像、笔画长短位置像、<b>笔画关系也像</b>，三者都说像才判像。当前最强判别。</div>
        <div class="term"><b>组合(C) = 组合树 × 矢量</b>：两条路互补。任一路说不象就压低。</div>
        <div class="term"><b>组合树(T)</b>：把字按"部首→结构→细节"三层比。部首不同直接判不同字。</div>
        <div class="term"><b>矢量(V)</b>：把字拆成"一笔一笔"，用坐标算每笔长短、方向、位置，按笔序对齐比。能指出"第几笔长短不同"。</div>
        <div class="term"><b>嵌入 / 余弦相似度</b>：把字变成一串数字(64个)，相似的字数字也近。余弦 1=一样，0=无关，负数=反向；本应用 ×100 当「相似度%」。</div>
        <div class="term"><b>近形字 / 硬骨头</b>：长得极像的字，如 王/玉、末/未、日/曰。人易看错，模型易混。</div>
        <div class="term"><b>留出测试 / 捷径(shortcut)</b>：用训练没见过的字对考模型；若只背题、换题就露馅，叫走了捷径（融合头就中招）。</div>
      </div>
    </details>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const DATA = __DATA__;
const TREE = __TREE__;
const STROKE = __STROKE__;
const STYPES = __STYPES__;
const EMB = DATA.embed;
const MODE_LABEL = {G:"组合(树×矢量×关系)", C:"组合(树×矢量)", T:"组合树", V:"矢量笔画", S:"结构化", R:"纯像素"};
const MODE_COLOR = {G:"#0b6e57", C:"#0e7c86", T:"#3a7ca5", V:"#7a4fb5", S:"#3a7ca5", R:"#b4530a"};

function emb(mode, ch){ const v = EMB[mode][ch]; return v ? v : null; }
function cosine(a,b){
  let d=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){ d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-12);
}
function pctOf(mode,a,b){
  const va=emb(mode,a), vb=emb(mode,b);
  if(!va||!vb) return null;
  const s=cosine(va,vb);
  return Math.max(0,Math.round(s*1000)/10);
}
// 组合树规则: 部首不同->0; 部首同结构不同->0.5*relu(det); 都相同->relu(det)
function treeSim(a,b){
  const ta=TREE[a], tb=TREE[b];
  if(!ta||!tb) return null;
  if(ta.rad!==tb.rad) return 0;
  const det=Math.max(0, cosine(ta.det, tb.det));
  return (ta.str!==tb.str) ? 0.5*det*100 : det*100;
}
// 矢量笔画: 按笔序(上->下)对齐, 重权长度排位
function vSim(a,b){
  const fa=STROKE[a], fb=STROKE[b];
  if(!fa||!fb) return null;
  const nA=fa.length, nB=fb.length;
  const base = 1 - Math.abs(nA-nB)/Math.max(nA,nB);
  const m=Math.min(nA,nB);
  if(m===0) return 0;
  let gs=[];
  for(let i=0;i<m;i++){
    const sa=fa[i], sb=fb[i];
    const lsim = 1 - Math.abs(sa.fl - sb.fl);
    const rsim = 1 - Math.abs(sa.rank - sb.rank)/Math.max(nA,nB);
    const dsim = (sa.sin*sb.sin + sa.cos*sb.cos + 1)/2;
    const pd = Math.hypot(sa.x-sb.x, sa.y-sb.y);
    const psim = Math.max(0, 1 - pd/0.4);
    gs.push(0.28*lsim + 0.50*rsim + 0.12*dsim + 0.10*psim);
  }
  return base * (gs.reduce((x,y)=>x+y,0)/m) * 100;
}
// 组合 = 树 × 矢量
function cSim(a,b){
  const t=treeSim(a,b), v=vSim(a,b);
  if(t===null||v===null) return null;
  return t*v/100;
}
// 笔画级关系编码: 笔画数 + 笔顺类型序列 + 按笔序对齐的长度轮廓(乘积)
function relSim(a,b){
  const fa=STROKE[a], fb=STROKE[b];
  const ta=STYPES[a], tb=STYPES[b];
  if(!fa||!fb||!ta||!tb) return null;
  const nA=fa.length, nB=fb.length;
  const cntF = 1 - 0.6*Math.abs(nA-nB)/Math.max(nA,nB,1);
  const m=Math.min(nA,nB);
  if(m===0) return 0;
  let tm=0; for(let i=0;i<m;i++) if(ta[i]===tb[i]) tm++;
  const typeF = 0.3 + 0.7*(tm/m);
  let maxL=0; for(let i=0;i<m;i++){ maxL=Math.max(maxL, fa[i].L, fb[i].L); }
  maxL += 1e-9;
  let prof=1;
  for(let i=0;i<m;i++){ prof*=Math.max(0, 1 - Math.abs(fa[i].L - fb[i].L)/maxL); }
  return cntF*(0.30*typeF + 0.70*prof)*100;
}
// 组合关系 = 组合(C) × 关系(R)
function gSim(a,b){
  const c=cSim(a,b), r=relSim(a,b);
  if(c===null||r===null) return null;
  return c*r/100;
}
// 矢量诊断: 差异最大的笔画
function vDiag(a,b){
  const fa=STROKE[a], fb=STROKE[b];
  if(!fa||!fb) return null;
  const m=Math.min(fa.length, fb.length);
  let best=null;
  for(let i=0;i<m;i++){
    const d=Math.abs(fa[i].L - fb[i].L) + 0.05*Math.abs(fa[i].rank - fb[i].rank);
    if(!best||d>best.d) best={d, i, sa:fa[i], sb:fb[i]};
  }
  return best;
}
// 关系诊断: 哪两笔的长短顺序相反
function rDiag(a,b){
  const fa=STROKE[a], fb=STROKE[b];
  const n=Math.min(fa.length, fb.length);
  let best=null;
  for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){
    const sa=Math.sign(fa[i].L-fa[j].L), sb=Math.sign(fb[i].L-fb[j].L);
    if(sa!==sb){
      const d=Math.abs(fa[i].L-fb[i].L)+Math.abs(fa[j].L-fb[j].L);
      if(!best||d>best.d) best={d,i,j};
    }
  }
  return best;
}
function verdictOf(p){
  if(p>=80) return "模型认为：极像（疑似异体/微调）";
  if(p>=50) return "模型认为：很像";
  if(p>=25) return "模型认为：有些像";
  if(p>=8)  return "模型认为：基本不像（已分得较开）";
  return "模型认为：几乎无关";
}
function colorOf(p){ return p>=70?"var(--bad)":(p>=25?"var(--accent2)":"var(--good)"); }
function run(){
  const a=document.getElementById("a").value.trim();
  const b=document.getElementById("b").value.trim();
  const res=document.getElementById("result");
  const note=document.getElementById("note");
  if(!a||!b){ alert("请输入两个字"); return; }
  if(a===b){ res.style.display="block";
    document.getElementById("bigPct").textContent="100%";
    document.getElementById("bigPct").style.color="var(--accent)";
    document.getElementById("verdict").textContent="这是同一个字";
    document.getElementById("bars").innerHTML="";
    note.innerHTML="两个输入是同一个字，相似度自然 100%。"; return; }
  let miss=[];
  if(!(TREE[a]&&TREE[b])) miss.push("组合树/组合/G");
  if(!(EMB.S[a]&&EMB.S[b])) miss.push("结构化");
  if(!(EMB.R[a]&&EMB.R[b])) miss.push("纯像素");
  if(!(STROKE[a]&&STROKE[b])) miss.push("矢量笔画/关系");
  if(miss.length){
    res.style.display="block";
    document.getElementById("bigPct").textContent="—";
    document.getElementById("bigPct").style.color="var(--ink)";
    document.getElementById("verdict").innerHTML='<span class="warn">字库未收录：'+miss.join("、")+' 模式下该字无数据</span>';
    document.getElementById("bars").innerHTML="";
    note.innerHTML="字库约 6622 字（组合树 "+Object.keys(TREE).length+" 字、矢量笔画 "+Object.keys(STROKE).length+" 字）。试试常见 GB2312 汉字。";
    return;
  }
  const rG=gSim(a,b), rC=cSim(a,b), rT=treeSim(a,b), rV=vSim(a,b), rS=pctOf("S",a,b), rR=pctOf("R",a,b);
  res.style.display="block";
  document.getElementById("bigPct").textContent=rG.toFixed(1)+"%";
  document.getElementById("bigPct").style.color=colorOf(rG);
  document.getElementById("verdict").textContent=verdictOf(rG)+"（组合G=树×矢量×关系）";
  const rows=[["G",rG],["C",rC],["T",rT],["V",rV],["S",rS],["R",rR]];
  let html="";
  rows.forEach(([m,r])=>{
    const p=(r===null)?0:r;
    const c=MODE_COLOR[m];
    const strong = (m==="G") ? "font-weight:700" : "";
    html+='<div class="barrow"><div class="barlabel" style="'+strong+'">'+MODE_LABEL[m]+'</div>'
      +'<div class="bartrack"><div class="barfill" style="width:'+p+'%;background:'+c+'"></div></div>'
      +'<div class="barval">'+(r===null?"—":r.toFixed(1)+"%")+'</div></div>';
  });
  document.getElementById("bars").innerHTML=html;
  // 解读
  let msg="";
  const dg = vDiag(a,b);
  const diagTxt = (dg && dg.d>0.05) ? "　矢量诊断：第"+(dg.i+1)+"笔（从上数）长 "+dg.sa.L.toFixed(3)+"（排"+dg.sa.rank+"） vs "+dg.sb.L.toFixed(3)+"（排"+dg.sb.rank+"）。" : "";
  const rd = rDiag(a,b);
  const relTxt = (rd) ? "　关系诊断：第"+(rd.i+1)+"笔与第"+(rd.j+1)+"笔的<b>长短顺序相反</b>（末式前者长、未式后者长）。" : "";
  if(rG<8 && rR>50){
    msg="<b>组合路把字分开了</b>：纯像素给 "+rR.toFixed(1)+"%（把不同的字当成同一个），组合(G)压到 "+rG.toFixed(1)+"%。"+diagTxt;
  } else if(rG<8 && rS>50){
    msg="<b>组合路修好了过度合并</b>：结构化给 "+rS.toFixed(1)+"% 还糊着，组合(G)用「部首/笔画关系」判为不同（"+rG.toFixed(1)+"%）。"+diagTxt;
  } else if(rG>=50){
    msg="<b>诚实的缺口</b>：组合(G)也只给 "+rG.toFixed(1)+"%。这俩字部首、结构、笔顺、笔画长度排位都高度重合，是真正的硬骨头。"+diagTxt+relTxt;
  } else {
    msg="组合(G) "+rG.toFixed(1)+"% / 组合(C) "+rC.toFixed(1)+"% / 树 "+rT.toFixed(1)+"% / 矢量 "+rV.toFixed(1)+"% / 结构化 "+rS.toFixed(1)+"% / 纯像素 "+rR.toFixed(1)+"%。"+diagTxt+relTxt;
  }
  note.innerHTML=msg;
}

const PRESETS=[["末","未"],["田","由"],["由","甲"],["风","冈"],["上","月"],["日","曰"],
               ["后","右"],["王","玉"],["干","千"],["大","太"],["土","士"],["刀","力"],["天","夫"],["风","凤"]];
const pc=document.getElementById("presets");
PRESETS.forEach(([a,b])=>{
  const s=document.createElement("span"); s.className="chip"; s.textContent=a+" / "+b;
  s.onclick=()=>{document.getElementById("a").value=a;document.getElementById("b").value=b;run();};
  pc.appendChild(s);
});
const REF=__REF__;
const tb=document.querySelector("#reftab tbody");
REF.forEach(([a,b])=>{
  const rG=gSim(a,b), rC=cSim(a,b), rT=treeSim(a,b), rV=vSim(a,b), rS=pctOf("S",a,b), rR=pctOf("R",a,b);
  const tr=document.createElement("tr");
  tr.style.cursor="pointer";
  tr.onclick=()=>{document.getElementById("a").value=a;document.getElementById("b").value=b;run();};
  tr.innerHTML="<td>"+a+" / "+b+"</td>"
    +"<td>"+(rG!==null?rG.toFixed(1)+"%":"—")+"</td>"
    +"<td>"+(rC!==null?rC.toFixed(1)+"%":"—")+"</td>"
    +"<td>"+(rT!==null?rT.toFixed(1)+"%":"—")+"</td>"
    +"<td>"+(rV!==null?rV.toFixed(1)+"%":"—")+"</td>"
    +"<td>"+(rS!==null?rS.toFixed(1)+"%":"—")+"</td>"
    +"<td>"+(rR!==null?rR.toFixed(1)+"%":"—")+"</td>";
  tb.appendChild(tr);
});
document.getElementById("foot").textContent="组合树+矢量+笔画级关系来自已解析的字形表示 · 字库 "+DATA.meta.vocab+" 字 · "+DATA.meta.version;
run();
</script>
</body>
</html>
"""

html = (HTML.replace("__DATA__", DATA_STR)
            .replace("__TREE__", TREE)
            .replace("__STROKE__", STROKE)
            .replace("__STYPES__", STYPES_STR)
            .replace("__REF__", json.dumps(ref_pairs, ensure_ascii=False)))
outp = os.path.join(BASE, "glyph_discriminator.html")
open(outp, "w", encoding="utf-8").write(html)
print("written", outp, round(os.path.getsize(outp)/1e6, 2), "MB")
