# -*- coding: utf-8 -*-
"""
字指针 · 下一个字的概率 —— 离线交互 HTML 生成器
读 trans_index.json, 生成自包含(无CDN)页面:
  - 输入一字 -> 显示 top 下一字概率榜(条形)
  - 放射状指针图: 中心字向各下一字连线, 线宽=概率, 标注 P(下字|本字)
  - 随机基线 1/字符表, 直观看出"学到了下一个字的结构"
"""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
IDX  = json.load(open(os.path.join(BASE, "trans_index.json"), encoding='utf-8'))
OUT  = os.path.join(BASE, "char_pointer_graph.html")

# 字符表大小(用于随机基线)
chars = set(IDX.keys())
for v in IDX.values():
    for nxt, _, _ in v["next"]:
        chars.add(nxt)
VOCAB = len(chars)

# 演示字: 挑常见且出度够大的
DEMO_POOL = list("长牛明江中水大国学木火山金土日月天山风人心王子女时年文书人法市生力小高")
DEMOS = [c for c in DEMO_POOL if c in IDX and IDX[c]["total"] >= 200][:16]
if not DEMOS:
    DEMOS = sorted(IDX.keys(), key=lambda c: IDX[c]["total"], reverse=True)[:16]

data_js = json.dumps(IDX, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>字指针 · 下一个字的概率</title>
<style>
:root{--bg:#f6f8fa;--card:#fff;--ink:#1f2933;--sub:#687586;--line:#e3e8ee;--accent:#0e7c86;--accent2:#b4530a;--good:#0b6e57}
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
.btn:hover{opacity:.9}
.presets{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:6px}
.chip{background:#eef6f7;color:var(--accent);border:1px solid #cfe9ec;border-radius:999px;padding:5px 13px;font-size:14px;cursor:pointer}
.chip:hover{background:#e0eff1}
.res{margin-top:8px}
.big{font-size:15px;color:var(--sub);text-align:center;margin:4px 0 10px}
.barrow{display:flex;align-items:center;gap:10px;margin:6px 0}
.barlabel{width:60px;font-size:22px;text-align:center;flex:none;font-weight:700}
.bartrack{flex:1;height:22px;background:#eef1f4;border-radius:6px;overflow:hidden;position:relative}
.barfill{height:100%;border-radius:6px;background:linear-gradient(90deg,#0e7c86,#0b6e57);transition:width .35s}
.barval{width:120px;text-align:right;font-size:13px;font-variant-numeric:tabular-nums;flex:none;color:var(--sub)}
.note{font-size:13px;color:var(--sub);background:#f3f7f8;border-left:3px solid var(--accent);padding:8px 12px;border-radius:6px;margin-top:12px}
.warn{color:#c0392b;font-weight:600}
.foot{color:var(--sub);font-size:12px;text-align:center;margin-top:6px}
svg{display:block;margin:0 auto}
.tlabel{font-size:13px;fill:var(--ink)}
</style></head>
<body><div class="wrap">
<h1>字指针 · 下一个字的概率</h1>
<p class="lead">把每个汉字当成<b>一个节点</b>，它身上的「指针」指向<b>紧跟它的下一个字</b>，指针权重 = <b>P(下字 | 本字)</b>。
这是<b>字符级马尔可夫转移概率</b>，从本地 10 域知识图谱数有序字对得到，<b>零训练、离线</b>。所有指针汇起来就是一张有向加权图（你说的"图谱/学习下一个字"）。</p>

<div class="card">
  <div class="io">
    <input id="a" class="box" maxlength="1" value="长"/>
    <button class="btn" onclick="run()">下一个字是谁？</button>
  </div>
  <div class="presets" id="presets"></div>
</div>

<div class="card res" id="result" style="display:none">
  <div class="big" id="headline"></div>
  <div class="barrow" style="font-weight:700;color:var(--sub);font-size:12px"><div class="barlabel">字</div><div class="bartrack"></div><div class="barval">P(下字|本字)</div></div>
  <div id="bars"></div>
  <div id="graph"></div>
  <div class="note" id="note"></div>
</div>

<div class="card">
  <details><summary>看不懂「指针 / 马尔可夫」？点开用大白话解释</summary>
  <div style="margin-top:8px;font-size:14px">
    <div style="margin:8px 0"><b>指针</b>：编程里指针不存内容、只存"指向哪"的地址。这里每个字就是个节点，它身上的指针指向"下一个字"——像链表里每个节点指向下一个节点。</div>
    <div style="margin:8px 0"><b>P(下字|本字)</b>：给定本字，下一个字是某个字的概率。比如「长」后面最常见是「江」(长江)，所以 P(江|长) 高。这是<b>条件概率</b>，从资料里数"长后面跟江出现多少次 / 长后面总共跟过多少字"算出来。</div>
    <div style="margin:8px 0"><b>为什么不受齐夫律影响</b>：之前单字→学科域分布被"语料体量"搞崩，是因为绝对频率受齐夫律支配。但<b>条件转移</b> P(江|长) 只关心"长后面"，长本身多频繁不重要——所以这条路稳。</div>
    <div style="margin:8px 0"><b>图谱</b>：所有字的指针连起来 = 一张有向加权图。中心字向各下一字连线，线越粗=概率越大。</div>
  </div>
</details>
</div>

<div class="foot" id="foot"></div>
</div>

<script>
const TRANS = __DATA__;
const META = {vocab: __VOCAB__, demos: __DEMOS__};
function run(){
  const c = document.getElementById("a").value.trim();
  const res = document.getElementById("result");
  const note = document.getElementById("note");
  const headline = document.getElementById("headline");
  if(!c){ alert("请输入一个字"); return; }
  const info = TRANS[c];
  if(!info){ res.style.display="block"; headline.textContent=""; document.getElementById("bars").innerHTML="";
    note.innerHTML='<span class="warn">字库未收录足够有序共现</span>：该字在资料里作为"本字"出现不足，无法估计下一字。试试常见字（长/牛/明/江/中…）。'; drawGraph(null); return; }
  res.style.display="block";
  const base = (1/META.vocab*100).toFixed(3);
  const top = info.next;
  const top1 = top[0];
  headline.innerHTML = '「'+c+'」后面，最可能的下一个字是 <b style="font-size:22px;color:var(--accent)">'+top1[0]+'</b> （'+ (top1[1]*100).toFixed(1) +'%），随机瞎猜仅 '+base+'%。';
  // 条形榜
  let html="";
  top.forEach(([nc, p, cnt])=>{
    const pc=(p*100);
    html+='<div class="barrow"><div class="barlabel">'+nc+'</div>'
      +'<div class="bartrack"><div class="barfill" style="width:'+Math.max(2,pc)+'%"></div></div>'
      +'<div class="barval">'+pc.toFixed(1)+'%  ·  '+cnt.toLocaleString()+'次</div></div>';
  });
  document.getElementById("bars").innerHTML=html;
  drawGraph(c, top);
  note.innerHTML='共从资料里数到「'+c+'」作本字 <b>'+info.total.toLocaleString()+'</b> 次；展示 top '+top.length+' 下一字。'
    +'指针权重即条件概率，越粗越可能是"下一个字"。这正是不训练、纯从资料"自带学习"出的语言结构。';
}
function drawGraph(c, top){
  const el=document.getElementById("graph");
  if(!c||!top){ el.innerHTML=""; return; }
  const W=560,H=380,cx=W/2,cy=H/2,R=140;
  const n=top.length;
  let s='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
  // 连线
  for(let i=0;i<n;i++){
    const ang=-Math.PI/2 + i*2*Math.PI/n;
    const x=cx+R*Math.cos(ang), y=cy+R*Math.sin(ang);
    const w=Math.max(1.2, top[i][1]*60);
    s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+x.toFixed(1)+'" y2="'+y.toFixed(1)+'" stroke="#0e7c86" stroke-width="'+w.toFixed(2)+'" stroke-opacity="0.55"/>';
    s+='<text x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" class="tlabel" text-anchor="middle" dy="4">'+top[i][0]+'</text>';
    s+='<text x="'+((cx+x)/2).toFixed(1)+'" y="'+((cy+y)/2).toFixed(1)+'" font-size="10" fill="#0e7c86" text-anchor="middle" dy="-2">'+(top[i][1]*100).toFixed(1)+'%</text>';
  }
  // 中心
  s+='<circle cx="'+cx+'" cy="'+cy+'" r="30" fill="#0b6e57"/>';
  s+='<text x="'+cx+'" y="'+cy+'" class="tlabel" fill="#fff" text-anchor="middle" dy="6" style="font-size:22px;font-weight:700">'+c+'</text>';
  s+='</svg>';
  el.innerHTML=s;
}
// 预设
const pc=document.getElementById("presets");
META.demos.forEach(d=>{
  const s=document.createElement("span"); s.className="chip"; s.textContent=d;
  s.onclick=()=>{document.getElementById("a").value=d; run();};
  pc.appendChild(s);
});
document.getElementById("foot").textContent="字符级马尔可夫转移 · 字表 "+META.vocab.toLocaleString()+" 字 · 从 10 域知识图谱有序字对计数(零训练)";
run();
</script></body></html>
"""

html = html.replace("__DATA__", data_js).replace("__VOCAB__", str(VOCAB)).replace("__DEMOS__", json.dumps(DEMOS, ensure_ascii=False))
open(OUT, 'w', encoding='utf-8').write(html)
print(f"[done] {OUT}  ({os.path.getsize(OUT)//1024:,} KB, 字表 {VOCAB:,} 字, 演示字 {len(DEMOS)} 个)")
