# -*- coding: utf-8 -*-
"""
字关联图谱 · 本地免训练交互原型生成器
====================================
读 assoc_index.json(语义共现,来自10域KG) + struct32_aligned.json(声母),
生成离线自包含 HTML: 中心字 + 语义邻字(义边) + 同音邻字(音边) + 跨三元组(关边)。
纯前端 vanilla JS 力导向, 无 CDN, 双击即用。
"""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
KG_PARENT = r'D:/desktop/学习结果小文件/neuroglyph34_project'
IDX = os.path.join(BASE, 'assoc_index.json')
STRUCT = os.path.join(KG_PARENT, 'struct32_aligned.json')
OUT = os.path.join(BASE, 'char_assoc_graph.html')
K = 18  # 每个字保留的共现邻字数

def sheng(c, all32):
    try:
        return int(all32[c][26])
    except Exception:
        return -1

def main():
    idx = json.load(open(IDX, encoding='utf-8'))
    all32 = json.load(open(STRUCT, encoding='utf-8'))['all32']
    DATA = {}
    for c in idx['meta']['targets']:
        ch = idx['chars'][c]
        cooc = list(ch['cooc'].items())
        rel = list(ch['rel'].items())
        neigh, seen = [], set()
        for o, cnt in cooc[:K]:
            neigh.append((o, cnt, '义')); seen.add(o)
        for o, cnt in rel[:8]:
            if o not in seen and len(neigh) < K + 6:
                neigh.append((o, cnt, '关')); seen.add(o)
        sc = sheng(c, all32)
        nodes = [{'id': 0, 'label': c, 'r': 26, 'center': True}]
        edges = []
        for i, (o, cnt, typ) in enumerate(neigh, start=1):
            nodes.append({'id': i, 'label': o, 'r': 8 + min(16, (cnt ** 0.5) * 2.2)})
            edges.append({'si': 0, 'ti': i, 'type': typ, 'w': cnt})
            if sheng(o, all32) == sc and sc >= 0:
                edges.append({'si': 0, 'ti': i, 'type': '音', 'w': cnt})
        DATA[c] = {'center': c, 'nodes': nodes, 'edges': edges,
                   'phrases': ch.get('phrases', [])[:14]}

    html = TEMPLATE.replace('/*DATA*/', json.dumps(DATA, ensure_ascii=False))
    open(OUT, 'w', encoding='utf-8').write(html)
    print('已生成', OUT)
    for c in DATA:
        print(f"  {c}: {len(DATA[c]['nodes'])-1} 个邻字, {len(DATA[c]['edges'])} 条边")

TEMPLATE = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>字关联图谱 · 本地免训练原型</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
   background:#0f1115;color:#e8e8ea}
 #bar{display:flex;gap:8px;align-items:center;padding:12px 16px;flex-wrap:wrap;
   border-bottom:1px solid #23262e}
 #bar h1{font-size:15px;font-weight:500;margin:0 12px 0 0;color:#fff}
 .btn{background:#1b1f27;color:#cfd2da;border:1px solid #2c313c;border-radius:8px;
   padding:6px 14px;font-size:14px;cursor:pointer}
 .btn.on{background:#2d6cdf;color:#fff;border-color:#2d6cdf}
 .legend{display:flex;gap:14px;margin-left:auto;font-size:12px;color:#9aa0aa}
 .lg{display:flex;align-items:center;gap:5px}
 .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
 #wrap{display:flex;height:calc(100vh - 56px)}
 #stage{flex:1;position:relative;overflow:hidden}
 svg{width:100%;height:100%;display:block}
 #side{width:320px;border-left:1px solid #23262e;padding:14px;overflow:auto;font-size:13px}
 #side h2{font-size:13px;font-weight:500;color:#fff;margin:4px 0 10px}
 .as{margin:0 0 12px;padding:8px 10px;background:#161a21;border:1px solid #23262e;border-radius:8px}
 .as b{color:#fff}
 .as .rel{color:#8b9099;font-size:12px}
 .tri{font-size:12px;color:#b9bec7;margin:2px 0;line-height:1.5}
 .tri .h{color:#e8e8ea}.tri .r{color:#7f8a99}.tri .t{color:#9fc1ff}
 .tip{position:absolute;pointer-events:none;background:#000d;color:#fff;padding:4px 8px;
   border-radius:6px;font-size:12px;display:none;z-index:5}
 text{fill:#e8e8ea;font-size:12px;pointer-events:none;text-anchor:middle}
 text.center{fill:#fff;font-size:16px;font-weight:600}
</style></head>
<body>
<div id="bar">
  <h1>字关联图谱</h1>
  <span id="btns"></span>
  <span class="legend">
    <span class="lg"><span class="sw" style="background:#e0803a"></span>义·KG共现</span>
    <span class="lg"><span class="sw" style="background:#3a8de0"></span>音·同声母</span>
    <span class="lg"><span class="sw" style="background:#2fae7a"></span>关·跨三元组</span>
  </span>
</div>
<div id="wrap">
  <div id="stage"><div class="tip" id="tip"></div></div>
  <div id="side"></div>
</div>
<script>
const DATA = /*DATA*/;
const COL={义:'#e0803a',音:'#3a8de0',关:'#2fae7a'};
const stage=document.getElementById('stage'), svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
stage.insertBefore(svg,stage.firstChild);
const tip=document.getElementById('tip');
let cur=null, raf=null;

function render(key){
  if(raf) cancelAnimationFrame(raf);
  cur=DATA[key];
  const W=stage.clientWidth, H=stage.clientHeight;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const cx=W/2, cy=H/2;
  const ns='http://www.w3.org/2000/svg';
  const nodes=cur.nodes.map((n,i)=>({...n,x:cx+(Math.random()-.5)*220,y:cy+(Math.random()-.5)*220,vx:0,vy:0}));
  const edges=cur.edges.map(e=>({...e}));
  const linkEls=edges.map(e=>{const l=document.createElementNS(ns,'line');
    l.setAttribute('stroke',COL[e.type]); l.setAttribute('stroke-opacity','0.55');
    l.setAttribute('stroke-width', e.type==='关'?1: Math.max(1,Math.min(4,e.w/40)));
    svg.appendChild(l); return l;});
  const nodeEls=nodes.map(n=>{const g=document.createElementNS(ns,'g');
    const c=document.createElementNS(ns,'circle');
    c.setAttribute('r',n.r); c.setAttribute('fill', n.center?'#2d6cdf':'#39414f');
    c.setAttribute('stroke', n.center?'#9cc0ff':'#5a6373'); c.setAttribute('stroke-width', n.center?2:1);
    const t=document.createElementNS(ns,'text'); t.textContent=n.label;
    if(n.center) t.setAttribute('class','center');
    t.setAttribute('dy', n.r+12);
    g.appendChild(c); g.appendChild(t); svg.appendChild(g);
    g.addEventListener('mousemove',ev=>{tip.style.display='block';
      tip.style.left=(ev.offsetX+10)+'px'; tip.style.top=(ev.offsetY+10)+'px';
      tip.textContent= n.center?('中心字 · '+n.label):(n.label+' · 共现 '+n.w)});
    g.addEventListener('mouseleave',()=>tip.style.display='none');
    return {g,c,t};});

  let alpha=1;
  function tick(){
    for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      let dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y, d2=dx*dx+dy*dy+0.01;
      let f=2600/d2, d=Math.sqrt(d2), fx=f*dx/d, fy=f*dy/d;
      nodes[i].vx+=fx; nodes[i].vy+=fy; nodes[j].vx-=fx; nodes[j].vy-=fy;
    }
    edges.forEach(e=>{let a=nodes[e.si],b=nodes[e.ti];
      let dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01;
      let f=(d-86)*0.03*alpha, fx=f*dx/d, fy=f*dy/d;
      a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;});
    nodes.forEach(n=>{n.vx+=(cx-n.x)*0.006; n.vy+=(cy-n.y)*0.006;
      n.vx*=0.86; n.vy*=0.86;
      n.vx=Math.max(-12,Math.min(12,n.vx)); n.vy=Math.max(-12,Math.min(12,n.vy));
      n.x+=n.vx; n.y+=n.vy;});
    nodeEls.forEach((ne,i)=>{ne.g.setAttribute('transform',`translate(${nodes[i].x},${nodes[i].y})`);});
    edges.forEach((e,i)=>{const a=nodes[e.si],b=nodes[e.ti];
      linkEls[i].setAttribute('x1',a.x);linkEls[i].setAttribute('y1',a.y);
      linkEls[i].setAttribute('x2',b.x);linkEls[i].setAttribute('y2',b.y);});
    alpha*=0.992;
    if(alpha>0.03) raf=requestAnimationFrame(tick);
  }
  tick();
  // 侧栏
  const side=document.getElementById('side'); side.innerHTML='';
  const h=document.createElement('h2'); h.textContent='「'+key+'」的关联（来自本地10域知识图谱）'; side.appendChild(h);
  const top=[...cur.nodes].slice(1).sort((a,b)=>b.w-a.w);
  const seen=new Set();
  top.forEach(n=>{ if(seen.has(n.label))return; seen.add(n.label);
    const d=document.createElement('div'); d.className='as';
    d.innerHTML=`<b>${n.label}</b> <span class="rel">共现 ${n.w}</span>`; side.appendChild(d);});
  const ph=document.createElement('h2'); ph.textContent='资料里的样例短语'; ph.style.marginTop='14px'; side.appendChild(ph);
  cur.phrases.slice(0,12).forEach(p=>{const d=document.createElement('div'); d.className='tri';
    d.innerHTML=`<span class="h">${p.head}</span> <span class="r">${p.rel}</span> <span class="t">${p.tail}</span> <span class="rel">[${p.domain}]</span>`;
    side.appendChild(d);});
}

const btns=document.getElementById('btns');
Object.keys(DATA).forEach((k,i)=>{const b=document.createElement('button');
  b.className='btn'+(i===0?' on':''); b.textContent=k;
  b.onclick=()=>{document.querySelectorAll('.btn').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); render(k);}; btns.appendChild(b);});
render(Object.keys(DATA)[0]);
window.addEventListener('resize',()=>render(document.querySelector('.btn.on').textContent));
</script>
</body></html>'''

if __name__ == '__main__':
    main()
