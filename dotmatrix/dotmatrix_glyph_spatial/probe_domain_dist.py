# 域归属探针：统计目标字在 10 个 KG 域的出现分布
# 验证"域归属"能否作为可解读语义轴（猫狗->兽医, 树花->园艺）
import csv, json, os, sys
from collections import defaultdict

KG_DIR = r"D:/desktop/学习结果小文件/neuroglyph34_project/knowledge_graph"
OUT = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial/domain_dist_probe.json"
TARGET = ['猫','狗','鸟','树','牛','花','羊','鱼','江','河','木','草','虫','马']
MAX_ROWS = 400000  # 每域最多扫这么多行（控制大域耗时，同时保证跨域可比）

files = sorted(f for f in os.listdir(KG_DIR) if f.endswith('.csv'))
N = {c: defaultdict(int) for c in TARGET}   # N[char][domain] = 含该字的 token 数
T_tokens = defaultdict(int)                  # 每域总 token 数
T_chars = defaultdict(int)                   # 每域总字符数

for f in files:
    dom = f.replace('_all.csv','')
    path = os.path.join(KG_DIR, f)
    sys.stderr.write(f"scan {f} ...\n"); sys.stderr.flush()
    n = 0
    with open(path, encoding='utf-8', errors='replace') as fh:
        r = csv.reader(fh)
        for row in r:
            if len(row) < 3:
                continue
            head, tail = row[0], row[2]
            T_tokens[dom] += 2
            T_chars[dom] += len(head) + len(tail)
            for c in TARGET:
                if c in head: N[c][dom] += 1
                if c in tail: N[c][dom] += 1
            n += 1
            if n >= MAX_ROWS:
                break

domains = [f.replace('_all.csv','') for f in files]
result = {'targets': TARGET, 'domains': domains,
          'N': {c: dict(N[c]) for c in TARGET},
          'T_tokens': dict(T_tokens), 'T_chars': dict(T_chars)}

# ---- 分析 ----
print("\n=== 每字 域归属（按 P(域|字) 与 领域特异性 spec） ===")
for c in TARGET:
    row = N[c]
    tot = sum(row.values())
    if tot == 0:
        print(f"  {c}: 全域无出现"); continue
    # P(域|字)
    p = {d: row.get(d,0)/tot for d in domains}
    # 领域特异性 spec = 该字在某域的浓度 / 该域整体浓度（tf-idf 思路）
    # spec_cd = (N_cd / T_tokens[dom])  ; 跨域归一为相对倍数
    spec = {}
    for d in domains:
        dom_chars = T_tokens[d]
        if dom_chars > 0:
            # 该字在该域的"字频占比" vs 全局该字占比
            char_in_dom = row.get(d,0)
            spec[d] = (char_in_dom / dom_chars) * (tot / sum(T_tokens.values()))
        else:
            spec[d] = 0.0
    top_p = sorted(p.items(), key=lambda x:-x[1])[:3]
    # spec 用相对全局倍数表达：spec_cd / mean(spec)
    spec_mean = sum(spec.values())/len(spec) if spec else 1
    top_spec = sorted(((d, spec[d]/spec_mean if spec_mean else 0) for d in domains),
                      key=lambda x:-x[1])[:3]
    print(f"  {c}: 总出现 {tot}")
    print(f"      P(域|字) top3: " + "  ".join(f"{d}={v*100:.1f}%" for d,v in top_p))
    print(f"      特异性× top3: " + "  ".join(f"{d}×{m:.1f}" for d,m in top_spec))

json.dump(result, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"\nDONE -> {OUT}")
