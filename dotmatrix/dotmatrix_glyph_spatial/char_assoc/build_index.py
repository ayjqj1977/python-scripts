# -*- coding: utf-8 -*-
"""
字关联索引构建器（本地免训练原型）
=================================
扫描 10 域中文知识图谱三元组 CSV(Head|Relation|Tail|...|Subdomain)，
对演示字(牛/江/明)抽取：
  (a) 单字共现邻域 cooc : 与该字出现在「同一个词/短语」里的其他单字及计数 -> 义边(语义邻字)
  (b) 跨三元组关联 rel  : 该字所在短语 --关系--> 另一短语里的单字 -> 关系型语义关联
  (c) 含该字的样例 triple 短语 (domain,head,rel,tail)
输出 assoc_index.json。全程流式、不训练、纯本地。
"""
import os, re, csv, json, time

KG_DIR = r'D:/desktop/学习结果小文件/neuroglyph34_project/knowledge_graph'
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assoc_index.json')
TARGETS = ['牛', '江', '明']
CJK = re.compile(r'[一-鿿]')
COOC_KEEP = 250      # 每个字保留的共现邻字上限
PHRASE_KEEP = 160    # 每个字保留的样例 triple 上限

def singles(s):
    return CJK.findall(s)

def main():
    files = [os.path.join(KG_DIR, f) for f in os.listdir(KG_DIR) if f.endswith('.csv')]
    files.sort()
    data = {c: {'match_count': 0, 'cooc': {}, 'rel': {}, 'phrases': []} for c in TARGETS}
    total_rows = 0
    t0 = time.time()
    for fi, fpath in enumerate(files):
        dom = os.path.basename(fpath).replace('_all.csv', '')
        with open(fpath, encoding='utf-8', errors='replace') as fh:
            r = csv.reader(fh)
            header = next(r, None)
            for row in r:
                total_rows += 1
                if len(row) < 3:
                    continue
                head, rel, tail = row[0], row[1], row[2]
                # 快速子串预筛，只对命中的行做正则
                hit = [c for c in TARGETS if c in head or c in tail]
                if not hit:
                    continue
                for c in hit:
                    d = data[c]
                    d['match_count'] += 1
                    # (a) 同短语内单字共现
                    if c in head:
                        for o in singles(head):
                            if o != c:
                                d['cooc'][o] = d['cooc'].get(o, 0) + 1
                    if c in tail:
                        for o in singles(tail):
                            if o != c:
                                d['cooc'][o] = d['cooc'].get(o, 0) + 1
                    # (b) 跨三元组：c 所在短语 --rel--> 另一短语里的单字
                    if c in head:
                        for o in singles(tail):
                            if o != c:
                                d['rel'][o] = d['rel'].get(o, 0) + 1
                    if c in tail:
                        for o in singles(head):
                            if o != c:
                                d['rel'][o] = d['rel'].get(o, 0) + 1
                    # (c) 样例短语
                    if len(d['phrases']) < PHRASE_KEEP:
                        d['phrases'].append({'domain': dom, 'head': head, 'rel': rel, 'tail': tail})
        print(f'  [{fi+1}/{len(files)}] {dom}: rows={total_rows:,}', flush=True)

    # 修剪共现表：只保留计数靠前的邻字
    for c in TARGETS:
        d = data[c]
        d['cooc'] = dict(sorted(d['cooc'].items(), key=lambda kv: -kv[1])[:COOC_KEEP])
        d['rel']  = dict(sorted(d['rel'].items(),  key=lambda kv: -kv[1])[:COOC_KEEP])
    out = {
        'meta': {'kg_dir': KG_DIR, 'targets': TARGETS,
                 'total_rows_scanned': total_rows,
                 'elapsed_sec': round(time.time()-t0, 1),
                 'note': 'cooc=同短语单字共现(义边); rel=跨三元组关系关联; phrases=样例triple'},
        'chars': data,
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'已写出 {OUT}')
    for c in TARGETS:
        d = data[c]
        top = list(d['cooc'].items())[:12]
        print(f'  {c}: 命中 {d["match_count"]:,} 行 | 共现邻字top: {top}')

if __name__ == '__main__':
    main()
