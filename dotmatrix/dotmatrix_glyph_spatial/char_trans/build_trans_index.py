# -*- coding: utf-8 -*-
"""
字符级「指针」索引构建器 (本地免训练)
====================================
从 10 域中文知识图谱 CSV 里, 对每条三元组的字段(Head/Relation/Tail)抽 CJK 字符序列,
数**有序相邻字对** (c1 -> c2) 的出现次数 —— 这就是字符级马尔可夫/转移概率的素材。

  P(下字=c2 | 本字=c1) = count(c1->c2) / sum_c count(c1->c)

每个字 = 一个节点, 它身上的「指针」指向紧跟它的下一个字, 指针权重 = 上述条件概率。
纯计数(最大似然估计), 零训练, 完全契合 no-training 约束。

条件转移不受齐夫律影响(与之前单字->域分布被语料体量搞崩不同): P(江|长) 高只因「长江」高频。

输出: trans_index.json = { char: {"total":N, "next":[[nextchar, prob, count], ...top20]} }
只保留 总出度>=MIN_OUT 的源字, 每字存 top-K 下一字。
"""
import os, csv, json, sys

KG_DIR = r"D:/desktop/学习结果小文件/neuroglyph34_project/knowledge_graph"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trans_index.json")

MIN_OUT = 50     # 源字总出度阈值(出现够多才可信)
TOP_K  = 20      # 每字保留 top-K 下一字

def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

def cjk_seq(s):
    return [ch for ch in s if is_cjk(ch)]

def main():
    trans = {}          # trans[c1][c2] = count
    vocab = set()
    files = sorted(f for f in os.listdir(KG_DIR) if f.endswith('.csv'))
    print(f"扫描 {len(files)} 个域 CSV ...", flush=True)
    for fi, fname in enumerate(files, 1):
        path = os.path.join(KG_DIR, fname)
        print(f"  [{fi}/{len(files)}] {fname}", flush=True)
        nrow = 0
        with open(path, encoding='utf-8', errors='replace', newline='') as fh:
            r = csv.reader(fh)
            for row in r:
                nrow += 1
                for field in row:
                    if not field:
                        continue
                    seq = cjk_seq(field)
                    for i in range(len(seq) - 1):
                        c1, c2 = seq[i], seq[i+1]
                        vocab.add(c1); vocab.add(c2)
                        d = trans.get(c1)
                        if d is None:
                            trans[c1] = {c2: 1}
                        else:
                            d[c2] = d.get(c2, 0) + 1
                if nrow % 2000000 == 0:
                    print(f"    已处理 {nrow:,} 行, 唯一有序对 {sum(len(v) for v in trans.values()):,}", flush=True)
        print(f"    {fname} 完成, 累计 {nrow:,} 行", flush=True)

    print(f"\n总唯一源字 {len(trans):,}, 唯一有序对 {sum(len(v) for v in trans.values()):,}, 字符表 {len(vocab):,}", flush=True)

    # 剪枝 + top-K + 概率
    out = {}
    kept = 0
    for c1, d in trans.items():
        total = sum(d.values())
        if total < MIN_OUT:
            continue
        ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        out[c1] = {
            "total": total,
            "next": [[c2, round(cnt/total, 5), cnt] for c2, cnt in ranked]
        }
        kept += 1
    print(f"保留源字 {kept:,} (出度>={MIN_OUT}), 写出 {os.path.getsize(OUT)//1024 if os.path.exists(OUT) else 0}KB 之前...", flush=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f"[done] -> {OUT}  ({os.path.getsize(OUT)//1024:,} KB, 源字 {kept:,})", flush=True)

if __name__ == '__main__':
    main()
