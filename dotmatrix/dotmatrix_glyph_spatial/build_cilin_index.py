# -*- coding: utf-8 -*-
"""
构建单字 -> 同义词词林(扩展版) 索引
=====================================
输出 cilin_index.json，供对比网页使用：
  - da_lei: 大类字母 -> 中文名
  - char_codes: {字: {codes:[...], src:"exact"/"compound"/"none"}}
  - branch_n: {commonStr: 该层分支数 n}  (田久乐公式预计算)
  - code_words: {code: [同义词群]}  (仅保留被用到的 code，供可解读展示)
"""
import json, os, re
from collections import defaultdict

BASE = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
CILIN = os.path.join(BASE, "cilin_extended.txt")
KGIDX = os.path.join(BASE, "kg_full_index.json")
OUT = os.path.join(BASE, "cilin_index.json")

# 词林 13 大类中文名
DA_LEI = {
    "A": "人", "B": "物", "C": "时间·空间", "D": "抽象事物", "E": "特征",
    "F": "动作", "G": "心理活动", "H": "活动", "I": "社会活动", "J": "状态",
    "K": "关系", "L": "助语", "M": "敬语",
}

# 1) 解析词林
code_words = {}     # code -> [words]
word_codes = {}     # word -> [codes]
all_codes = []
with open(CILIN, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^([A-Za-z0-9]+)([=#@]?)\s+(.*)$", line)
        if not m:
            continue
        code, rel, words = m.group(1), m.group(2) or "=", m.group(3)
        code += rel
        ws = words.split()
        if not ws:
            continue
        code_words[code] = ws
        all_codes.append(code)
        for w in ws:
            word_codes.setdefault(w, []).append(code)

# 2) 预计算 branch_n：对每个 code 的 4 个目标前缀长度，统计该层不同分支数
#    Layer 结构: [c0, c1, c2:4, c4, c5:7, c7]
#    LAYER_TABLE: len1->layer1(中类), len2->layer2(小类), len4->layer3(词群), len5->layer4(原子词群)
LAYER_TABLE = {1: 1, 2: 2, 4: 3, 5: 4}
def new_layer(c):
    return [c[0], c[1], c[2:4], c[4], c[5:7], c[7]]

branch_vals = defaultdict(set)   # commonStr -> set(layerVal)
for code in all_codes:
    L = new_layer(code)
    for Llen, layerIdx in [(1, 1), (2, 2), (4, 3), (5, 4)]:
        p = code[:Llen]
        branch_vals[p].add(L[layerIdx])
branch_n = {p: len(s) for p, s in branch_vals.items()}

# 3) 单字映射（6622 字表）
with open(KGIDX, encoding="utf-8") as f:
    kg = json.load(f)
chars = kg["targets"]

char_codes = {}
used_codes = set()
for ch in chars:
    if ch in word_codes:
        codes = word_codes[ch]
        src = "exact"
    else:
        # 复合词聚合：出现在任何词中的字，收集这些词的 code
        agg = set()
        for code, ws in code_words.items():
            for w in ws:
                if ch in w:
                    agg.add(code)
                    break
        if agg:
            codes = sorted(agg)
            src = "compound"
        else:
            codes = []
            src = "none"
    char_codes[ch] = {"codes": codes, "src": src}
    for c in codes:
        used_codes.add(c)

# 仅保留被用到的 code 的同义词群，控制体积
code_words_out = {c: code_words[c] for c in used_codes if c in code_words}

result = {
    "da_lei": DA_LEI,
    "char_codes": char_codes,
    "branch_n": branch_n,
    "code_words": code_words_out,
}
json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

n_exact = sum(1 for v in char_codes.values() if v["src"] == "exact")
n_comp = sum(1 for v in char_codes.values() if v["src"] == "compound")
n_none = sum(1 for v in char_codes.values() if v["src"] == "none")
print(f"[OK] cilin_index.json")
print(f"  字表={len(chars)}  exact={n_exact}({100*n_exact/len(chars):.1f}%)  compound={n_comp}({100*n_comp/len(chars):.1f}%)  none={n_none}({100*n_none/len(chars):.1f}%)")
print(f"  branch_n 条目={len(branch_n)}  用到的 code_words={len(code_words_out)}")
print(f"  文件大小={os.path.getsize(OUT)//1024} KB")
