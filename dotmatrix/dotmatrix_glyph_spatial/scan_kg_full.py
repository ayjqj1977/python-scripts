# -*- coding: utf-8 -*-
"""
全字 KG 索引扫描
================
把 10 域知识图谱 CSV 扫成「每个字的 10 域出现计数」，供对比网页对任意字查语义。

优化点：
  - 字表 = 有完整字形数据的字（tree_data ∩ stroke_features ∩ struct32），保证 G2 可算
  - 逐字符查表（字符 ∈ 字表 才计数），而非对每字做 `if c in 行`，避免 O(行×字数) 爆炸
  - 每域最多扫 MAX_ROWS 行（与之前探针一致，控制大域耗时）

输出 kg_full_index.json:
  { targets:[字...], domains:[10域], N:{字:{域:计数}}, T_tokens:{域:总token}, T_chars:{域:总字符} }
"""
import csv, json, os, sys
from collections import defaultdict

BASE = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
KG_DIR = "D:/desktop/学习结果小文件/neuroglyph34_project/knowledge_graph"
NO_PATH = "D:/desktop/学习结果小文件/neuroglyph34_project/struct32_aligned.json"
OUT = os.path.join(BASE, "kg_full_index.json")
MAX_ROWS = 400000

# ---------- 字表：三源交集 ----------
TREE = json.load(open(os.path.join(BASE, "tree_data.json"), encoding="utf-8"))
SF = json.load(open(os.path.join(BASE, "stroke_features.json"), encoding="utf-8"))
NO = json.load(open(NO_PATH, encoding="utf-8"))["nopy26"]
char_table = set(TREE) & set(SF) & set(NO)
sys.stderr.write(f"char_table size = {len(char_table)}\n")
sys.stderr.flush()

files = sorted(f for f in os.listdir(KG_DIR) if f.endswith(".csv"))
domains = [f.replace("_all.csv", "") for f in files]
N = {c: defaultdict(int) for c in char_table}
T_tokens = defaultdict(int)
T_chars = defaultdict(int)

for f in files:
    dom = f.replace("_all.csv", "")
    path = os.path.join(KG_DIR, f)
    sys.stderr.write(f"scan {f} ...\n"); sys.stderr.flush()
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        for row in r:
            if len(row) < 3:
                continue
            head, tail = row[0], row[2]
            T_tokens[dom] += 2
            T_chars[dom] += len(head) + len(tail)
            # 逐字符查表：head / tail 各计一次（与原探针逻辑一致）
            for ch in set(head):
                if ch in char_table:
                    N[ch][dom] += 1
            for ch in set(tail):
                if ch in char_table:
                    N[ch][dom] += 1
            n += 1
            if n >= MAX_ROWS:
                break

result = {
    "targets": sorted(char_table),
    "domains": domains,
    "N": {c: dict(N[c]) for c in char_table},
    "T_tokens": dict(T_tokens),
    "T_chars": dict(T_chars),
}
json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
sys.stderr.write(f"DONE -> {OUT}\n")
sys.stderr.write(f"chars={len(char_table)} domains={len(domains)}\n")
