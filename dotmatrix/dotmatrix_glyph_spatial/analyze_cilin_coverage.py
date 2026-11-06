#!/usr/bin/env python3
# 量化 词林 对单字的覆盖：精确词条 vs 仅出现在复合词中 vs 完全缺席
import json, re, os

CILIN = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial/cilin_extended.txt"
KGIDX = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial/kg_full_index.json"

# 1) 解析词林
tokens = set()          # 所有独立词条
char_lines = {}         # 单字 -> 出现为独立词条的 code 列表
compound_chars = {}     # 单字 -> 出现于复合词(非独立)的 code 列表
code_words = {}         # code -> [words]
with open(CILIN, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^([A-Za-z0-9]+)([=#@]?)\s+(.*)$", line)
        if not m:
            continue
        code, rel, words = m.group(1), m.group(2) or "=", m.group(3)
        ws = words.split()
        code_words[code + rel] = ws
        for w in ws:
            tokens.add(w)
            if len(w) == 1:
                char_lines.setdefault(w, []).append(code + rel)
            else:
                for ch in w:
                    if len(ch) == 1:
                        compound_chars.setdefault(ch, set()).add(code + rel)

# 2) 载入 6622 字表
with open(KGIDX, encoding="utf-8") as f:
    kg = json.load(f)
chars = kg["targets"]            # 正确：字表在 targets 字段
N = len(chars)

exact = [c for c in chars if c in char_lines]
compound_only = [c for c in chars if c not in char_lines and c in compound_chars]
absent = [c for c in chars if c not in char_lines and c not in compound_chars]

print(f"总字数: {N}")
print(f"独立词条覆盖: {len(exact)} ({100*len(exact)/N:.1f}%)")
print(f"仅复合词出现: {len(compound_only)} ({100*len(compound_only)/N:.1f}%)")
print(f"完全缺席: {len(absent)} ({100*len(absent)/N:.1f}%)")
print(f"至少出现(独立+复合): {len(exact)+len(compound_only)} ({100*(len(exact)+len(compound_only))/N:.1f}%)")

print("\n--- 测试字情况 ---")
for t in ["猫","狗","树","末","未","牛","羊","江","河","花","鸟","鱼","虫","人","水","火","木","金","土","心"]:
    if t in char_lines:
        print(f"  {t}: 独立词条 -> {char_lines[t][:3]}")
    elif t in compound_chars:
        print(f"  {t}: 仅复合词 -> 例码 {list(compound_chars[t])[:3]}")
    else:
        print(f"  {t}: 缺席")
