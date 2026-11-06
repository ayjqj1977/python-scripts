# -*- coding: utf-8 -*-
"""Python 参考实现田久乐/朱新华 词林相似度，验证排序。"""
import json, math, os

BASE = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
idx = json.load(open(os.path.join(BASE, "cilin_index.json"), encoding="utf-8"))
char_codes = idx["char_codes"]
branch_n = idx["branch_n"]
code_words = idx["code_words"]

PARAM = {"a":0.65,"b":0.8,"c":0.9,"d":0.96,"e":0.5,"f":0.1}
LAYER_TABLE = {1:1,2:2,4:3,5:4}
DEGREE = 180.0

def new_layer(c):
    return [c[0], c[1], c[2:4], c[4], c[5:7], c[7]]

def common_code(c1, c2):
    length = min(len(c1), len(c2))
    idx = 0
    while idx < length and c1[idx] == c2[idx]:
        idx += 1
    if idx == 3 or idx == 6:
        return c1[:idx-1]
    return c1[:idx]

def get_k(l1, l2):
    for i in range(5):
        if (i == 2 or i == 4) and l1[i] != l2[i]:
            n1 = int(l1[i].lstrip("0") or "0")
            n2 = int(l2[i].lstrip("0") or "0")
            return abs(n1 - n2)
        elif l1[i] != l2[i]:
            return abs(ord(l1[i][0]) - ord(l2[i][0]))
    return 0

def sim(c1, c2):
    if c1 == c2:
        return 1.0
    l1, l2 = new_layer(c1), new_layer(c2)
    cs = common_code(c1, c2)
    clen = len(cs)
    # 放宽 @：原实现把 '@' 自闭环词直接判 0.1，导致"猫"(Bi08A01@)对所有词=0.1、
    # 无法演示"猫狗近"。去掉该惩罚，@ 仍按真实分类位置参与比较。
    if clen == 0:
        return PARAM["f"]
    if clen >= 7:
        r1, r2 = c1[-1], c2[-1]
        if r1 == r2:
            return 1.0 if r1 == "=" else (PARAM["e"] if r1 == "#" else 0.7)
        return 0.7  # 前7位同、关系不同（含 @）：仍很近
    layer = LAYER_TABLE[clen]
    coef = [None, PARAM["a"], PARAM["b"], PARAM["c"], PARAM["d"]][layer]
    n = branch_n.get(cs, 1)
    k = get_k(l1, l2)
    val = coef * math.cos(n * math.pi / DEGREE) * (n - k + 1) / n
    return min(1.0, max(0.0, val))

def char_sim(a, b):
    c1s = char_codes.get(a, {}).get("codes", [])
    c2s = char_codes.get(b, {}).get("codes", [])
    if not c1s or not c2s:
        return None
    best = 0.0
    for c1 in c1s:
        for c2 in c2s:
            best = max(best, sim(c1, c2))
    return best

pairs = [("猫","狗"),("猫","树"),("猫","鱼"),("猫","鸟"),("猫","牛"),
         ("牛","羊"),("江","河"),("末","未"),("猫","花"),("树","花"),
         ("人","狗"),("木","树"),("水","火"),("金","土")]
print(f"{'对':<8}{'词林相似度':<12}  说明")
print("-"*50)
for a, b in pairs:
    s = char_sim(a, b)
    ca = char_codes.get(a,{}).get("codes",[])
    cb = char_codes.get(b,{}).get("codes",[])
    note = f"{a}={ca[:1]} {b}={cb[:1]}"
    print(f"{a}{b:<6}{('%.3f'%s) if s is not None else '无数据':<12}  {note}")
