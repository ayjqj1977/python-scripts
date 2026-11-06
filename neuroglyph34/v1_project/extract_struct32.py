# -*- coding: utf-8 -*-
"""
extract_struct32.py — 把 32维向量_完整数字版.xlsx 转成 char->[32] 的 JSON
供融合探针(fuse_struct_pixel_v2.py)使用。
运行环境：受管 venv（含 openpyxl），不需要 torch。
"""
import openpyxl, json, os, numpy as np

XLSX = "D:/desktop/学习结果小文件/neuroglyph34_project/32维向量_完整数字版.xlsx"
OUT = "D:/desktop/学习结果小文件/neuroglyph34_project/struct32_aligned.json"

wb = openpyxl.load_workbook(XLSX, read_only=True)
ws = wb["32维向量"]
rows = list(ws.iter_rows(values_only=True))
header = list(rows[0])
data = rows[1:]
idx = {h: i for i, h in enumerate(header)}

# 列区间（基于 header 顺序）：
# 1..5 四角, 6..9 五笔, 10 总笔画, 11..13 部首索引, 14 部首外笔画,
# 15 同部同画序号, 16 结构, 17..26 笔顺(10), 27..32 拼音(6), 33..34 符号(2)
COL_ALL32 = list(range(1, 33))          # 去掉符号(33,34) 的 32 维
COL_NOPY = list(range(1, 27))          # 再去掉拼音(27..32) 的 26 维结构码
COL_SYM = [33, 34]

m_all32 = {}
m_nopy = {}
m_withsym = {}   # 26 结构 + 2 符号 = 28
miss = 0
for r in data:
    ch = r[0]
    if not ch or not isinstance(ch, str):
        continue
    vec = []
    ok = True
    for c in COL_ALL32:
        v = r[c]
        if v is None:
            v = 0
        try:
            vec.append(float(v))
        except (ValueError, TypeError):
            ok = False
            break
    if not ok or len(vec) != 32:
        miss += 1
        continue
    m_all32[ch] = vec
    m_nopy[ch] = [vec[i] for i in range(26)]  # COL_NOPY 即前 26 列
    m_withsym[ch] = [vec[i] for i in range(26)] + [float(r[c]) if r[c] is not None else 0.0 for c in COL_SYM]

print(f"[Extract] 有效字符: all32={len(m_all32)} nopy26={len(m_nopy)} withsym28={len(m_withsym)}, 跳过={miss}")

out = {
    "meta": {
        "source": os.path.basename(XLSX),
        "col_all32": "四角5+五笔4+总笔画1+部首索引3+部首外笔画1+同部同画序号1+结构1+笔顺10+拼音6",
        "col_nopy26": "all32 去掉拼音6位（纯结构码）",
        "col_withsym28": "nopy26 + 符号2",
    },
    "all32": m_all32,
    "nopy26": m_nopy,
    "withsym28": m_withsym,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"[Save] {OUT}  size={os.path.getsize(OUT)//1024}KB")
