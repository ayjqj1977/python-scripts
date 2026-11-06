import openpyxl, json, os

XLSX = r"D:/desktop/学习结果小文件/neuroglyph34_project/32维向量_完整数字版.xlsx"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_tree_fields.json")

def to_int(v):
    try:
        return int(v)
    except Exception:
        return 0

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb.active
rows = ws.iter_rows(values_only=True)
header = next(rows)

data = {}
for row in rows:
    ch = row[0]
    if not ch:
        continue
    ch = str(ch)
    radical_idx = (to_int(row[11]), to_int(row[12]), to_int(row[13]))
    rec = {
        "char": ch,
        "four_corner": [to_int(row[1 + i]) for i in range(5)],
        "five_stroke": [to_int(row[6 + i]) for i in range(4)],
        "total_strokes": to_int(row[10]),
        "radical_idx": radical_idx,
        "radical_id": radical_idx[0] * 100 + radical_idx[1] * 10 + radical_idx[2],
        "radical_outside": to_int(row[14]),
        "same_rad_seq": to_int(row[15]),
        "structure": to_int(row[16]),
        "stroke_order": [to_int(row[17 + i]) for i in range(10)],
    }
    data[ch] = rec

json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

# 对照 spot-check
checks = ["风", "冈", "王", "玉", "上", "月", "后", "右", "末", "未", "干", "千", "日", "曰", "凤"]
print(f"[extract] 总字 {len(data)}，写出 {OUT}")
print(f"{'字':<4}{'部首索引':<12}{'结构':<5}{'四角':<14}{'五笔':<12}{'笔顺前4'}")
for c in checks:
    r = data.get(c)
    if not r:
        print(f"{c:<4} 无数据"); continue
    print(f"{c:<4}{str(r['radical_idx']):<12}{r['structure']:<5}{str(r['four_corner']):<14}{str(r['five_stroke']):<12}{str(r['stroke_order'][:4])}")
