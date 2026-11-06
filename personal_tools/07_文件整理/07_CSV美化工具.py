import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule

# 读取CSV数据
csv_path = "豆瓣TOP250_初代版.csv"
excel_path = "豆瓣TOP250_美化版.xlsx"
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 保存为Excel
df.to_excel(excel_path, index=False, engine="openpyxl")

# 打开Excel并美化格式
wb = load_workbook(excel_path)
ws = wb.active

# 1. 设置表头样式
header_font = Font(bold=True, size=12)
header_fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
for cell in ws[1]:
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal="center", vertical="center")

# 2. 调整列宽
ws.column_dimensions["A"].width = 20  # 书名
ws.column_dimensions["B"].width = 8   # 评分
ws.column_dimensions["C"].width = 20  # 作者
ws.column_dimensions["D"].width = 50  # 简介

# 3. 数据对齐
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for cell in row:
        cell.alignment = Alignment(horizontal="left", vertical="center")

# 4. 评分列添加色阶（高分标红）
ws.conditional_formatting.add(
    "B2:B{}".format(ws.max_row),
    ColorScaleRule(
        start_type="num", start_value=9.0, start_color="FFFFFF",
        mid_type="num", mid_value=9.3, mid_color="FFE6E6",
        end_type="num", end_value=9.6, end_color="FF0000"
    )
)

# 保存美化后的Excel
wb.save(excel_path)
print(f"Excel美化完成，保存至：{excel_path}")