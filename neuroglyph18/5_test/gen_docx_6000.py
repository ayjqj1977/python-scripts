# -*- coding: utf-8 -*-
"""
用原始6000字库生成Word文档
==========================
"""

import json
import os
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库.json")
OUTPUT_DOCX = os.path.join(WORK_DIR, "五千常用汉字十八维向量编码表_正式版.docx")

print("=" * 60)
print("📄 用原始6000字库生成Word文档")
print("=" * 60)

# 加载原始6000字库
print("\n📖 加载原始字库...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

characters = vector_data['characters']
print(f"   字符数: {len(characters)}")

# 创建Word文档
doc = Document()

def set_chinese_font(run, font_name='宋体', font_size=10):
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

# 标题
title = doc.add_heading('', level=0)
run = title.add_run('五千常用汉字十八维向量编码表')
set_chinese_font(run, '黑体', 22)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 说明
doc.add_paragraph()
intro = doc.add_paragraph()
run = intro.add_run('编码规则：')
set_chinese_font(run, '宋体', 10)
run.bold = True
intro.add_run('\n1-5位：四角号码')
intro.add_run('\n6-9位：五笔编码(a=1,...,z=26)')
intro.add_run('\n10-15位：拼音编码(a=1,...,z=26)')
intro.add_run('\n16位：声调(1=平,2=上,3=去,4=入)')
intro.add_run('\n17-18位：类型(0=汉字,1=数字,2=字母,3=符号)')

# 来源
source = doc.add_paragraph()
run = source.add_run(f'\n共 {len(characters)} 个字符')
set_chinese_font(run, '宋体', 9)
source.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 表格
print("\n📊 创建表格...")
table = doc.add_table(rows=1, cols=6)
table.style = 'Table Grid'

col_widths = [Cm(1.0), Cm(1.2), Cm(2.2), Cm(2.2), Cm(2.5), Cm(1.2)]
for i, width in enumerate(col_widths):
    table.columns[i].width = width

# 表头
header_cells = table.rows[0].cells
headers = ['序号', '汉字', '四角号', '五笔码', '拼音码', '声调类型']
for i, header in enumerate(headers):
    header_cells[i].text = header
    for p in header_cells[i].paragraphs:
        for r in p.runs:
            set_chinese_font(r, '黑体', 10)
            r.bold = True

# 数据行
print("   写入数据...")
count = 0
for char_info in characters:
    char = char_info['char']
    vector = char_info['vector']
    
    sijiao = ''.join(str(v) for v in vector[0:5])
    wubi = ','.join(str(v) for v in vector[5:9])
    pinyin_code = ','.join(str(v) for v in vector[9:15])
    tone = str(vector[15])
    char_type = str(vector[16])
    
    row_cells = table.add_row().cells
    row_cells[0].text = str(count + 1)
    row_cells[1].text = char
    row_cells[2].text = sijiao
    row_cells[3].text = wubi
    row_cells[4].text = pinyin_code
    row_cells[5].text = f"{tone}/{char_type}"
    
    for cell in row_cells:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                set_chinese_font(r, '宋体', 9)
    
    count += 1
    if count % 2000 == 0:
        print(f"   已处理 {count} 行...")

print(f"   完成! 共 {count} 行")

# 保存
print(f"\n💾 保存: {OUTPUT_DOCX}")
doc.save(OUTPUT_DOCX)
print("\n✅ 完成!")
