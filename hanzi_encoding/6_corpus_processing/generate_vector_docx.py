#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成18维向量编码表Word文档
"""

import json
import os

# 文件路径
WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
OUTPUT_FILE = os.path.join(WORK_DIR, "五千常用汉字18维向量编码表.docx")

print("📖 读取18维向量数据...")

with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

characters = data['characters']
print(f"   字符数量: {len(characters)}")

# 使用python-docx创建Word文档
try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
except ImportError:
    print("正在安装 python-docx...")
    import subprocess
    subprocess.run(['pip', 'install', 'python-docx'], check=True)
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

print("📝 创建Word文档...")

doc = Document()

# 设置中文字体
def set_chinese_font(run, font_name='宋体', font_size=12):
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

# ===== 封面 =====
title = doc.add_heading('', level=0)
run = title.add_run('五千常用汉字18维向量编码表')
run.font.size = Pt(28)
run.font.bold = True

doc.add_paragraph()
doc.add_paragraph()

# 副标题
p = doc.add_paragraph()
run = p.add_run('——四角号码·五笔字型·拼音·声调编码')
run.font.size = Pt(18)

doc.add_paragraph()
doc.add_paragraph()
doc.add_paragraph()

# 基本信息
p = doc.add_paragraph()
run = p.add_run(f'字符总数：{len(characters)} 个')
run.font.size = Pt(14)

p = doc.add_paragraph()
run = p.add_run(f'向量维度：18 维')
run.font.size = Pt(14)

p = doc.add_paragraph()
run = p.add_run(f'版本：2.0 扩展版')
run.font.size = Pt(14)

# 页面分隔
doc.add_page_break()

# ===== 编码规则说明 =====
doc.add_heading('一、编码规则说明', level=1)

# 表格：编码位置说明
table = doc.add_table(rows=7, cols=4)
table.style = 'Table Grid'

# 表头
headers = ['位置', '含义', '编码规则', '示例']
for i, header in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = header
    cell.paragraphs[0].runs[0].font.bold = True

# 数据
rules_data = [
    ['1-5位', '四角号码', '0-9数字，按汉字四角编码', '例：汉=3710'],
    ['6-9位', '五笔字型', 'a-z对应1-26，无码补0', '例：汉=4514'],
    ['10-15位', '拼音编码', '字母a-z，不足6位用0补齐', '例：汉=han00'],
    ['16位', '声调', '1=平声 2=上声 3=去声 4=入声', '例：汉=4(入声)'],
    ['17位', '字符类型', '0=汉字 1=数字 2=字母 3=符号', '例：汉=0(汉字)'],
    ['18位', '保留位', '预留扩展', '默认=0'],
]

for row_idx, row_data in enumerate(rules_data, start=1):
    for col_idx, cell_data in enumerate(row_data):
        table.rows[row_idx].cells[col_idx].text = cell_data

doc.add_paragraph()

# 详细说明
doc.add_heading('二、详细说明', level=2)

doc.add_paragraph('1. 四角号码查字法：用汉字的四个角的号码来检索汉字。')
doc.add_paragraph('2. 五笔字型输入法：将汉字拆分为字根，每个字根对应一个按键（a=1, b=2, ..., z=26）。')
doc.add_paragraph('3. 拼音编码：采用汉语拼音方案，不带声调，用字母表示。')
doc.add_paragraph('4. 声调编码：仅适用于汉字，数字/字母/符号的声调设为0。')

doc.add_page_break()

# ===== 向量编码表 =====
doc.add_heading('三、五千常用汉字向量编码表', level=1)

# 每100个字符创建一个小表格（共10114个字符，需要约102页）
chars_per_table = 100

for i, char_info in enumerate(characters):
    if i % chars_per_table == 0:
        if i > 0:
            doc.add_page_break()
        
        start_idx = i
        end_idx = min(i + chars_per_table, len(characters))
        doc.add_heading(f'（第 {start_idx+1} - {end_idx} 个字符）', level=2)
        
        # 创建表格
        table = doc.add_table(rows=0, cols=5)
        table.style = 'Table Grid'
        
        # 表头
        header_cells = table.add_row().cells
        header_cells[0].text = '序号'
        header_cells[1].text = '汉字'
        header_cells[2].text = '拼音'
        header_cells[3].text = '18维向量编码'
        header_cells[4].text = '四角号'
        
        for cell in header_cells:
            cell.paragraphs[0].runs[0].font.bold = True
    
    # 添加行
    char = char_info['char']
    pinyin = char_info.get('pinyin', '')
    vector = char_info['vector']
    
    # 四角号码（取前5位）
    sijiao = ''.join(map(str, vector[0:5]))
    
    # 五笔（取4-8位）
    wubi = ''.join(map(str, vector[5:9]))
    
    # 拼音（取9-15位）
    pinyin_code = ''.join([chr(c + 96) if c > 0 else '0' for c in vector[9:15]])
    
    # 声调（16位）
    tone = vector[15]
    
    # 类型（17位）
    char_type = vector[16]
    
    # 完整编码
    code = f"{sijiao}|{wubi}|{pinyin_code}|{tone}|{char_type}|0"
    
    row_cells = table.add_row().cells
    row_cells[0].text = str(i + 1)
    row_cells[1].text = char
    row_cells[2].text = pinyin
    row_cells[3].text = code
    row_cells[4].text = sijiao

# 保存
print(f"💾 保存文档: {OUTPUT_FILE}")
doc.save(OUTPUT_FILE)
print("✅ 完成!")
