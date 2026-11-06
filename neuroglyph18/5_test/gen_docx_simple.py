# -*- coding: utf-8 -*-
"""
直接用现有字库生成Word文档
==========================
"""

import json
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
OUTPUT_DOCX = os.path.join(WORK_DIR, "1_18维向量表", "五千常用汉字十八维向量编码表.docx")

print("=" * 60)
print("📄 直接用现有字库生成Word文档")
print("=" * 60)

# 加载数据
print("\n📖 加载字库...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

characters = vector_data['characters']
print(f"   字符数: {len(characters)}")

# 创建Word文档
doc = Document()

# 设置中文字体
def set_chinese_font(run, font_name='宋体', font_size=10):
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

# 标题
title = doc.add_heading('', level=0)
run = title.add_run('五千常用汉字十八维向量编码表')
set_chinese_font(run, '黑体', 22)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 副标题
subtitle = doc.add_paragraph()
run = subtitle.add_run('—— 基于6000常用字扩展版 ——')
set_chinese_font(run, '宋体', 14)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 说明
doc.add_paragraph()
intro = doc.add_paragraph()
run = intro.add_run('编码规则说明：')
set_chinese_font(run, '宋体', 10)
run.bold = True

intro.add_run('\n维度 1-5（四角号码）：字形结构编码')
intro.add_run('\n维度 6-9（五笔编码）：字根编码，a=1, b=2, ..., z=26')
intro.add_run('\n维度 10-15（拼音编码）：拼音字母对应数字，a=1, ..., z=26，不足6位用0补齐')
intro.add_run('\n维度 16（声调）：1=平声，2=上声，3=去声，4=入声')
intro.add_run('\n维度 17-18（类型）：0=汉字，1=数字，2=字母，3=符号')

# 示例
example = doc.add_paragraph()
run = example.add_run('\n示例：' + '一' + ' → 四角[7,8,2,2,2] 五笔[25,9,1,1] 拼音[25,9,0,0,0,0] 声调[1] 类型[0,0]')
set_chinese_font(run, '宋体', 9)

# 来源
source = doc.add_paragraph()
run = source.add_run(f'\n数据来源：汉字向量字库_扩展.json（共 {len(characters)} 个字符）')
set_chinese_font(run, '宋体', 9)
source.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 创建表格
print("\n📊 创建表格...")
table = doc.add_table(rows=1, cols=7)
table.style = 'Table Grid'

# 列宽
col_widths = [Cm(1.0), Cm(1.2), Cm(2.0), Cm(2.0), Cm(2.5), Cm(0.8), Cm(1.0)]
for i, width in enumerate(col_widths):
    table.columns[i].width = width

# 表头
header_cells = table.rows[0].cells
headers = ['序号', '汉字', '四角号码', '五笔编码', '拼音编码', '声调', '类型']
for i, header in enumerate(headers):
    header_cells[i].text = header
    for paragraph in header_cells[i].paragraphs:
        for run in paragraph.runs:
            set_chinese_font(run, '黑体', 10)
            run.bold = True
    header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

# 添加数据行（取前5000）
print("   写入数据行...")
count = 0
for char_info in characters[:5000]:
    char = char_info['char']
    vector = char_info['vector']
    pinyin = char_info.get('pinyin', '')
    
    # 提取各维度
    sijiao = ''.join(str(v) for v in vector[0:5])  # 四角号码
    wubi = ''.join(chr(ord('a') + v - 1) if 1 <= v <= 26 else '0' for v in vector[5:9])  # 五笔
    pinyin_code = str(vector[9:15]).replace(' ', '')  # 拼音编码
    tone = vector[15]  # 声调
    char_type = vector[16]  # 类型
    
    # 声调名称
    tone_names = {1: '平', 2: '上', 3: '去', 4: '入', 0: '-'}
    tone_name = tone_names.get(tone, str(tone))
    
    # 类型名称
    type_names = {0: '汉字', 1: '数字', 2: '字母', 3: '符号'}
    type_name = type_names.get(char_type, str(char_type))
    
    # 添加行
    row_cells = table.add_row().cells
    row_cells[0].text = str(count + 1)
    row_cells[1].text = char
    row_cells[2].text = sijiao
    row_cells[3].text = wubi
    row_cells[4].text = pinyin_code
    row_cells[5].text = tone_name
    row_cells[6].text = type_name
    
    # 格式化
    for i, cell in enumerate(row_cells):
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_chinese_font(run, '宋体', 9)
    
    count += 1
    if count % 1000 == 0:
        print(f"   已处理 {count} 行...")

print(f"   完成! 共写入 {count} 行")

# 统计
doc.add_paragraph()
stats = doc.add_paragraph()
run = stats.add_run(f'\n统计：共收录 {len(characters)} 个字符，本表展示前 {count} 个')
set_chinese_font(run, '宋体', 9)

# 保存
print(f"\n💾 保存文档: {OUTPUT_DOCX}")
doc.save(OUTPUT_DOCX)

print("\n✅ Word文档生成完成!")
