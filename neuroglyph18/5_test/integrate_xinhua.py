# -*- coding: utf-8 -*-
"""
整合新华字典数据到18维向量表并生成Word文档
============================================
"""

import json
import os
import re

# 路径配置
WORK_DIR = r"D:\desktop\学习结果小文件"
XINHUA_FILE = os.path.join(WORK_DIR, "7_训练语料", "xinhua_word.json")
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
OUTPUT_JSON = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_新华字典版.json")
OUTPUT_DOCX = os.path.join(WORK_DIR, "1_18维向量表", "五千常用汉字18维向量编码表_新华字典版.docx")

print("=" * 60)
print("📚 整合新华字典到18维向量表")
print("=" * 60)

# 1. 加载新华字典数据
print("\n📖 加载新华字典...")
with open(XINHUA_FILE, 'r', encoding='utf-8') as f:
    xinhua_data = json.load(f)
print(f"   新华字典汉字数: {len(xinhua_data)}")

# 2. 加载现有18维向量
print("📊 加载18维向量表...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

# 构建现有字符向量字典
existing_chars = {}
for char_info in vector_data['characters']:
    char = char_info['char']
    existing_chars[char] = char_info['vector']

print(f"   现有向量字符数: {len(existing_chars)}")

# 3. 创建拼音到声调的映射
def get_tone_from_pinyin(pinyin):
    """从拼音提取声调"""
    if not pinyin:
        return 0
    # 去除声调数字，获取韵母部分
    tone_match = re.search(r'([1-4])$', pinyin)
    if tone_match:
        return int(tone_match.group(1))
    return 1  # 默认平声

def pinyin_to_numbers(pinyin):
    """拼音字母转数字 (a=1, b=2, ..., z=26)"""
    if not pinyin:
        return [0] * 6
    
    # 去除声调数字
    pinyin_clean = re.sub(r'[1-4]$', '', pinyin).lower()
    
    # 只保留字母
    letters = re.findall(r'[a-z]', pinyin_clean)
    
    # 转换为数字
    result = []
    for c in letters[:6]:  # 最多6位
        result.append(ord(c) - ord('a') + 1)
    
    # 补齐到6位
    while len(result) < 6:
        result.append(0)
    
    return result[:6]

# 4. 创建新华字典整合数据
print("\n🔄 整合数据...")

# 新华字典 → 18维向量映射
# 维度定义：
# 0-4: 四角号码 (暂时用0填充，需要专门查询)
# 5-8: 五笔编码 (暂时用0填充，需要专门查询)
# 9-14: 拼音编码
# 15: 声调
# 16-17: 类型

integrated_data = {
    "meta": {
        "name": "汉字向量字库_新华字典版",
        "version": "3.0",
        "total_count": 0,
        "vector_dimensions": 18,
        "description": "汉字18维向量编码表 - 整合新华字典数据",
        "data_source": "新华字典 + 扩展向量表"
    },
    "characters": []
}

# 处理新华字典中的每个汉字
xinhua_dict = {}
for item in xinhua_data:
    char = item.get('word', '')
    if not char or len(char) != 1:
        continue
    
    pinyin = item.get('pinyin', 'yi1')
    explanation = item.get('explanation', '')
    radicals = item.get('radicals', '')
    strokes = item.get('strokes', '0')
    
    # 提取拼音数字编码
    pinyin_nums = pinyin_to_numbers(pinyin)
    
    # 提取声调
    tone = get_tone_from_pinyin(pinyin)
    
    # 18维向量
    vector = [
        0, 0, 0, 0, 0,  # 四角号码 (0-4)
        0, 0, 0, 0,     # 五笔编码 (5-8)
        pinyin_nums[0], pinyin_nums[1], pinyin_nums[2],  # 拼音 (9-14)
        pinyin_nums[3], pinyin_nums[4], pinyin_nums[5],
        tone,           # 声调 (15)
        0, 0            # 类型 (16-17): 0=汉字
    ]
    
    # 如果现有向量表中有这个字符，合并数据
    if char in existing_chars:
        existing_vec = existing_chars[char]
        # 使用现有的四角号码和五笔数据
        vector[0:5] = existing_vec[0:5]
        vector[5:9] = existing_vec[5:9]
    
    # 简化解释（取前100字）
    explanation_short = explanation[:100].replace('\n', ' ') if explanation else ''
    
    xinhua_dict[char] = {
        "char": char,
        "pinyin": pinyin,
        "radicals": radicals,
        "strokes": strokes,
        "explanation": explanation_short,
        "vector": vector
    }

# 添加到整合数据
integrated_data['characters'] = list(xinhua_dict.values())
integrated_data['meta']['total_count'] = len(integrated_data['characters'])

print(f"   整合后字符数: {len(integrated_data['characters'])}")

# 5. 保存JSON
print(f"\n💾 保存JSON: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(integrated_data, f, ensure_ascii=False, indent=2)

print("✅ JSON保存完成!")

# 6. 输出统计信息
print("\n📊 数据统计:")
print(f"   总汉字数: {len(integrated_data['characters'])}")

# 统计声调分布
tone_count = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
for char_info in integrated_data['characters']:
    tone = char_info['vector'][15]
    tone_count[tone] = tone_count.get(tone, 0) + 1

print(f"   声调分布: 平声={tone_count.get(1,0)}, 上声={tone_count.get(2,0)}, 去声={tone_count.get(3,0)}, 入声={tone_count.get(4,0)}, 未知={tone_count.get(0,0)}")

print("\n✅ 整合完成!")
print(f"\n📝 接下来生成Word文档...")
