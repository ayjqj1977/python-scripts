# -*- coding: utf-8 -*-
"""
修复拼音和声调编码
==================
"""

import json
import os
import re
import unicodedata

WORK_DIR = r"D:\desktop\学习结果小文件"
XINHUA_FILE = os.path.join(WORK_DIR, "7_训练语料", "xinhua_word.json")
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
OUTPUT_JSON = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_新华字典版.json")

print("=" * 60)
print("🔧 修复拼音和声调编码")
print("=" * 60)

# 声调符号映射：直接根据最后的符号判断声调
# 第一声(阴平): ā ē ī ō ū ǖ 
# 第二声(阳平): á é í ó ú ǘ
# 第三声(上声): ǎ ě ǐ ǒ ǔ ǚ
# 第四声(去声): à è ì ò ù ǜ

# 完整的声调映射表
TONE_MARK_MAP = {
    # 第一声
    'ā': (1, 'a'), 'ē': (1, 'e'), 'ī': (1, 'i'), 'ō': (1, 'o'), 'ū': (1, 'u'), 'ǖ': (1, 'v'),
    'āi': (1, 'ai'), 'ēi': (1, 'ei'), 'ān': (1, 'an'), 'ēn': (1, 'en'), 'īn': (1, 'in'), 'ūn': (1, 'un'), 'ǖn': (1, 'vn'),
    'āng': (1, 'ang'), 'ēng': (1, 'eng'), 'īng': (1, 'ing'), 'ōng': (1, 'ong'),
    # 第二声  
    'á': (2, 'a'), 'é': (2, 'e'), 'í': (2, 'i'), 'ó': (2, 'o'), 'ú': (2, 'u'), 'ǘ': (2, 'v'),
    'ái': (2, 'ai'), 'éi': (2, 'ei'), 'án': (2, 'an'), 'én': (2, 'en'), 'ín': (2, 'in'), 'ún': (2, 'un'), 'ǘn': (2, 'vn'),
    'áng': (2, 'ang'), 'éng': (2, 'eng'), 'íng': (2, 'ing'), 'óng': (2, 'ong'),
    # 第三声
    'ǎ': (3, 'a'), 'ě': (3, 'e'), 'ǐ': (3, 'i'), 'ǒ': (3, 'o'), 'ǔ': (3, 'u'), 'ǚ': (3, 'v'),
    'ǎi': (3, 'ai'), 'ěi': (3, 'ei'), 'ǎn': (3, 'an'), 'ěn': (3, 'en'), 'ǐn': (3, 'in'), 'ǔn': (3, 'un'), 'ǚn': (3, 'vn'),
    'ǎng': (3, 'ang'), 'ěng': (3, 'eng'), 'ǐng': (3, 'ing'), 'ǒng': (3, 'ong'),
    # 第四声
    'à': (4, 'a'), 'è': (4, 'e'), 'ì': (4, 'i'), 'ò': (4, 'o'), 'ù': (4, 'u'), 'ǜ': (4, 'v'),
    'ài': (4, 'ai'), 'èi': (4, 'ei'), 'àn': (4, 'an'), 'èn': (4, 'en'), 'ìn': (4, 'in'), 'ùn': (4, 'un'), 'ǜn': (4, 'vn'),
    'àng': (4, 'ang'), 'èng': (4, 'eng'), 'ìng': (4, 'ing'), 'òng': (4, 'ong'),
}

# 简化版：单个字符
SIMPLE_TONE_MAP = {
    'ā': (1, 'a'), 'á': (2, 'a'), 'ǎ': (3, 'a'), 'à': (4, 'a'),
    'ē': (1, 'e'), 'é': (2, 'e'), 'ě': (3, 'e'), 'è': (4, 'e'),
    'ī': (1, 'i'), 'í': (2, 'i'), 'ǐ': (3, 'i'), 'ì': (4, 'i'),
    'ō': (1, 'o'), 'ó': (2, 'o'), 'ǒ': (3, 'o'), 'ò': (4, 'o'),
    'ū': (1, 'u'), 'ú': (2, 'u'), 'ǔ': (3, 'u'), 'ù': (4, 'u'),
    'ǖ': (1, 'v'), 'ǘ': (2, 'v'), 'ǚ': (3, 'v'), 'ǜ': (4, 'v'),
    'ü': (0, 'v'),  # 无声调
}

def extract_tone_and_letters(pinyin):
    """从拼音提取声调和纯字母"""
    if not pinyin:
        return 1, 'yi'
    
    p = pinyin.strip()
    
    # 检查完整匹配（2-4个字符）
    for i in range(len(p), 1, -1):
        substr = p[:i]
        if substr in TONE_MARK_MAP:
            tone, letters = TONE_MARK_MAP[substr]
            # 加上剩余部分
            letters = letters + p[i:] if p[i:] else letters
            return tone, letters
    
    # 检查单个字符
    if p[0] in SIMPLE_TONE_MAP:
        tone, letter = SIMPLE_TONE_MAP[p[0]]
        return tone, letter + p[1:] if len(p) > 1 else letter
    
    # 没有声调符号，默认第一声
    return 1, p


def pinyin_to_numbers(pinyin):
    """拼音字母转数字 (a=1, b=2, ..., z=26)"""
    if not pinyin:
        return [0] * 6
    
    # 清理：移除数字后缀，转小写
    clean = re.sub(r'[1-4]$', '', pinyin).lower()
    
    # ü 转为 v
    clean = clean.replace('ü', 'v')
    
    # 转换为数字
    result = []
    for c in clean[:6]:
        if 'a' <= c <= 'z':
            result.append(ord(c) - ord('a') + 1)
        else:
            result.append(0)
    
    # 补齐
    while len(result) < 6:
        result.append(0)
    
    return result[:6]


# 测试
print("\n🔬 测试声调提取:")
test_cases = ['á', 'ā', 'à', 'ái', 'ài', 'ēi', 'āng', 'òng', 'yi']
for p in test_cases:
    tone, letters = extract_tone_and_letters(p)
    nums = pinyin_to_numbers(letters)
    tone_names = {1: '平', 2: '上', 3: '去', 4: '入'}
    print(f"   {p} -> 声调:{tone_names.get(tone,'?')}({tone}), 字母:{letters}, 编码:{nums}")

# 加载数据
print("\n📖 加载数据...")
with open(XINHUA_FILE, 'r', encoding='utf-8') as f:
    xinhua_data = json.load(f)

with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

existing_chars = {}
for char_info in vector_data['characters']:
    char = char_info['char']
    existing_chars[char] = char_info['vector']

# 整合
print("\n🔄 整合数据...")

integrated_data = {
    "meta": {
        "name": "汉字向量字库_新华字典版",
        "version": "3.2",
        "total_count": 0,
        "vector_dimensions": 18,
        "description": "汉字18维向量编码表 - 整合新华字典数据",
    },
    "characters": []
}

xinhua_dict = {}
for item in xinhua_data:
    char = item.get('word', '')
    if not char or len(char) != 1:
        continue
    
    pinyin_raw = item.get('pinyin', 'yi')
    explanation = item.get('explanation', '')
    radicals = item.get('radicals', '')
    strokes = item.get('strokes', '0')
    
    # 提取声调和拼音字母
    tone, pinyin_clean = extract_tone_and_letters(pinyin_raw)
    pinyin_nums = pinyin_to_numbers(pinyin_clean)
    
    # 18维向量
    vector = [
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
        pinyin_nums[0], pinyin_nums[1], pinyin_nums[2],
        pinyin_nums[3], pinyin_nums[4], pinyin_nums[5],
        tone,
        0, 0
    ]
    
    # 合并现有的四角号码和五笔数据
    if char in existing_chars:
        existing_vec = existing_chars[char]
        vector[0:5] = existing_vec[0:5]
        vector[5:9] = existing_vec[5:9]
    
    explanation_short = explanation[:100].replace('\n', ' ') if explanation else ''
    
    xinhua_dict[char] = {
        "char": char,
        "pinyin": pinyin_raw,  # 保留原始拼音
        "radicals": radicals,
        "strokes": strokes,
        "explanation": explanation_short,
        "vector": vector
    }

integrated_data['characters'] = list(xinhua_dict.values())
integrated_data['meta']['total_count'] = len(integrated_data['characters'])

# 统计
print("\n📊 统计:")
tone_count = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
for char_info in integrated_data['characters']:
    tone = char_info['vector'][15]
    tone_count[tone] = tone_count.get(tone, 0) + 1

print(f"   总字符: {len(integrated_data['characters'])}")
print(f"   声调: 平={tone_count.get(1,0)}, 上={tone_count.get(2,0)}, 去={tone_count.get(3,0)}, 入={tone_count.get(4,0)}")

# 示例
print("\n📝 示例:")
for char_info in integrated_data['characters'][:5]:
    c = char_info['char']
    p = char_info['pinyin']
    v = char_info['vector']
    print(f"   {c}: 拼音={p}, 声调={v[15]}, 拼音码={v[9:15]}")

# 保存
print(f"\n💾 保存: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(integrated_data, f, ensure_ascii=False, indent=2)

print("\n✅ 完成!")
