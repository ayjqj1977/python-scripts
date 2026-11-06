# -*- coding: utf-8 -*-
"""
修复拼音编码并重新生成Word文档
================================
"""

import json
import os
import re
import unicodedata

# 路径配置
WORK_DIR = r"D:\desktop\学习结果小文件"
XINHUA_FILE = os.path.join(WORK_DIR, "7_训练语料", "xinhua_word.json")
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")
OUTPUT_JSON = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_新华字典版.json")

print("=" * 60)
print("🔧 修复拼音编码")
print("=" * 60)

# 声调符号映射表
PINYIN_TONE_MAP = {
    'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
    'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
    'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
    'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
    'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
    'ü': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ǜ': 'v',
    'ń': 'n', 'ň': 'n', 'n': 'n',
    'ê': 'e', 'ɑ': 'a',
    'āi': 'ai', 'ái': 'ai', 'ǎi': 'ai', 'ài': 'ai',
    'ēi': 'ei', 'éi': 'ei', 'ěi': 'ei', 'èi': 'ei',
    'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
    'ōu': 'ou', 'óu': 'ou', 'ǒu': 'ou', 'òu': 'ou',
    'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
}

# 完整的声母韵母组合
COMPLEX_TONE_MAP = {
    'āi': 'ai', 'ái': 'ai', 'ǎi': 'ai', 'ài': 'ai',
    'ēi': 'ei', 'éi': 'ei', 'ěi': 'ei', 'èi': 'ei',
    'ōu': 'ou', 'óu': 'ou', 'ǒu': 'ou', 'òu': 'ou',
    'ān': 'an', 'án': 'an', 'ǎn': 'an', 'àn': 'an',
    'ēn': 'en', 'én': 'en', 'ěn': 'en', 'èn': 'en',
    'īn': 'in', 'ín': 'in', 'ǐn': 'in', 'ìn': 'in',
    'ūn': 'un', 'ún': 'un', 'ǔn': 'un', 'ùn': 'un',
    'ǖn': 'vn', 'ǘn': 'vn', 'ǚn': 'vn', 'ǜn': 'vn',
    'āng': 'ang', 'áng': 'ang', 'ǎng': 'ang', 'àng': 'ang',
    'ēng': 'eng', 'éng': 'eng', 'ěng': 'eng', 'èng': 'eng',
    'īng': 'ing', 'íng': 'ing', 'ǐng': 'ing', 'ìng': 'ing',
    'ōng': 'ong', 'óng': 'ong', 'ǒng': 'ong', 'òng': 'ong',
    'ia': 'ia', 'ie': 'ie', 'iao': 'iao', 'iu': 'iu',
    'ian': 'ian', 'iang': 'iang', 'iong': 'iong',
    'ua': 'ua', 'uo': 'uo', 'uai': 'uai', 'ui': 'ui',
    'uan': 'uan', 'uang': 'uang', 'üan': 'van', 'ün': 'vn',
}


def remove_tone(pinyin):
    """移除拼音中的声调符号，转换为纯字母"""
    if not pinyin:
        return '', 1
    
    # 先提取声调
    tone = 1  # 默认平声
    tone_match = re.search(r'([1-4])$', pinyin)
    if tone_match:
        tone = int(tone_match.group(1))
        pinyin = pinyin[:-1]  # 移除数字
    
    # 处理带声调的元音符号
    result = pinyin
    
    # 替换带声调的字母为不带声调的
    for i in range(len(pinyin), 0, -1):
        for tone_mark, plain in COMPLEX_TONE_MAP.items():
            if result.endswith(tone_mark):
                result = result[:-len(tone_mark)] + plain
                break
    
    # 单独处理单个带声调字符
    new_result = ''
    for char in result:
        code = unicodedata.name(char, '')
        if 'LATIN' in code:
            # 已经是字母
            new_result += char
        else:
            # 尝试从声调映射表获取
            found = False
            for mark, plain in PINYIN_TONE_MAP.items():
                if char == mark:
                    new_result += plain
                    found = True
                    break
            if not found:
                # 尝试unicodedata
                try:
                    decomp = unicodedata.decomposition(char)
                    if decomp:
                        # 提取基本字母
                        for c in decomp.split():
                            if '0041' <= c <= '005A' or '0061' <= c <= '007A':
                                new_result += chr(int(c, 16))
                except:
                    pass
    
    result = new_result
    
    # 移除所有非字母字符
    result = re.sub(r'[^a-z]', '', result.lower())
    
    return result if result else 'yi', tone


def pinyin_to_numbers(pinyin):
    """拼音字母转数字 (a=1, b=2, ..., z=26)"""
    if not pinyin:
        return [0] * 6
    
    # 移除声调并获取干净拼音
    clean_pinyin, tone = remove_tone(pinyin)
    
    # 转换为数字
    result = []
    for c in clean_pinyin[:6]:  # 最多6位
        result.append(ord(c) - ord('a') + 1)
    
    # 补齐到6位
    while len(result) < 6:
        result.append(0)
    
    return result[:6], tone


# 1. 加载数据
print("\n📖 加载新华字典...")
with open(XINHUA_FILE, 'r', encoding='utf-8') as f:
    xinhua_data = json.load(f)
print(f"   原始汉字数: {len(xinhua_data)}")

print("📊 加载18维向量表...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    vector_data = json.load(f)

existing_chars = {}
for char_info in vector_data['characters']:
    char = char_info['char']
    existing_chars[char] = char_info['vector']

# 2. 测试转换
print("\n🔬 测试拼音转换:")
test_pinyins = ['á', 'ā', 'à', 'ái', 'āi', 'ài', 'ei', 'ou', 'ang', 'eng', 'ong']
for p in test_pinyins:
    nums, tone = pinyin_to_numbers(p)
    print(f"   {p} -> {nums}, 声调={tone}")

# 3. 重新生成整合数据
print("\n🔄 重新整合数据...")

integrated_data = {
    "meta": {
        "name": "汉字向量字库_新华字典版",
        "version": "3.1",
        "total_count": 0,
        "vector_dimensions": 18,
        "description": "汉字18维向量编码表 - 整合新华字典数据（修复版）",
        "data_source": "新华字典 + 扩展向量表"
    },
    "characters": []
}

xinhua_dict = {}
for item in xinhua_data:
    char = item.get('word', '')
    if not char or len(char) != 1:
        continue
    
    pinyin = item.get('pinyin', 'yi1')
    explanation = item.get('explanation', '')
    radicals = item.get('radicals', '')
    strokes = item.get('strokes', '0')
    
    # 转换拼音
    pinyin_nums, tone = pinyin_to_numbers(pinyin)
    
    # 18维向量
    vector = [
        0, 0, 0, 0, 0,  # 四角号码
        0, 0, 0, 0,     # 五笔编码
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
        "pinyin": pinyin,
        "radicals": radicals,
        "strokes": strokes,
        "explanation": explanation_short,
        "vector": vector
    }

integrated_data['characters'] = list(xinhua_dict.values())
integrated_data['meta']['total_count'] = len(integrated_data['characters'])

print(f"   整合后字符数: {len(integrated_data['characters'])}")

# 4. 统计
print("\n📊 修复后统计:")
non_zero_count = 0
for char_info in integrated_data['characters']:
    vec = char_info['vector']
    if any(v != 0 for v in vec[9:15]):
        non_zero_count += 1

print(f"   有拼音编码的字符: {non_zero_count}/{len(integrated_data['characters'])}")

# 声调分布
tone_count = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
for char_info in integrated_data['characters']:
    tone = char_info['vector'][15]
    tone_count[tone] = tone_count.get(tone, 0) + 1

print(f"   声调分布: 平声={tone_count.get(1,0)}, 上声={tone_count.get(2,0)}, 去声={tone_count.get(3,0)}, 入声={tone_count.get(4,0)}")

# 5. 保存
print(f"\n💾 保存: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(integrated_data, f, ensure_ascii=False, indent=2)

print("\n✅ 修复完成!")
