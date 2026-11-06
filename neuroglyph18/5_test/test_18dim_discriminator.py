#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18维向量判别器应用套件
- 判别任务比生成更有希望
- 测试18维的极限在哪里
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter

# 设置路径
WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")

print("📚 加载18维向量数据...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

characters = data['characters']
print(f"   字符数量: {len(characters)}")

# ====== 构建向量索引 ======
char_vectors = {}
char_pinyin = {}
char_sijiao = {}
char_wubi = {}

for char_info in characters:
    char = char_info['char']
    vector = char_info['vector']
    char_vectors[char] = np.array(vector, dtype=np.float32)
    
    # 提取各维度信息
    # 1-5: 四角号码
    char_sijiao[char] = ''.join(map(str, vector[0:5]))
    # 6-9: 五笔
    char_wubi[char] = ''.join(map(str, vector[5:9]))
    # 拼音（需要转换）
    pinyin_code = vector[9:15]
    char_pinyin[char] = ''.join([chr(c + 96) if c > 0 else '0' for c in pinyin_code])

print("✅ 数据加载完成!\n")

# ====== 任务1: 汉字相似度计算 ======
def find_similar_chars(target_char, top_k=10):
    """基于18维向量找相似汉字"""
    if target_char not in char_vectors:
        print(f"   找不到字符: {target_char}")
        return
    
    target_vec = char_vectors[target_char]
    similarities = []
    
    for char, vec in char_vectors.items():
        if char == target_char:
            continue
        
        # 余弦相似度
        cos_sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-8)
        similarities.append((char, cos_sim))
    
    # 排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   与 '{target_char}' 最相似的 {top_k} 个字:")
    print(f"   四角:{char_sijiao.get(target_char,'')} 五笔:{char_wubi.get(target_char,'')} 拼音:{char_pinyin.get(target_char,'')}")
    print("   " + "-"*40)
    for char, sim in similarities[:top_k]:
        print(f"   {char} (相似度:{sim:.3f}) 四角:{char_sijiao.get(char,'')} 五笔:{char_wubi.get(char,'')}")

# ====== 任务2: 拼音匹配 ======
def find_by_pinyin(pinyin, top_k=10):
    """基于拼音找汉字"""
    matches = []
    
    for char, py in char_pinyin.items():
        # 模糊匹配
        if pinyin in py or py in pinyin:
            matches.append(char)
    
    if not matches:
        # 找首字母相同
        if pinyin:
            first_letter = pinyin[0]
            matches = [c for c, py in char_pinyin.items() if py and py[0] == first_letter]
    
    print(f"\n   拼音 '{pinyin}' 匹配的汉字 ({len(matches)}个):")
    print("   " + ",".join(matches[:top_k]))
    if len(matches) > top_k:
        print(f"   ... 还有 {len(matches) - top_k} 个")

# ====== 任务3: 分类任务 - 声调分类 ======
def analyze_tone_distribution():
    """分析声调分布"""
    tones = Counter()
    tone_chars = {1: [], 2: [], 3: [], 4: []}
    
    for char_info in characters:
        vector = char_info['vector']
        tone = vector[15]  # 第16位是声调
        if 1 <= tone <= 4:
            tones[tone] += 1
            tone_chars[tone].append(char_info['char'])
    
    print("\n   声调分布:")
    tone_names = {1: '平声', 2: '上声', 3: '去声', 4: '入声'}
    for tone in [1, 2, 3, 4]:
        print(f"   {tone_names[tone]}: {tones[tone]} 字")
        print(f"   示例: {','.join(tone_chars[tone][:10])}")

# ====== 任务4: 聚类分析 - 找同一韵母字 ======
def find_rhyme_chars(rhyme_char, top_k=15):
    """找同韵母字（四角号码相近）"""
    if rhyme_char not in char_sijiao:
        return
    
    target_sijiao = char_sijiao[rhyme_char]
    
    # 找四角号码相近的
    matches = []
    for char, sj in char_sijiao.items():
        if char == rhyme_char:
            continue
        # 前3位相同
        if len(sj) >= 3 and len(target_sijiao) >= 3 and sj[:3] == target_sijiao[:3]:
            matches.append((char, sj))
    
    print(f"\n   与 '{rhyme_char}' 四角相近的字 ({len(matches)}个):")
    print(f"   目标: {target_sijiao}")
    for char, sj in matches[:top_k]:
        print(f"   {char}: {sj}")

# ====== 任务5: 判别器测试 ======
def test_classification():
    """测试18维向量的分类能力"""
    print("\n" + "="*50)
    print("🧪 18维向量分类能力测试")
    print("="*50)
    
    # 测试1: 同音字分类
    print("\n【测试1】同拼音汉字聚类")
    test_pinyins = ['yi', 'shi', 'li', 'wang', 'zhang']
    for py in test_pinyins:
        matches = [c for c, p in char_pinyin.items() if py in p]
        print(f"   拼音含'{py}': {len(matches)}字 - {','.join(matches[:8])}")
    
    # 测试2: 相似字聚类
    print("\n【测试2】结构相似字聚类")
    test_chars = ['林', '森', '吕', '品', '鑫']
    for tc in test_chars:
        if tc in char_vectors:
            target = char_vectors[tc]
            # 找距离最近的
            distances = []
            for char, vec in char_vectors.items():
                if char != tc:
                    dist = np.linalg.norm(target - vec)
                    distances.append((char, dist))
            distances.sort(key=lambda x: x[1])
            print(f"   '{tc}' 最近邻: {','.join([d[0] for d in distances[:5]])}")
    
    # 测试3: 类型判别（数字vs汉字）
    print("\n【测试3】字符类型判别")
    # 18维编码第17位是类型
    chinese_chars = [c for c in characters if c['vector'][16] == 0]
    digits = [c for c in characters if c['vector'][16] == 1]
    letters = [c for c in characters if c['vector'][16] == 2]
    symbols = [c for c in characters if c['vector'][16] == 3]
    print(f"   汉字: {len(chinese_chars)}")
    print(f"   数字: {len(digits)}")
    print(f"   字母: {len(letters)}")
    print(f"   符号: {len(symbols)}")
    
    # 测试4: 零声母判别
    print("\n【测试4】声调判别能力")
    tone_1 = [c['char'] for c in characters if c['vector'][15] == 1]
    tone_4 = [c['char'] for c in characters if c['vector'][15] == 4]
    print(f"   平声(1): {len(tone_1)} 字 - 示例: {','.join(tone_1[:10])}")
    print(f"   入声(4): {len(tone_4)} 字 - 示例: {','.join(tone_4[:10])}")

# ====== 主界面 ======
def main():
    print("\n" + "="*50)
    print("🔬 18维向量判别器测试套件")
    print("="*50)
    
    # 先运行完整测试
    test_classification()
    
    # 交互测试
    print("\n" + "="*50)
    print("🎯 交互测试")
    print("="*50)
    
    while True:
        print("\n可选任务:")
        print("  1. 找相似字")
        print("  2. 拼音查字")
        print("  3. 找同韵字")
        print("  4. 退出")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            char = input("输入目标汉字: ").strip()
            if char:
                find_similar_chars(char)
        elif choice == '2':
            pinyin = input("输入拼音: ").strip()
            if pinyin:
                find_by_pinyin(pinyin)
        elif choice == '3':
            char = input("输入参考汉字: ").strip()
            if char:
                find_rhyme_chars(char)
        elif choice == '4':
            print("再见!")
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    main()
