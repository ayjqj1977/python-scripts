#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并所有语料：诗词 + 知识图谱 + 汉字词语成语歇后语
"""

import json
import os

# 输入文件
POETRY_FILE = r"D:\desktop\学习结果小文件\corpus_training.txt"
KG_FILE = r"D:\desktop\学习结果小文件\corpus_knowledge_graph.txt"
HANZI_FILE = r"D:\desktop\学习结果小文件\7_训练语料\汉字.json"
CIYU_FILE = r"D:\desktop\学习结果小文件\7_训练语料\词语.json"
XIEHOUYU_FILE = r"D:\desktop\学习结果小文件\7_训练语料\歇后语.json"
CHENGYU_FILE = r"D:\desktop\学习结果小文件\7_训练语料\成语.json"

# 输出文件
OUTPUT_FILE = r"D:\desktop\学习结果小文件\corpus_combined.txt"

def read_lines(filepath):
    """读取文本文件的所有行"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取 {filepath} 失败: {e}")
        return []

def read_json_lines(filepath):
    """读取JSON数组文件，每行一个JSON对象"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data if item]
            return []
    except Exception as e:
        print(f"读取 {filepath} 失败: {e}")
        return []

def read_json_dict(filepath):
    """读取JSON对象文件，提取所有值"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # 提取所有中文文本
                texts = []
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        texts.append(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and v:
                                texts.append(v)
                return texts
            return []
    except Exception as e:
        print(f"读取 {filepath} 失败: {e}")
        return []

def main():
    print("📚 开始合并语料...")
    
    all_texts = []
    
    # 1. 诗词语料
    print("📜 读取诗词语料...")
    poetry = read_lines(POETRY_FILE)
    print(f"   诗词: {len(poetry)} 条")
    all_texts.extend(poetry)
    
    # 2. 知识图谱语料
    print("🔗 读取知识图谱语料...")
    kg = read_lines(KG_FILE)
    print(f"   知识图谱: {len(kg)} 条")
    all_texts.extend(kg)
    
    # 3. 汉字JSON
    print("✍️ 读取汉字...")
    hanzi = read_json_dict(HANZI_FILE)
    print(f"   汉字: {len(hanzi)} 条")
    all_texts.extend(hanzi)
    
    # 4. 词语JSON
    print("📝 读取词语...")
    ciyu = read_json_dict(CIYU_FILE)
    print(f"   词语: {len(ciyu)} 条")
    all_texts.extend(ciyu)
    
    # 5. 歇后语JSON
    print("🎭 读取歇后语...")
    xiehouyu = read_json_lines(XIEHOUYU_FILE)
    print(f"   歇后语: {len(xiehouyu)} 条")
    all_texts.extend(xiehouyu)
    
    # 6. 成语JSON
    print("🏆 读取成语...")
    chengyu = read_json_lines(CHENGYU_FILE)
    print(f"   成语: {len(chengyu)} 条")
    all_texts.extend(chengyu)
    
    print(f"\n📊 合并前总计: {len(all_texts)} 条")
    
    # 去重
    unique_texts = list(set(all_texts))
    print(f"去重后: {len(unique_texts)} 条")
    
    # 按长度排序（从长到短，便于后续处理）
    unique_texts.sort(key=len, reverse=True)
    
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for text in unique_texts:
            f.write(text + '\n')
    
    print(f"✅ 已保存到: {OUTPUT_FILE}")
    
    # 统计信息
    lengths = [len(t) for t in unique_texts]
    print(f"\n📊 文本长度统计:")
    print(f"   最短: {min(lengths)} 字符")
    print(f"   最长: {max(lengths)} 字符")
    print(f"   平均: {sum(lengths)/len(lengths):.1f} 字符")

if __name__ == "__main__":
    main()
