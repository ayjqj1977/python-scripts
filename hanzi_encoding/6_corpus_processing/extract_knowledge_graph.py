#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取知识图谱CSV中的中文文本，构建大规模语料库
"""

import os
import csv
import json
from pathlib import Path

# 知识图谱根目录
KG_DIR = r"D:\desktop\学习结果小文件\onegraphv2-\OneGraphV2-中文"

# 输出文件
OUTPUT_FILE = r"D:\desktop\学习结果小文件\corpus_knowledge_graph.txt"

def extract_text_from_csv(csv_path):
    """从CSV中提取Head和Tail列的中文文本"""
    texts = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                head = row.get('Head', '').strip()
                tail = row.get('Tail', '').strip()
                if head:
                    texts.append(head)
                if tail:
                    texts.append(tail)
    except Exception as e:
        print(f"  错误: {e}")
    return texts

def main():
    print("📚 开始提取知识图谱文本...")
    
    all_texts = []
    csv_files = []
    
    # 遍历所有CSV文件
    for root, dirs, files in os.walk(KG_DIR):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for i, csv_path in enumerate(csv_files):
        rel_path = os.path.relpath(csv_path, KG_DIR)
        print(f"[{i+1}/{len(csv_files)}] 处理: {rel_path}")
        
        texts = extract_text_from_csv(csv_path)
        all_texts.extend(texts)
        print(f"    提取到 {len(texts)} 条文本")
    
    print(f"\n✅ 总共提取 {len(all_texts)} 条文本")
    
    # 去重
    unique_texts = list(set(all_texts))
    print(f"去重后: {len(unique_texts)} 条")
    
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for text in unique_texts:
            f.write(text + '\n')
    
    print(f"📄 已保存到: {OUTPUT_FILE}")
    
    # 统计信息
    lengths = [len(t) for t in unique_texts]
    print(f"\n📊 文本长度统计:")
    print(f"   最短: {min(lengths)} 字符")
    print(f"   最长: {max(lengths)} 字符")
    print(f"   平均: {sum(lengths)/len(lengths):.1f} 字符")

if __name__ == "__main__":
    main()
