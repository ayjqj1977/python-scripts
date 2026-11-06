# -*- coding: utf-8 -*-
"""
检查语料提取问题
"""
import json
import os

corpus_dir = r'D:\desktop\学习结果小文件\corpus'

# 检查诗经结构
print("=== 诗经 ===")
with open(os.path.join(corpus_dir, 'shijing_shijing.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"总数: {len(data)}")
print(f"示例: {data[0]}")

print("\n=== 元曲 ===")
with open(os.path.join(corpus_dir, 'yuanqu_yuanqu.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"总数: {len(data)}")
print(f"示例: {data[0]}")

print("\n=== 全唐诗 ===")
with open(os.path.join(corpus_dir, 'quantangshi_poet.song.0.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"总数: {len(data)}")
print(f"示例: {data[0]}")

print("\n=== 宋词 ===")
with open(os.path.join(corpus_dir, 'songci_ci.song.0.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"总数: {len(data)}")
print(f"示例: {data[0]}")
