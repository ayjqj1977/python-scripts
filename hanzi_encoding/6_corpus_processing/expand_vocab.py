# -*- coding: utf-8 -*-
"""
扩大词表 - 正确版本
"""
import json
import glob
import os
import collections

corpus_dir = r'D:\desktop\学习结果小文件\corpus'
WORK_DIR = r'D:\desktop\学习结果小文件'

print("=== 扩大词表 ===\n")

# 收集所有字符
all_chars = []
file_count = 0

for f in glob.glob(os.path.join(corpus_dir, '*.json')):
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 尝试多个可能的字段
                    for key in ['paragraphs', 'content', 'text']:
                        if key in item:
                            content = item[key]
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, str):
                                        all_chars.extend(c)
                            elif isinstance(content, str):
                                all_chars.extend(content)
                            break
        file_count += 1
    except Exception as e:
        print(f"  警告: {os.path.basename(f)} - {e}")

print(f"处理了 {file_count} 个文件")
print(f"收集到 {len(all_chars)} 个字符")

# 统计并排序（按频率）
counter = collections.Counter(all_chars)
sorted_chars = [c for c, _ in counter.most_common()]

print(f"独特字符: {len(sorted_chars)}")

# 读取原词表
with open(os.path.join(WORK_DIR, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    old_vocab = json.load(f)

old_char_to_idx = old_vocab['char_to_idx']
print(f"原词表: {len(old_char_to_idx)}")

# 创建新词表
new_char_to_idx = {}
new_idx_to_char = {}

# 特殊字符
special = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']
for i, char in enumerate(special):
    new_char_to_idx[char] = i
    new_idx_to_char[str(i)] = char

# 从原词表添加
next_idx = len(special)
for char, idx in old_char_to_idx.items():
    if char not in special:
        new_char_to_idx[char] = next_idx
        new_idx_to_char[str(next_idx)] = char
        next_idx += 1

# 添加新字符（按频率）
added = 0
for char in sorted_chars:
    if char not in new_char_to_idx and char.strip() and len(char.strip()) > 0:
        new_char_to_idx[char] = next_idx
        new_idx_to_char[str(next_idx)] = char
        next_idx += 1
        added += 1

print(f"新增字符: {added}")
print(f"新词表大小: {len(new_char_to_idx)}")

# 保存新词表
new_vocab = {
    'char_to_idx': new_char_to_idx,
    'idx_to_char': new_idx_to_char,
    'description': '扩大后的词表，包含所有语料字符'
}

with open(os.path.join(WORK_DIR, 'vocab_large.json'), 'w', encoding='utf-8') as f:
    json.dump(new_vocab, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已保存到: vocab_large.json")
