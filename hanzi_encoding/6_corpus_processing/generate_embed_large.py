# -*- coding: utf-8 -*-
"""
为新词表生成18维嵌入矩阵
"""
import json
import numpy as np
import os

WORK_DIR = r'D:\desktop\学习结果小文件'

print("=== 生成新嵌入矩阵 ===\n")

# 读取原始词表的嵌入
original_embed = np.load(os.path.join(WORK_DIR, 'embedding_matrix.npy'))
print(f"原始嵌入矩阵: {original_embed.shape}")

# 读取原始词表
with open(os.path.join(WORK_DIR, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    old_vocab = json.load(f)
old_char_to_idx = old_vocab['char_to_idx']

# 读取新词表
with open(os.path.join(WORK_DIR, 'vocab_large.json'), 'r', encoding='utf-8') as f:
    new_vocab = json.load(f)
new_char_to_idx = new_char_to_idx = new_vocab['char_to_idx']

new_vocab_size = len(new_char_to_idx)
print(f"新词表大小: {new_vocab_size}")

# 读取原始汉字向量数据（用于获取编码规则）
with open(os.path.join(WORK_DIR, '汉字向量字库.json'), 'r', encoding='utf-8') as f:
    char_vectors = json.load(f)

print(f"汉字向量数据: {len(char_vectors)} 个字符")

# 创建新嵌入矩阵
new_embed = np.zeros((new_vocab_size, 18), dtype=np.float32)

# 复制原有嵌入
copied = 0
for char, new_idx in new_char_to_idx.items():
    if char in old_char_to_idx:
        old_idx = old_char_to_idx[char]
        new_embed[new_idx] = original_embed[old_idx]
        copied += 1

print(f"复制原有嵌入: {copied}")

# 为新字符生成随机嵌入（初始化）
# 注意：这些新字符需要后续训练来学习有意义的表示
np.random.seed(42)
for char, idx in new_char_to_idx.items():
    if new_embed[idx].sum() == 0:  # 未复制的
        # 初始化为小的随机值
        new_embed[idx] = np.random.randn(18) * 0.1

print(f"新字符已初始化")

# 保存
output_path = os.path.join(WORK_DIR, 'embedding_large.npy')
np.save(output_path, new_embed)

print(f"\n✅ 已保存到: embedding_large.npy")
print(f"   形状: {new_embed.shape}")
