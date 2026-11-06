#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并汉字向量字库和符号向量字库，生成统一嵌入层
支持PyTorch和TensorFlow直接加载使用
"""

import json
import numpy as np

# ============== 配置 ==============
CHARS_FILE = r"D:\desktop\学习结果小文件\汉字向量字库.json"
SYMBOLS_FILE = r"D:\desktop\学习结果小文件\符号向量字库.json"
OUTPUT_DIR = r"D:\desktop\学习结果小文件"

# 特殊token配置
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SOS>": 2,
    "<EOS>": 3
}
PAD_ID = 0
UNK_ID = 1

# ============== 加载数据 ==============
print("正在加载字库文件...")

with open(CHARS_FILE, 'r', encoding='utf-8') as f:
    hanzi_data = json.load(f)

with open(SYMBOLS_FILE, 'r', encoding='utf-8') as f:
    symbol_data = json.load(f)

hanzi_chars = hanzi_data['characters']
symbol_chars = symbol_data['characters']

print(f"汉字数量: {len(hanzi_chars)}")
print(f"符号数量: {len(symbol_chars)}")

# ============== 构建词汇表 ==============
print("\n正在构建词汇表...")

vocab = SPECIAL_TOKENS.copy()
char_to_idx = {}
idx_to_char = {}

# 添加特殊token到映射
for token, idx in SPECIAL_TOKENS.items():
    char_to_idx[token] = idx
    idx_to_char[idx] = token

# 添加汉字 (从索引4开始)
base_idx = len(SPECIAL_TOKENS)
for i, item in enumerate(hanzi_chars):
    char = item['char']
    idx = base_idx + i
    vocab[char] = idx
    char_to_idx[char] = idx
    idx_to_char[idx] = char

# 添加符号 (从汉字之后开始)
symbol_base_idx = base_idx + len(hanzi_chars)
for i, item in enumerate(symbol_chars):
    char = item['char']
    idx = symbol_base_idx + i
    # 避免重复字符
    if char not in vocab:
        vocab[char] = idx
        char_to_idx[char] = idx
        idx_to_char[idx] = char

print(f"总词汇量: {len(vocab)}")
print(f"  - 特殊token: {len(SPECIAL_TOKENS)}")
print(f"  - 汉字: {len(hanzi_chars)}")
print(f"  - 符号: {len(symbol_chars)}")

# ============== 构建嵌入矩阵 ==============
print("\n正在构建嵌入矩阵...")

embedding_dim = 18  # 18维向量
total_vocab_size = len(vocab)

# 初始化嵌入矩阵
embedding_matrix = np.zeros((total_vocab_size, embedding_dim), dtype=np.float32)

# 填充汉字向量
print("填充汉字向量...")
for i, item in enumerate(hanzi_chars):
    char = item['char']
    vector = item['vector']
    idx = char_to_idx[char]
    embedding_matrix[idx] = vector

# 填充符号向量
print("填充符号向量...")
for i, item in enumerate(symbol_chars):
    char = item['char']
    vector = item['vector']
    if char in char_to_idx:
        idx = char_to_idx[char]
        embedding_matrix[idx] = vector

# 处理未登录词 - 使用汉字向量的均值
hanzi_vectors = embedding_matrix[4:base_idx]
mean_vector = np.mean(hanzi_vectors, axis=0)
embedding_matrix[UNK_ID] = mean_vector

print(f"嵌入矩阵形状: {embedding_matrix.shape}")

# ============== 保存词汇表 ==============
vocab_json = {
    "meta": {
        "total_vocab_size": total_vocab_size,
        "embedding_dim": embedding_dim,
        "hanzi_count": len(hanzi_chars),
        "symbol_count": len(symbol_chars),
        "special_tokens": list(SPECIAL_TOKENS.keys())
    },
    "char_to_idx": char_to_idx,
    "idx_to_char": {str(k): v for k, v in idx_to_char.items()}
}

vocab_file = f"{OUTPUT_DIR}\\merged_vocab.json"
with open(vocab_file, 'w', encoding='utf-8') as f:
    json.dump(vocab_json, f, ensure_ascii=False, indent=2)
print(f"\n词汇表已保存: {vocab_file}")

# ============== 保存嵌入矩阵 ==============
embedding_file = f"{OUTPUT_DIR}\\embedding_matrix.npy"
np.save(embedding_file, embedding_matrix)
print(f"嵌入矩阵已保存: {embedding_file}")

# ============== 保存完整合并字库 ==============
merged_chars = []

# 添加特殊token
for token in SPECIAL_TOKENS.keys():
    merged_chars.append({
        "char": token,
        "category": "special",
        "vector": [0] * embedding_dim
    })

# 添加汉字
for item in hanzi_chars:
    merged_chars.append({
        "char": item['char'],
        "pinyin": item.get('pinyin', ''),
        "category": "hanzi",
        "vector": item['vector']
    })

# 添加符号
for item in symbol_chars:
    char = item['char']
    if char not in char_to_idx:
        merged_chars.append({
            "char": char,
            "category": item['category'],
            "vector": item['vector']
        })

merged_data = {
    "meta": {
        "name": "合并向量字库",
        "version": "1.0",
        "total_count": len(merged_chars),
        "vector_dimensions": embedding_dim,
        "description": "汉字+符号统一向量字库，可作为嵌入层使用"
    },
    "characters": merged_chars
}

merged_file = f"{OUTPUT_DIR}\\合并向量字库.json"
with open(merged_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
print(f"合并字库已保存: {merged_file}")

print("\n============== 完成! ==============")
print(f"词汇表大小: {total_vocab_size}")
print(f"嵌入维度: {embedding_dim}")
print(f"嵌入矩阵形状: {embedding_matrix.shape}")
