#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整生成嵌入层所有文件 - 修复版
"""

import json
import numpy as np

def main():
    # 配置
    CHARS_FILE = r"D:\desktop\学习结果小文件\汉字向量字库.json"
    SYMBOLS_FILE = r"D:\desktop\学习结果小文件\符号向量字库.json"
    VOCAB_FILE = r"D:\desktop\学习结果小文件\merged_vocab.json"
    EMBED_FILE = r"D:\desktop\学习结果小文件\embedding_matrix.npy"
    
    print("加载字库...")
    
    with open(CHARS_FILE, 'r', encoding='utf-8') as f:
        hanzi_data = json.load(f)
    with open(SYMBOLS_FILE, 'r', encoding='utf-8') as f:
        symbol_data = json.load(f)
    
    hanzi_chars = hanzi_data['characters']
    symbol_chars = symbol_data['characters']
    
    print(f"汉字数量: {len(hanzi_chars)}")
    print(f"符号数量: {len(symbol_chars)}")
    
    # 特殊token
    special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    num_special = len(special_tokens)
    
    # 构建词汇表 - 用字典收集唯一的字符
    char_to_idx = {}
    idx_to_char = {}
    
    # 先添加特殊token
    for i, token in enumerate(special_tokens):
        char_to_idx[token] = i
        idx_to_char[str(i)] = token
    
    # 添加汉字
    hanzi_start = num_special
    for i, item in enumerate(hanzi_chars):
        char = item['char']
        if char not in char_to_idx:  # 避免重复
            idx = hanzi_start + len([c for c in char_to_idx if char_to_idx[c] >= hanzi_start])
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[str(idx)] = char
    
    # 添加符号 (跳过已存在的字符)
    symbol_start = len(char_to_idx)
    added_count = 0
    for item in symbol_chars:
        char = item['char']
        if char not in char_to_idx:
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[str(idx)] = char
            added_count += 1
    
    print(f"添加的符号数量: {added_count}")
    print(f"总词汇量: {len(char_to_idx)}")
    
    # 保存词汇表
    vocab_data = {
        "meta": {
            "total_vocab_size": len(char_to_idx),
            "embedding_dim": 18,
            "hanzi_count": len(hanzi_chars),
            "symbol_count": added_count,
            "special_tokens": special_tokens
        },
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }
    
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"词汇表已保存: {VOCAB_FILE}")
    
    # 生成嵌入矩阵
    print("\n生成嵌入矩阵...")
    
    total_size = len(char_to_idx)
    embedding_dim = 18
    matrix = np.zeros((total_size, embedding_dim), dtype=np.float32)
    
    # 填充汉字 - 使用原始顺序
    for i, item in enumerate(hanzi_chars):
        char = item['char']
        vector = item['vector']
        if char in char_to_idx:
            idx = char_to_idx[char]
            matrix[idx] = vector
    
    # 填充符号 - 使用原始顺序
    for item in symbol_chars:
        char = item['char']
        vector = item['vector']
        if char in char_to_idx:
            idx = char_to_idx[char]
            matrix[idx] = vector
    
    # UNK使用汉字均值
    hanzi_indices = [char_to_idx[c] for c in char_to_idx if c not in special_tokens and c not in [s['char'] for s in symbol_chars if char_to_idx.get(s['char'], -1) < symbol_start]]
    if hanzi_indices:
        hanzi_vectors = matrix[hanzi_indices]
        matrix[1] = np.mean(hanzi_vectors, axis=0)
    
    # 保存矩阵
    np.save(EMBED_FILE, matrix)
    print(f"嵌入矩阵已保存: {EMBED_FILE}")
    print(f"矩阵形状: {matrix.shape}")
    print("\n完成!")

if __name__ == "__main__":
    main()
