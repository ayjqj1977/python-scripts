#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成嵌入矩阵脚本
运行此脚本生成 embedding_matrix.npy 文件
"""

import json
import numpy as np
import os

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARS_FILE = os.path.join(SCRIPT_DIR, "汉字向量字库.json")
SYMBOLS_FILE = os.path.join(SCRIPT_DIR, "符号向量字库.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "embedding_matrix.npy")

# 特殊token数量
NUM_SPECIAL_TOKENS = 4  # <PAD>, <UNK>, <SOS>, <EOS>

def generate_embedding_matrix():
    """生成嵌入矩阵"""
    
    print("加载字库文件...")
    
    # 加载汉字字库
    with open(CHARS_FILE, 'r', encoding='utf-8') as f:
        hanzi_data = json.load(f)
    
    # 加载符号字库
    with open(SYMBOLS_FILE, 'r', encoding='utf-8') as f:
        symbol_data = json.load(f)
    
    hanzi_chars = hanzi_data['characters']
    symbol_chars = symbol_data['characters']
    
    print(f"汉字数量: {len(hanzi_chars)}")
    print(f"符号数量: {len(symbol_chars)}")
    
    # 计算总大小
    total_size = NUM_SPECIAL_TOKENS + len(hanzi_chars) + len(symbol_chars)
    embedding_dim = 18
    
    print(f"总词汇量: {total_size}")
    
    # 初始化矩阵
    matrix = np.zeros((total_size, embedding_dim), dtype=np.float32)
    
    # 填充汉字向量 (从索引4开始)
    print("填充汉字向量...")
    for i, item in enumerate(hanzi_chars):
        vector = item['vector']
        idx = NUM_SPECIAL_TOKENS + i
        matrix[idx] = vector
    
    # 填充符号向量
    print("填充符号向量...")
    for i, item in enumerate(symbol_chars):
        vector = item['vector']
        idx = NUM_SPECIAL_TOKENS + len(hanzi_chars) + i
        matrix[idx] = vector
    
    # 设置UNK向量为汉字均值
    hanzi_vectors = matrix[NUM_SPECIAL_TOKENS:NUM_SPECIAL_TOKENS + len(hanzi_chars)]
    matrix[1] = np.mean(hanzi_vectors, axis=0)  # <UNK>
    
    # 保存矩阵
    np.save(OUTPUT_FILE, matrix)
    print(f"\n嵌入矩阵已保存: {OUTPUT_FILE}")
    print(f"矩阵形状: {matrix.shape}")
    print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    generate_embedding_matrix()
