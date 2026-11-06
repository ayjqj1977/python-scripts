# -*- coding: utf-8 -*-
"""
shape_sim_test.py — 形近字聚类快速验证
======================================
基于 HZK32 点阵向量的余弦相似度，验证常见形近字对的相似性排名。

流程:
  1. 从 char_to_idx.json 加载词表，取前 500 个高频字
  2. 通过 hzk32_reader 读取各自的 1024 维点阵向量
  3. 计算两两余弦相似度矩阵
  4. 重点打印形近字对排名，并与随机基线对比

重点检查字形对:
  己/已/巳, 人/入/八, 未/末, 土/士, 干/千/于, 天/夫, 大/太/犬
"""

import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hzk32_reader import HZK32Reader


def load_char_to_idx(json_path: str) -> dict:
    """加载 char_to_idx.json 词表"""
    with open(json_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    # 按 index 排序，取前 N 个高频字（index 越小频率越高）
    sorted_chars = sorted(char_to_idx.items(), key=lambda x: x[1])
    # 去掉 <UNK>
    sorted_chars = [(ch, idx) for ch, idx in sorted_chars if ch != "<UNK>"]
    return sorted_chars


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2 归一化"""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    计算归一化后的余弦相似度矩阵
    vectors: (N, 1024)
    返回: (N, N) 对称相似度矩阵
    """
    # L2 归一化后，点积 = 余弦相似度
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T
    return sim_matrix


def find_top_similar(sim_matrix: np.ndarray, chars: list, top_k: int = 20) -> list:
    """
    找出相似度最高的 N 个字符对（排除自相似）
    返回: [(char_a, char_b, similarity), ...]
    """
    N = len(chars)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((chars[i], chars[j], sim_matrix[i, j]))

    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


def check_specific_groups(
    reader: HZK32Reader,
    groups: list,
    sim_matrix: np.ndarray = None,
    char_list: list = None,
    char_to_idx_map: dict = None,
):
    """
    检查特定形近字组的相似度。
    优先从预计算的 sim_matrix 查表；不在高频字范围内时，降级为从 HZK32 实时读取并计算。
    """
    print("\n" + "=" * 70)
    print("重点形近字组检查")
    print("=" * 70)

    results = []

    for group in groups:
        print(f"\n组: {' '.join(group)}")

        # 尝试从预计算矩阵取值
        group_sims = {}
        missing_chars = set()

        for ch in group:
            if char_to_idx_map is not None and ch in char_to_idx_map:
                group_sims[ch] = char_list.index(ch)
            elif char_to_idx_map is not None:
                missing_chars.add(ch)

        if missing_chars or char_to_idx_map is None:
            if missing_chars:
                print(f"  (字符 {', '.join(sorted(missing_chars))} 不在高频字范围，从 HZK32 实时读取)")
            # 实时读取整个组（统一计算，保证一致性）
            vectors = reader.get_batch_vectors(group)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    v1 = vectors[i]
                    v2 = vectors[j]
                    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
                    print(f"  {group[i]} vs {group[j]}: 余弦相似度 = {sim:.6f}")
                    results.append((group[i], group[j], sim))
        else:
            # 全部在高频字范围内，从 sim_matrix 查表
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    idx_i = char_list.index(group[i])
                    idx_j = char_list.index(group[j])
                    sim = sim_matrix[idx_i, idx_j]
                    print(f"  {group[i]} vs {group[j]}: 余弦相似度 = {sim:.6f}")
                    results.append((group[i], group[j], sim))

    return results


def main():
    # =========================================
    # 配置
    # =========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, "HZK32")
    json_path = os.path.join(script_dir, "char_to_idx.json")

    TOP_N_CHARS = 500  # 取前 500 个高频字

    # =========================================
    # 1. 加载词表
    # =========================================
    print("=" * 70)
    print("shape_sim_test.py — 形近字聚类验证")
    print("=" * 70)
    print(f"\n[1/4] 加载词表: {json_path}")

    sorted_chars = load_char_to_idx(json_path)
    total_chars = len(sorted_chars)
    print(f"  词表总数: {total_chars} 个字符（不含 <UNK>）")

    # 取前 TOP_N_CHARS 个高频字
    top_chars = [ch for ch, idx in sorted_chars[:TOP_N_CHARS]]
    print(f"  选取前 {TOP_N_CHARS} 个高频字进行分析")

    # =========================================
    # 2. 加载 HZK32 字库
    # =========================================
    print(f"\n[2/4] 加载 HZK32 字库: {font_path}")
    reader = HZK32Reader(font_path)
    print(f"  {reader}")

    # =========================================
    # 3. 批量提取点阵向量
    # =========================================
    print(f"\n[3/4] 提取 {TOP_N_CHARS} 个字符的 1024 维点阵向量...")
    t0 = time.time()

    vectors = reader.get_batch_vectors(top_chars)
    elapsed = time.time() - t0
    print(f"  完成，耗时 {elapsed:.2f}s")
    print(f"  向量矩阵形状: {vectors.shape}")

    # 统计信息
    active_counts = vectors.sum(axis=1)
    print(f"  平均活跃点数: {active_counts.mean():.1f} / 1024")
    print(f"  最少活跃点: {active_counts.min():.0f}, 最多活跃点: {active_counts.max():.0f}")

    # =========================================
    # 4. 相似度计算与分析
    # =========================================
    print(f"\n[4/4] 计算余弦相似度矩阵...")
    t0 = time.time()

    # L2 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms

    # 余弦相似度矩阵
    sim_matrix = normalized @ normalized.T
    elapsed = time.time() - t0
    print(f"  完成，耗时 {elapsed:.2f}s")
    print(f"  相似度矩阵形状: {sim_matrix.shape}")

    # 统计分布
    triu_indices = np.triu_indices(TOP_N_CHARS, k=1)
    all_sims = sim_matrix[triu_indices]
    print(f"\n  相似度分布统计:")
    print(f"    均值:   {all_sims.mean():.6f}")
    print(f"    标准差: {all_sims.std():.6f}")
    print(f"    中位数: {np.median(all_sims):.6f}")
    print(f"    最小值: {all_sims.min():.6f}")
    print(f"    最大值: {all_sims.max():.6f}")
    print(f"    前 1% 阈值: {np.percentile(all_sims, 99):.6f}")
    print(f"    前 0.1% 阈值: {np.percentile(all_sims, 99.9):.6f}")

    # =========================================
    # 4a. 全局 Top-20 最相似字对
    # =========================================
    print("\n" + "=" * 70)
    print("全局 Top-20 最相似字对（前500高频字中）")
    print("=" * 70)

    top_pairs = find_top_similar(sim_matrix, top_chars, top_k=20)
    for rank, (a, b, sim) in enumerate(top_pairs, 1):
        marker = " ★" if sim > 0.95 else ""
        print(f"  {rank:2d}. {a} vs {b}: {sim:.6f}{marker}")

    # =========================================
    # 4b. 重点形近字组检查
    # =========================================
    focus_groups = [
        ["己", "已", "巳"],
        ["人", "入", "八"],
        ["未", "末"],
        ["土", "士"],
        ["干", "千", "于"],
        ["天", "夫"],
        ["大", "太", "犬"],
        # 额外添加常见形近字
        ["日", "曰"],
        ["王", "玉", "主"],
        ["力", "刀"],
        ["几", "凡"],
        ["乌", "鸟"],
        ["问", "间", "闻"],
        ["风", "凤"],
        ["田", "由", "甲", "申"],
    ]

    # 建立字符到索引的映射
    char_to_idx_map = {ch: i for i, ch in enumerate(top_chars)}

    group_results = check_specific_groups(
        reader, focus_groups, sim_matrix, top_chars, char_to_idx_map
    )

    # =========================================
    # 4c. 汇总报告
    # =========================================
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)

    print(f"\n  分析字符数: {TOP_N_CHARS}")
    print(f"  向量维度: 1024 (32×32 点阵)")
    print(f"  相似度均值: {all_sims.mean():.6f}")
    print(f"  相似度标准差: {all_sims.std():.6f}")

    # 形近字组统计
    if group_results:
        group_sims = [s for _, _, s in group_results]
        print(f"\n  形近字组数: {len(focus_groups)}")
        print(f"  形近字对平均相似度: {np.mean(group_sims):.6f}")
        print(f"  形近字对最高相似度: {np.max(group_sims):.6f}")
        print(f"  全局 top-20 平均相似度: {np.mean([s for _, _, s in top_pairs]):.6f}")

        print(f"\n  验证结论:")
        print(f"    - 全局随机基线相似度: {all_sims.mean():.6f}")
        print(f"    - 形近字组相似度:     {np.mean(group_sims):.6f}")

        ratio = np.mean(group_sims) / (all_sims.mean() + 1e-8)
        print(f"    - 比例: {ratio:.2f}x")
        if ratio > 1.5:
            print(f"    ✓ 形近字组相似度显著高于随机基线 ({ratio:.2f}x)")
            print(f"    ✓ 32×32 点阵有效捕捉了字形结构信息")
        elif ratio > 1.1:
            print(f"    ○ 形近字组相似度略高于随机基线 ({ratio:.2f}x)")
        else:
            print(f"    ⚠ 形近字组相似度与随机基线差异不大")
            print(f"      备注: 当前 HZK32 为合成占位文件，非真实点阵数据")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
