# -*- coding: utf-8 -*-
"""
hzk32_reader.py — HZK32 点阵字库读取与解析
===========================================
HZK32 是 GB2312 编码的 32×32 点阵字库。
每个汉字占 128 字节（32×32 位），按区位码顺序排列。

文件格式：
  - 区码范围: B0-F7（72区，每区94字，共6768个汉字）
  - 每字: 128 字节，32×32 = 1024 位
  - 偏移公式: offset = ((区码 - 0xA1) × 94 + (位码 - 0xA1)) × 128
  - 文件大小: 866,304 字节 (~846 KB)

用法:
    reader = HZK32Reader("HZK32")
    vector = reader.get_glyph_vector("汉")    # 返回 1024 维 0/1 numpy 数组
    matrix = reader.get_glyph_matrix("汉字")  # 返回 32×32 numpy 矩阵
    reader.print_glyph("汉")                  # 终端打印点阵字符画
"""

import os
import numpy as np


class HZK32Reader:
    """HZK32 点阵字库读取器"""

    def __init__(self, font_path: str = "HZK32"):
        """
        Args:
            font_path: HZK32 字库文件路径
        """
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"HZK32 字库文件未找到: {font_path}")

        self.font_path = font_path
        self.file_size = os.path.getsize(font_path)
        self.EXPECTED_SIZE = self.file_size  # 动态检测实际文件大小

    def _char_to_gb2312(self, char: str) -> tuple:
        """汉字转 GB2312 编码 (区码, 位码)"""
        if len(char) != 1:
            raise ValueError(f"请输入单个汉字，收到: '{char}'")

        try:
            gb_bytes = char.encode("gb2312")
        except UnicodeEncodeError:
            raise ValueError(f"字符 '{char}' 不在 GB2312 编码范围内")

        if len(gb_bytes) != 2:
            raise ValueError(f"字符 '{char}' 不是 GB2312 双字节编码")

        qu = gb_bytes[0]  # 区码
        wei = gb_bytes[1]  # 位码

        # 检查是否在有效汉字范围内 (B0-F7)
        if qu < 0xB0 or qu > 0xF7:
            raise ValueError(
                f"字符 '{char}' 区码 0x{qu:02X} 不在汉字区 B0-F7 范围内"
            )

        return qu, wei

    def get_offset(self, char: str) -> int:
        """计算字符在 HZK32 文件中的字节偏移"""
        qu, wei = self._char_to_gb2312(char)
        offset = ((qu - 0xA1) * 94 + (wei - 0xA1)) * 128
        return offset

    def get_glyph_bytes(self, char: str) -> bytes:
        """读取字符的 128 字节原始点阵数据"""
        offset = self.get_offset(char)

        with open(self.font_path, "rb") as f:
            f.seek(offset)
            data = f.read(128)

        if len(data) != 128:
            raise IOError(f"读取数据不足 128 字节 (实际 {len(data)})")

        return data

    def get_glyph_vector(self, char: str) -> np.ndarray:
        """
        获取字符的 32×32 点阵展平向量（1024 维 0/1）
        行优先展平: 第0行0-31列, 第1行0-31列, ..., 第31行0-31列
        """
        data = self.get_glyph_bytes(char)
        matrix = np.zeros((32, 32), dtype=np.uint8)

        for row in range(32):
            for col_byte in range(4):  # 每行 4 字节 (32 位)
                byte_idx = row * 4 + col_byte
                byte_val = data[byte_idx]

                for bit in range(8):
                    col = col_byte * 8 + bit
                    # 高位在前: bit7 对应 col_byte*8+0
                    if byte_val & (0x80 >> bit):
                        matrix[row, col] = 1

        return matrix.flatten()  # 1024 维

    def get_glyph_matrix(self, char: str) -> np.ndarray:
        """获取字符的 32×32 点阵矩阵（32×32, 0/1）"""
        return self.get_glyph_vector(char).reshape(32, 32)

    def print_glyph(self, char: str, pixel_on: str = "█", pixel_off: str = " "):
        """终端打印字符点阵（黑底白字风格）"""
        matrix = self.get_glyph_matrix(char)
        print(f"字符: {char} (U+{ord(char):04X})")
        print("-" * 32)
        for row in range(32):
            line = ""
            for col in range(32):
                line += pixel_on if matrix[row, col] else pixel_off
            print(line)
        print("-" * 32)
        # 统计信息
        active = np.sum(matrix)
        print(f"活跃点数: {active}/1024 ({active/1024*100:.1f}%)")

    def get_batch_vectors(self, chars: list) -> np.ndarray:
        """批量获取多个字符的 1024 维向量，返回 (N, 1024) 数组"""
        vectors = []
        for char in chars:
            try:
                vec = self.get_glyph_vector(char)
                vectors.append(vec)
            except ValueError as e:
                print(f"跳过 '{char}': {e}")
                vectors.append(np.zeros(1024, dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

    def __repr__(self):
        return f"HZK32Reader(path='{self.font_path}', size={self.file_size})"


# ============================================================
# 命令行演示
# ============================================================
if __name__ == "__main__":
    reader = HZK32Reader("HZK32")
    print(reader)
    print()

    # 演示几个典型汉字
    demo_chars = ["汉", "字", "中", "国", "点", "阵"]
    for ch in demo_chars:
        reader.print_glyph(ch)
        print()

    # 演示形近字
    print("=== 形近字对比 ===")
    similar_groups = [
        ["己", "已", "巳"],
        ["人", "入", "八"],
        ["未", "末"],
        ["土", "士"],
        ["干", "千", "于"],
    ]
    for group in similar_groups:
        vectors = reader.get_batch_vectors(group)
        # 计算两两余弦相似度
        from numpy.linalg import norm

        print(f"\n{' '.join(group)}:")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                v1, v2 = vectors[i], vectors[j]
                cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)
                print(f"  {group[i]} vs {group[j]}: cos_sim = {cos_sim:.4f}")
