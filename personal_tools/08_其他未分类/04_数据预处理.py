#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据准备工具
把任意文本文件转换为训练格式
"""

import json
import sys
import os

def prepare_training_data(
    text_file: str,
    vocab_file: str,
    output_file: str = None,
    min_length: int = 100
):
    """
    准备训练数据
    
    Args:
        text_file: 原始文本文件路径 (UTF-8编码)
        vocab_file: 词汇表文件
        output_file: 输出文件路径 (默认与输入同目录)
        min_length: 最小文本长度
    """
    
    # 加载词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_to_idx = vocab_data['char_to_idx']
    unk_idx = char_to_idx.get('<UNK>', 1)
    
    # 读取文本
    print(f"读取文本文件: {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 清理文本
    text = clean_text(text)
    
    # 编码
    print("编码中...")
    encoded = []
    for char in text:
        idx = char_to_idx.get(char, unk_idx)
        encoded.append(idx)
    
    print(f"原始字符数: {len(text)}")
    print(f"有效token数: {len(encoded)}")
    print(f"UNK数量: {encoded.count(unk_idx)}")
    
    # 统计
    unique_chars = len(set(text))
    print(f"独特字符数: {unique_chars}")
    
    # 保存
    if output_file is None:
        output_file = text_file.replace('.txt', '_encoded.npy')
    
    import numpy as np
    np.save(output_file, np.array(encoded, dtype=np.int32))
    print(f"已保存: {output_file}")
    
    # 词汇覆盖率
    vocab_chars = set(char_to_idx.keys()) - {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
    covered = len([c for c in text if c in char_to_idx])
    coverage = covered / len(text) * 100 if len(text) > 0 else 0
    print(f"词汇覆盖率: {coverage:.1f}%")
    
    return encoded


def clean_text(text: str) -> str:
    """清理文本"""
    import re
    
    # 移除多余空白
    text = re.sub(r'\s+', '', text)
    
    # 只保留汉字、英文字母、数字和常见标点
    # 允许的字符模式
    pattern = r'[\u4e00-\u9fff'  # 汉字
    pattern += r'a-zA-Z'           # 英文字母
    pattern += r'0-9'              # 数字
    pattern += r'，。、；：？！""''（）【】《》—…·'  # 中文标点
    pattern += r'.,!?()[]{}\-:;"\'@#$%^&*+=/<>~`|'  # 英文标点
    pattern += r' ]'
    pattern += r']'
    
    text = ''.join(re.findall(pattern, text))
    
    return text


def merge_training_files(
    vocab_file: str,
    *text_files: str,
    output_file: str = "train_data.npy"
):
    """合并多个文本文件为训练数据"""
    
    import numpy as np
    
    # 加载词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_to_idx = vocab_data['char_to_idx']
    unk_idx = char_to_idx.get('<UNK>', 1)
    
    all_tokens = []
    
    for text_file in text_files:
        if not os.path.exists(text_file):
            print(f"文件不存在: {text_file}")
            continue
            
        with open(text_file, 'r', encoding='utf-8') as f:
            text = clean_text(f.read())
        
        tokens = [char_to_idx.get(c, unk_idx) for c in text]
        all_tokens.extend(tokens)
        print(f"{text_file}: {len(tokens)} tokens")
    
    # 保存
    data = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, data)
    print(f"\n总token数: {len(all_tokens)}")
    print(f"已保存到: {output_file}")
    
    return data


def create_training_data_from_folder(
    folder_path: str,
    vocab_file: str,
    output_file: str = "train_data.npy"
):
    """从文件夹批量处理文本文件"""
    
    import glob
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    txt_files += glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
    
    print(f"找到 {len(txt_files)} 个文本文件")
    
    if not txt_files:
        print("没有找到文本文件！")
        return
    
    # 合并
    return merge_training_files(vocab_file, *txt_files, output_file=output_file)


# ============== 使用示例 ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练数据准备工具")
    parser.add_argument("text_file", help="文本文件路径")
    parser.add_argument("--vocab", default="merged_vocab.json", help="词汇表文件")
    parser.add_argument("--output", default=None, help="输出文件")
    parser.add_argument("--merge", nargs="+", help="合并多个文件")
    parser.add_argument("--folder", help="处理整个文件夹")
    
    args = parser.parse_args()
    
    if args.merge:
        # 合并多个文件
        merge_training_files(args.vocab, *args.merge)
    elif args.folder:
        # 处理文件夹
        create_training_data_from_folder(args.folder, args.vocab)
    else:
        # 单文件处理
        prepare_training_data(args.text_file, args.vocab, args.output)
