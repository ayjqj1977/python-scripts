# -*- coding: utf-8 -*-
"""
整理文件夹 - 创建分类目录并整理文件
"""
import os
import shutil

WORK_DIR = r'D:\desktop\学习结果小文件'

# 创建分类目录
folders = {
    '1_18维向量表': ['汉字向量字库.json', '符号向量字库.json', 'merged_vocab.json', 'vocab_large.json'],
    '2_嵌入矩阵': ['embedding_matrix.npy', 'embedding_large.npy'],
    '3_模型文件': [],  # checkpoints子目录
    '4_训练脚本': [
        'nanotext18_model.py',
        'nanotext18_train.py', 
        'nanotext18_train_v2.py',
        'train_100.py',
        'train_3000.py',
        'train_500.py',
        'train_ancient.py',
        'train_from_json.py',
        'train_full.py',
        'train_large.py',
        'train_3000_bg.py',
        'quick_train.py',
    ],
    '5_测试脚本': [
        'test.py',
        'test_2500.py',
        'test_generate.py',
        'test_large.py',
        'test_new.py',
        'generate_poetry.py',
        '生成文本.py',
    ],
    '6_语料处理': [
        'download_corpus.py',
        'download_more.py',
        'download_extra.py',
        'expand_vocab.py',
        'generate_embed_large.py',
        'generate_all.py',
        'generate_embedding_matrix.py',
        'create_embedding_layer.py',
        'process_corpus.py',
        'process_corpus_v2.py',
        'check_corpus.py',
        'debug_corpus.py',
    ],
    '7_训练语料': ['corpus'],  # 子目录
}

print("=== 开始整理文件 ===\n")

# 创建目录
for folder in folders.keys():
    folder_path = os.path.join(WORK_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✓ 创建目录: {folder}")

# 移动文件
import glob

for folder, patterns in folders.items():
    folder_path = os.path.join(WORK_DIR, folder)
    
    # 如果是corpus目录，保留原位置，只记录
    if folder == '7_训练语料':
        print(f"\n📁 语料目录: {WORK_DIR}\\corpus")
        corpus_files = glob.glob(os.path.join(WORK_DIR, 'corpus', '*.json'))
        print(f"   包含 {len(corpus_files)} 个JSON文件")
        continue
    
    if folder == '3_模型文件':
        checkpoint_dir = os.path.join(WORK_DIR, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            print(f"\n📁 模型目录: {checkpoint_dir}")
            pt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
            print(f"   包含 {len(pt_files)} 个模型文件")
        continue
    
    # 移动文件
    for pattern in patterns:
        src = os.path.join(WORK_DIR, pattern)
        if os.path.exists(src):
            dst = os.path.join(WORK_DIR, folder, pattern)
            # 如果目标已存在，加编号
            if os.path.exists(dst):
                base, ext = os.path.splitext(pattern)
                dst = os.path.join(WORK_DIR, folder, f"{base}_1{ext}")
            shutil.move(src, dst)
            print(f"  ✓ 移动: {pattern} → {folder}")

print("\n=== 整理完成 ===")

# 列出最终结构
print("\n=== 最终目录结构 ===")
for folder in folders.keys():
    folder_path = os.path.join(WORK_DIR, folder)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        print(f"\n📁 {folder}")
        for f in files[:5]:
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... 共 {len(files)} 个文件")
