# -*- coding: utf-8 -*-
"""CNN模型生成测试"""
import torch
import sys
import os

WORK_DIR = r'D:\desktop\学习结果小文件'

# 加载诗词模型
print("📦 加载诗词模型...")
checkpoint = torch.load(os.path.join(WORK_DIR, 'checkpoints', 'poetry_cnn_model.pt'), weights_only=False)

sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from cnn_text_model import create_cnn_model

vocab_size = len(checkpoint['char_to_idx'])
# 诗词模型参数
model = create_cnn_model(vocab_size=vocab_size, embed_dim=18, num_filters=128, filter_sizes=[2,3,4,5])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']

print(f"   词表: {vocab_size}")
print("\n🧪 诗词生成测试:\n")

# 测试生成 - 用诗句开头
test_starts = ["春", "江", "花", "月", "夜", "床", "前", "明", "月", "光", "白", "日", "山", "水", "云"]

for start in test_starts[:8]:
    result = model.generate(start, char_to_idx, idx_to_char, max_len=20, temperature=0.8, top_k=30, repeat_penalty=2.0)
    print(f"   '{start}' → '{result}'")
