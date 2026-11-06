#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
测试Transformer模型 - 原始版本
"""
import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

print("📥 加载Transformer模型...")
from nanotext18_model import create_model
import numpy as np

# 加载原始词汇表
with open(os.path.join(WORK_DIR, '1_18维向量表', 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)

# 加载原始嵌入
embed_matrix = np.load(os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_matrix.npy'))

# 加载Transformer模型 (最早的版本)
checkpoint = torch.load(
    os.path.join(WORK_DIR, 'checkpoints', 'nanotext18_iter2500.pt'), 
    map_location='cpu'
)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# 创建Transformer模型
model = create_model(vocab_size, 18, 3, 6, 128, torch.tensor(embed_matrix, dtype=torch.float32))
model.load_state_dict(state_dict)
model.eval()

print("✅ Transformer模型加载完成!\n")

# 生成函数
def generate(prompt, max_len=30):
    input_ids = [char_to_idx.get(c, 1) for c in prompt]
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-63:] if len(generated) > 63 else generated
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == 0: break
            generated.append(next_token)
    
    return ''.join([idx_to_char.get(i, '') for i in generated])

# 测试
print("="*50)
print("🖋️ Transformer模型生成测试")
print("="*50)

for p in ["春", "江", "花", "月", "夜", "床", "明"]:
    result = generate(p, 25)
    print(f"'{p}' → '{result}'")
