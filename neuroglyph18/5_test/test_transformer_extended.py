#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
测试新的Transformer模型 - 扩展词汇表+诗词
"""
import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

print("📥 加载新的Transformer模型...")

from nanotext18_model import create_model
import numpy as np

# 加载词汇表
with open(os.path.join(WORK_DIR, 'vocab_extended.json'), 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)

# 加载嵌入
embed_matrix = np.load(os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy'))

# 加载新模型
checkpoint = torch.load(
    os.path.join(WORK_DIR, 'checkpoints', 'transformer_poetry_extended.pt'), 
    map_location='cpu'
)

model = create_model(
    vocab_size=vocab_size,
    n_embd=18,
    n_head=3,
    n_layer=6,
    block_size=64,
    embedding_weights=torch.tensor(embed_matrix, dtype=torch.float32)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 模型加载完成! 词表: {vocab_size}\n")

# 生成函数
def generate(prompt, max_len=30):
    input_ids = [char_to_idx.get(c, 0) for c in prompt]
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
print("🖋️ 新Transformer模型生成测试")
print("="*50)

for p in ["春", "江", "花", "月", "夜", "床", "明", "山水", "相思"]:
    result = generate(p, 25)
    print(f"'{p}' → '{result}'")
