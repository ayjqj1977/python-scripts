#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
测试新训练的模型
"""
import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

print("📥 加载新训练的模型...\n")

from nanotext18_model import create_model
import numpy as np

# 加载词表
with open(os.path.join(WORK_DIR, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)

embed_matrix = np.load(os.path.join(WORK_DIR, 'embedding_matrix.npy'))

# 加载新训练的模型
checkpoint = torch.load(
    os.path.join(WORK_DIR, 'checkpoints', 'full_500.pt'), 
    map_location='cpu'
)
state_dict = checkpoint['model_state_dict']

model = create_model(vocab_size, 18, 3, 6, 64, torch.tensor(embed_matrix, dtype=torch.float32))
model.load_state_dict(state_dict)
model.eval()

print("✅ 模型加载完成!\n")

# 生成
def generate(prompt, max_len=50):
    input_ids = [char_to_idx.get(c, 1) for c in prompt]
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-63:] if len(generated) > 63 else generated
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == char_to_idx.get('<EOS>', 0): break
            generated.append(next_token)
    
    return ''.join([idx_to_char.get(i, '') for i in generated])

# 测试
print("="*50)
print("🖋️ 文本生成测试 (新模型 - 500次)")
print("="*50)

for p in ["汉字", "春风", "明月", "山水", "相思", "离别"]:
    result = generate(p, 40)
    print(f"提示: {p}")
    print(f"生成: {result}")
    print("-"*50)
