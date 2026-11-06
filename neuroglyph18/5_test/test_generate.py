#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
测试生成 - 使用刚训练好的模型
"""
import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

# 加载模型
print("📥 加载模型...", flush=True)
from nanotext18_model import create_model
import numpy as np

checkpoint = torch.load(os.path.join(WORK_DIR, 'checkpoints', 'quick_test.pt'), map_location='cpu')
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']

vocab_size = len(char_to_idx)
embed_matrix = np.load(os.path.join(WORK_DIR, 'embedding_matrix.npy'))

model = create_model(vocab_size, 18, 3, 6, 64, torch.tensor(embed_matrix, dtype=torch.float32))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 模型加载完成\n", flush=True)

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
            if next_token == char_to_idx.get('<EOS>', 0): break
            generated.append(next_token)
    
    return ''.join([idx_to_char.get(i, '') for i in generated])

# 测试
print("="*50, flush=True)
print("🖋️ 文本生成测试", flush=True)
print("="*50, flush=True)

prompts = ["汉字", "中华", "春风", "明月", "山水", "相思"]
for p in prompts:
    result = generate(p, 25)
    print(f"提示: {p} → {result}", flush=True)
    print("-"*40, flush=True)
