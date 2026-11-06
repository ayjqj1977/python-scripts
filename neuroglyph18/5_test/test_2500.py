#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
测试生成 - 使用2500次迭代的模型
"""
import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

print("📥 加载模型 (2500次迭代)...\n")

from nanotext18_model import create_model
import numpy as np

# 加载词表
with open(os.path.join(WORK_DIR, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}

# 加载检查点
checkpoint = torch.load(
    os.path.join(WORK_DIR, 'checkpoints', 'nanotext18_iter2500.pt'), 
    map_location='cpu'
)

# 检查点格式不同，需要用model_state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

vocab_size = len(char_to_idx)
embed_matrix = np.load(os.path.join(WORK_DIR, 'embedding_matrix.npy'))

model = create_model(vocab_size, 18, 3, 6, 128, torch.tensor(embed_matrix, dtype=torch.float32))
model.load_state_dict(state_dict)
model.eval()

print(f"✅ 模型加载完成!\n")

# 生成函数
def generate(prompt, max_len=40):
    input_ids = [char_to_idx.get(c, 1) for c in prompt]
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-127:] if len(generated) > 127 else generated
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == char_to_idx.get('<EOS>', 0): break
            generated.append(next_token)
    
    return ''.join([idx_to_char.get(i, '') for i in generated])

# 测试各种提示词
print("="*60)
print("🖋️ 文本生成测试 (2500次迭代)")
print("="*60)

test_prompts = [
    "汉字", "中华", "春风", "明月", "山水", 
    "相思", "离别", "归乡", "春花", "秋月",
    "床前", "白日", "红豆", "青天", "故乡"
]

for p in test_prompts:
    result = generate(p, 35)
    print(f"提示: {p}")
    print(f"生成: {result}")
    print("-"*60)
