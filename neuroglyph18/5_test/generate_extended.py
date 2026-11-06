# -*- coding: utf-8 -*-
"""
文本生成脚本 - 使用扩展词汇表模型
"""
import torch
import json
import numpy as np
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
MODEL_PATH = os.path.join(WORK_DIR, 'checkpoints', 'extended_vocab_1500.pt')
VOCAB_FILE = os.path.join(WORK_DIR, 'vocab_extended.json')

print("="*60)
print("📝 文本生成 - 扩展词汇表模型")
print("="*60)

# 加载词汇表
print("\n📚 加载词汇表...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
vocab_size = len(char_to_idx)
print(f"   词表: {vocab_size}")

# 加载模型
print("\n🔄 加载模型...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# 创建模型
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))
from nanotext18_model import create_model

embed_matrix = np.load(os.path.join(WORK_DIR, '2_嵌入矩阵', 'embedding_extended.npy'))
embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)

model = create_model(vocab_size, 18, 3, 6, 64, embedding_weights)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("   模型加载完成!")

# 生成函数
@torch.no_grad()
def generate(text, max_len=100, temperature=0.8):
    model.eval()
    
    # 编码
    context = [char_to_idx.get(c, 0) for c in text]
    context = context[-63:] if len(context) > 63 else context
    
    generated = context.copy()
    
    for _ in range(max_len):
        x = torch.tensor([generated[-64:]], dtype=torch.long)
        
        # 确保x长度够64
        if x.shape[1] < 64:
            x = torch.cat([torch.zeros(1, 64 - x.shape[1], dtype=torch.long), x], dim=1)
        
        logits, _ = model(x, torch.zeros_like(x))
        logits = logits[0, -1] / temperature
        
        probs = torch.softmax(logits, dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        
        generated.append(next_idx)
        
        if idx_to_char.get(next_idx, '') == '' or next_idx == 0:
            break
    
    # 解码
    result = ''.join([idx_to_char.get(i, '') for i in generated])
    return result

# 测试生成
print("\n🎯 测试生成...")
test_prompts = [
    "太阳",
    "月亮",
    "中国",
    "电脑",
    "学习",
]

for prompt in test_prompts:
    print(f"\n输入: {prompt}")
    result = generate(prompt, max_len=50)
    print(f"生成: {result}")

print("\n" + "="*60)
print("✅ 测试完成!")
