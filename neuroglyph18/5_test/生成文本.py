#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
简单易用的文本生成工具
输入提示词，自动生成后续内容
"""

import torch
import json
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'

print("="*60)
print("🖋️ NanoText-18 文本生成工具")
print("="*60)

# 加载模型
print("\n📥 加载模型中...")

from nanotext18_model import create_model
import numpy as np

# 加载词表
with open(os.path.join(WORK_DIR, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char_to_idx = vocab_data['char_to_idx']
idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
vocab_size = len(char_to_idx)

# 加载嵌入
embed_matrix = np.load(os.path.join(WORK_DIR, 'embedding_matrix.npy'))

# 加载训练好的模型
checkpoint = torch.load(
    os.path.join(WORK_DIR, 'checkpoints', 'nanotext18_iter2500.pt'), 
    map_location='cpu'
)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# 创建模型 (block_size=128 与训练时一致)
model = create_model(vocab_size, 18, 3, 6, 128, torch.tensor(embed_matrix, dtype=torch.float32))
model.load_state_dict(state_dict)
model.eval()

print("✅ 模型加载完成!\n")

# 生成函数
def generate_text(prompt, max_len=50, temperature=0.8):
    """根据提示词生成文本"""
    # 编码提示词
    input_ids = [char_to_idx.get(c, char_to_idx.get('<UNK>', 1)) for c in prompt]
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            # 取最后128个字符作为上下文
            context = generated[-127:] if len(generated) > 127 else generated
            
            # 前向传播
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            
            # 取最后一个位置的预测
            next_logits = logits[0, -1, :]
            
            # 应用温度（控制随机性）
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # 采样
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # 遇到结束符就停止
            if next_token == char_to_idx.get('<EOS>', 0):
                break
            
            generated.append(next_token)
    
    # 解码
    result = ''.join([idx_to_char.get(i, '') for i in generated])
    return result

# 主循环
print("\n" + "="*60)
print("使用说明:")
print("  - 输入提示词（如：汉字、中华、春风）")
print("  - 按回车生成后续内容")
print("  - 输入 'quit' 或 '退出' 结束程序")
print("="*60 + "\n")

while True:
    try:
        # 获取用户输入
        prompt = input("请输入提示词 > ").strip()
        
        # 检查退出
        if prompt in ['quit', '退出', 'q', 'exit']:
            print("\n👋 再见!")
            break
        
        # 检查空输入
        if not prompt:
            print("⚠️ 请输入提示词！\n")
            continue
        
        # 生成并显示结果
        print("\n生成中...\n")
        result = generate_text(prompt, max_len=50, temperature=0.8)
        
        print("-"*60)
        print(f"提示词: {prompt}")
        print(f"生成结果:")
        print(result)
        print("-"*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 再见!")
        break
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
