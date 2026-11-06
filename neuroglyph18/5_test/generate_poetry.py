#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型生成古诗词
"""
import torch
import json
import numpy as np
import os
import sys

WORK_DIR = r'D:\desktop\学习结果小文件'
sys.path.insert(0, os.path.join(WORK_DIR, '4_训练脚本'))

def load_model():
    """加载模型"""
    from nanotext18_model import create_model
    
    # 加载词表
    vocab_file = os.path.join(WORK_DIR, 'merged_vocab.json')
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    vocab_size = len(char_to_idx)
    
    # 加载嵌入
    embed_file = os.path.join(WORK_DIR, 'embedding_matrix.npy')
    embedding_weights = torch.tensor(np.load(embed_file), dtype=torch.float32)
    
    # 创建模型
    model = create_model(
        vocab_size=vocab_size,
        n_embd=18,
        n_head=3,
        n_layer=6,
        block_size=128,
        embedding_weights=embedding_weights
    )
    
    # 加载检查点
    checkpoint_path = os.path.join(WORK_DIR, 'checkpoints', 'nanotext18_iter2500.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"模型已加载: {checkpoint_path}")
    
    return model, char_to_idx, idx_to_char

def generate(model, idx_to_char, char_to_idx, prompt, max_len=50, temperature=0.8):
    """生成文本"""
    model.eval()
    
    # 编码提示词
    input_ids = []
    for c in prompt:
        idx = char_to_idx.get(c, char_to_idx.get('<UNK>', 1))
        input_ids.append(idx)
    
    # 转换为tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # 生成
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            if len(generated) > 63:
                context = generated[-63:]
            else:
                context = generated
            
            logits, _ = model(torch.tensor([context], dtype=torch.long))
            next_logits = logits[0, -1, :]
            
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == char_to_idx.get('<EOS>', 0):
                break
            
            generated.append(next_token)
    
    result = ''.join([idx_to_char.get(i, '') for i in generated])
    return result

def main():
    print("加载模型...")
    model, char_to_idx, idx_to_char = load_model()
    
    print("\n=== 古诗词生成测试 ===\n")
    
    prompts = ["汉字", "中华", "春风", "明月", "山水", "相思", "送别"]
    
    for prompt in prompts:
        print(f"提示词: {prompt}")
        result = generate(model, idx_to_char, char_to_idx, prompt, max_len=30)
        print(f"生成: {result}")
        print("-" * 30)

if __name__ == '__main__':
    main()
