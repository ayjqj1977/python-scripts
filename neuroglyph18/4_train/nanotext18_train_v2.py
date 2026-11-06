#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 训练脚本 v2
支持自定义训练数据
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, data_file: str, block_size: int = 128):
        self.block_size = block_size
        
        # 加载数据
        if data_file.endswith('.npy'):
            self.data = np.load(data_file)
        elif data_file.endswith('.txt'):
            # 如果是文本文件，先编码
            self.data = self._encode_text(data_file)
        else:
            raise ValueError("不支持的文件格式")
        
        print(f"数据集: {len(self.data)} tokens")
    
    def _encode_text(self, text_file: str) -> np.ndarray:
        """编码文本文件"""
        VOCAB_FILE = r"D:\desktop\学习结果小文件\merged_vocab.json"
        
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        char_to_idx = vocab_data['char_to_idx']
        unk_idx = char_to_idx.get('<UNK>', 1)
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 简单清理
        import re
        text = re.sub(r'\s+', '', text)
        
        indices = [char_to_idx.get(c, unk_idx) for c in text]
        return np.array(indices, dtype=np.int64)
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    max_iters: int = 3000,
    eval_interval: int = 100,
    save_interval: int = 500,
    save_dir: str = './checkpoints'
):
    """训练函数"""
    model.train()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练 (设备: {device})")
    print(f"最大迭代次数: {max_iters}")
    
    iter_num = 0
    losses = []
    
    while iter_num < max_iters:
        for batch in train_loader:
            if iter_num >= max_iters:
                break
            
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # 评估
            if iter_num % eval_interval == 0:
                avg_loss = np.mean(losses[-eval_interval:])
                print(f"Iter {iter_num}: Loss = {avg_loss:.4f}")
            
            # 保存
            if iter_num % save_interval == 0 and iter_num > 0:
                save_path = os.path.join(save_dir, f'nanotext18_iter{iter_num}.pt')
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
            
            iter_num += 1
    
    print("\n训练完成!")
    return losses


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40
):
    """生成文本"""
    model.eval()
    
    VOCAB_FILE = r"D:\desktop\学习结果小文件\merged_vocab.json"
    
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    
    # 编码
    idx = torch.tensor([[char_to_idx.get(c, 1) for c in prompt]], dtype=torch.long)
    
    # 生成
    generated = model.generate(idx, max_new_tokens, temperature, top_k)
    
    # 解码
    result = ''.join([idx_to_char.get(i, '<UNK>') for i in generated[0].tolist()])
    
    return result


def main():
    """主函数"""
    
    # ===== 在这里修改配置 =====
    TRAIN_DATA_FILE = r"D:\desktop\学习结果小文件\train_text.txt"  # 你的训练数据
    # =================================
    
    VOCAB_FILE = r"D:\desktop\学习结果小文件\merged_vocab.json"
    EMBED_FILE = r"D:\desktop\学习结果小文件\embedding_matrix.npy"
    SAVE_DIR = r"D:\desktop\学习结果小文件\checkpoints"
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 超参数
    BATCH_SIZE = 64
    BLOCK_SIZE = 128
    LEARNING_RATE = 3e-4
    MAX_ITERS = 5000  # 可以加大
    EVAL_INTERVAL = 100
    SAVE_INTERVAL = 500
    
    # 数据集
    print(f"\n加载训练数据: {TRAIN_DATA_FILE}")
    dataset = TextDataset(TRAIN_DATA_FILE, BLOCK_SIZE)
    
    if len(dataset) == 0:
        print("错误: 训练数据为空!")
        return
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # 加载嵌入
    embedding_weights = torch.tensor(np.load(EMBED_FILE), dtype=torch.float32)
    
    # 模型
    from nanotext18_model import create_model
    model = create_model(
        vocab_size=6170,
        n_embd=18,
        n_head=3,
        n_layer=6,
        block_size=BLOCK_SIZE,
        embedding_weights=embedding_weights
    ).to(device)
    
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1
    )
    
    # 训练
    train(
        model, train_loader, optimizer,
        device=device,
        max_iters=MAX_ITERS,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        save_dir=SAVE_DIR
    )
    
    # 测试生成
    print("\n" + "="*50)
    print("生成测试:")
    print("="*50)
    
    prompts = ["汉字是", "中文的", "学习中", "文化是"]
    for p in prompts:
        result = generate(model, p, max_new_tokens=30)
        print(f"\n提示: {p}")
        print(f"生成: {result}")


if __name__ == "__main__":
    main()
