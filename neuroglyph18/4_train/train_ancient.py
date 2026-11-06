#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 训练脚本 - 使用古诗词语料
专为16GB内存笔记本优化
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import time
from pathlib import Path


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, data_path: str, vocab_file: str, block_size: int = 64):
        self.block_size = block_size
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        # 加载训练数据
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                # 读取每行作为独立样本
                lines = f.readlines()
                # 合并为连续文本
                self.text = ''.join([line.strip() for line in lines if line.strip()])
        else:
            self.text = self._get_default_text()
        
        # 编码文本
        self.data = self._encode_text(self.text)
        print(f"数据集: {len(self.data):,} tokens, {len(self.text):,} chars")
    
    def _get_default_text(self):
        return "汉字是中华文化的瑰宝"
    
    def _encode_text(self, text: str):
        """编码文本"""
        indices = []
        unk_idx = self.char_to_idx.get('<UNK>', 1)
        for char in text:
            idx = self.char_to_idx.get(char, unk_idx)
            indices.append(idx)
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
    save_dir: str = './checkpoints',
    vocab_file: str = None
):
    """训练函数"""
    model.train()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练 (设备: {device})")
    print(f"最大迭代次数: {max_iters}")
    print(f"保存间隔: {save_interval}")
    
    iter_num = 0
    train_losses = []
    
    # 创建一个小的固定验证样本
    val_text = "汉字是中华文化的瑰宝"
    val_data = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    char_to_idx = vocab['char_to_idx']
    unk_idx = char_to_idx.get('<UNK>', 1)
    for c in val_text:
        val_data.append(char_to_idx.get(c, unk_idx))
    val_x = torch.tensor(val_data[:-1], dtype=torch.long).unsqueeze(0).to(device)
    val_y = torch.tensor(val_data[1:], dtype=torch.long).unsqueeze(0).to(device)
    
    start_time = time.time()
    
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
            
            train_losses.append(loss.item())
            
            # 评估和日志
            if iter_num % eval_interval == 0:
                avg_loss = sum(train_losses[-eval_interval:]) / eval_interval
                elapsed = time.time() - start_time
                print(f"迭代 {iter_num:5d} | 损失: {avg_loss:.4f} | 时间: {elapsed:.1f}s")
            
            # 保存检查点
            if iter_num % save_interval == 0 and iter_num > 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{iter_num}.pt')
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"  ✓ 保存检查点: {checkpoint_path}")
            
            iter_num += 1
    
    # 最终保存
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\n训练完成! 最终损失: {train_losses[-1]:.4f}")
    print(f"模型保存至: {final_path}")
    
    return train_losses


def main():
    # 配置
    WORK_DIR = r'D:\desktop\学习结果小文件'
    VOCAB_FILE = os.path.join(WORK_DIR, 'merged_vocab.json')
    DATA_FILE = os.path.join(WORK_DIR, 'corpus_ancient_combined.txt')
    EMBED_FILE = os.path.join(WORK_DIR, 'embedding_matrix.npy')
    SAVE_DIR = os.path.join(WORK_DIR, 'checkpoints')
    
    # 检查文件
    if not os.path.exists(VOCAB_FILE):
        print(f"错误: 词表文件不存在: {VOCAB_FILE}")
        return
    
    if not os.path.exists(DATA_FILE):
        print(f"错误: 训练数据文件不存在: {DATA_FILE}")
        # 尝试使用备选文件
        DATA_FILE = os.path.join(WORK_DIR, 'corpus_ancient.txt')
        if not os.path.exists(DATA_FILE):
            print(f"错误: 备选文件也不存在")
            return
    
    # 加载词汇表获取大小
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data['char_to_idx'])
    print(f"词表大小: {vocab_size}")
    
    # 加载嵌入矩阵
    embed_matrix = np.load(EMBED_FILE)
    print(f"嵌入矩阵: {embed_matrix.shape}")
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n加载数据集...")
    dataset = TextDataset(DATA_FILE, VOCAB_FILE, block_size=64)
    
    # 数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 创建模型
    print("\n创建模型...")
    from nanotext18_model import create_model
    embedding_weights = torch.tensor(embed_matrix, dtype=torch.float32)
    model = create_model(
        vocab_size=vocab_size,
        n_embd=18,
        n_head=3,
        n_layer=6,
        block_size=64,
        embedding_weights=embedding_weights
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 训练
    train(
        model, 
        train_loader, 
        optimizer, 
        device=device,
        max_iters=1000,
        eval_interval=50,
        save_interval=500,
        save_dir=SAVE_DIR,
        vocab_file=VOCAB_FILE
    )


if __name__ == '__main__':
    main()
