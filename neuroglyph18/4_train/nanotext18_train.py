#!/usr/bin env python3
# -*- coding: utf-8 -*-
"""
NanoText-18 训练脚本
专为16GB内存笔记本优化
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from pathlib import Path


class TextDataset(Dataset):
    """简单文本数据集"""
    def __init__(self, data_path: str, vocab_file: str, block_size: int = 128):
        self.block_size = block_size
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        # 加载或创建训练数据
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
        else:
            # 使用默认训练数据
            self.text = self._get_default_text()
        
        # 编码文本
        self.data = self._encode_text(self.text)
        print(f"数据集: {len(self.data)} tokens")
    
    def _get_default_text(self):
        """获取默认训练文本"""
        return """汉字是中华文化的瑰宝，承载着数千年的文明传承。
每一个汉字都有其独特的形声义三位一体的结构。
中文是世界上最古老、最优美的语言之一。
学习汉字需要掌握其笔画、偏旁和部首。
汉字的演变经历了甲骨文、金文、篆书、隶书、楷书等阶段。
中文的语法灵活多变，表达丰富多样。
汉字的书写讲究横平竖直，布局合理。
中文诗词讲究意境和对仗工整。
汉字的输入法有拼音、五笔等多种方式。
中文的声调使语言更加富有音乐性。"""
    
    def _encode_text(self, text: str):
        """编码文本"""
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 1))
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
    max_iters: int = 1000,
    eval_interval: int = 100,
    save_interval: int = 500,
    save_dir: str = './checkpoints'
):
    """训练函数"""
    model.train()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练 (设备: {device})")
    print(f"最大迭代次数: {max_iters}")
    print(f"保存间隔: {save_interval}")
    
    iter_num = 0
    train_losses = []
    
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
            
            # 记录
            train_losses.append(loss.item())
            
            # 评估
            if iter_num % eval_interval == 0:
                avg_loss = np.mean(train_losses[-eval_interval:])
                print(f"Iter {iter_num}: Loss = {avg_loss:.4f}")
            
            # 保存
            if iter_num % save_interval == 0 and iter_num > 0:
                save_path = os.path.join(save_dir, f'nanotext18_iter{iter_num}.pt')
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_path)
                print(f"已保存: {save_path}")
            
            iter_num += 1
    
    print("\n训练完成!")
    return train_losses


@torch.no_grad()
def generate_text(
    model: nn.Module,
    vocab_file: str,
    device: str = 'cpu',
    prompt: str = "汉字",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40
):
    """生成文本"""
    model.eval()
    
    # 加载词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    
    # 编码提示词
    idx = torch.tensor([[char_to_idx.get(c, 1) for c in prompt]], dtype=torch.long).to(device)
    
    # 生成
    generated = model.generate(idx, max_new_tokens, temperature, top_k)
    
    # 解码
    result = ''.join([idx_to_char.get(i, '<UNK>') for i in generated[0].tolist()])
    
    return result


def main():
    """主函数"""
    # 配置
    VOCAB_FILE = r"D:\desktop\学习结果小文件\merged_vocab.json"
    EMBED_FILE = r"D:\desktop\学习结果小文件\embedding_matrix.npy"
    TRAIN_DATA = r"D:\desktop\学习结果小文件\train_text.txt"
    SAVE_DIR = r"D:\desktop\学习结果小文件\checkpoints"
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 超参数
    BATCH_SIZE = 64
    BLOCK_SIZE = 128
    LEARNING_RATE = 3e-4
    MAX_ITERS = 3000
    EVAL_INTERVAL = 100
    SAVE_INTERVAL = 500
    
    # 创建数据集
    print("\n加载数据集...")
    dataset = TextDataset(TRAIN_DATA, VOCAB_FILE, BLOCK_SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # CPU训练设为0
    )
    
    # 加载预训练嵌入
    print("加载预训练嵌入...")
    embedding_weights = torch.tensor(np.load(EMBED_FILE), dtype=torch.float32)
    print(f"嵌入形状: {embedding_weights.shape}")
    
    # 创建模型
    print("\n创建模型...")
    from nanotext18_model import create_model
    model = create_model(
        vocab_size=6170,
        n_embd=18,
        n_head=3,
        n_layer=6,
        block_size=BLOCK_SIZE,
        embedding_weights=embedding_weights
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # 训练
    train(
        model,
        train_loader,
        optimizer,
        device=device,
        max_iters=MAX_ITERS,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        save_dir=SAVE_DIR
    )
    
    # 生成测试
    print("\n" + "=" * 50)
    print("生成测试:")
    print("=" * 50)
    
    for i in range(3):
        result = generate_text(
            model, VOCAB_FILE, device,
            prompt="汉字",
            max_new_tokens=30,
            temperature=0.8,
            top_k=40
        )
        print(f"\n生成{i+1}: {result}")


if __name__ == "__main__":
    main()
