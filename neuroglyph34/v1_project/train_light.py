#!/usr/bin/env python3
"""
轻量版 NeuroGlyph-34 训练（仅用 numpy，无 PyTorch）
"""
import os, json, numpy as np, time, random

# 数据
data_dir = r"D:\Desktop\neuroglyph34_project"
emb = np.load(os.path.join(data_dir, "embedding_matrix.npy")).astype(np.float32)
with open(os.path.join(data_dir, "char_to_idx.json"), encoding='utf-8') as f:
    vocab = json.load(f)

# 语料
with open(os.path.join(data_dir, "corpus_training_combined.txt"), encoding='utf-8') as f:
    text = ''.join([l.strip() for l in f if l.strip()])

print(f"词表: {len(vocab)}")
print(f"嵌入矩阵: {emb.shape}")
print(f"语料长度: {len(text)} 字符")

# 简单前馈网络（演示用）
class SimpleNN:
    def __init__(self, d_in=128, d_hidden=64, d_out=7987):
        self.W1 = np.random.randn(d_in, d_hidden) * 0.01
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_hidden, d_out) * 0.01
        self.b2 = np.zeros(d_out)
    
    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def step(self, x, y, lr=0.001):
        # 前向
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        # softmax 交叉熵
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        loss = -np.log(probs[range(len(y)), y] + 1e-8).mean()
        # 梯度
        grad = probs
        grad[range(len(y)), y] -= 1
        grad /= len(y)
        dW2 = h.T @ grad
        db2 = grad.sum(axis=0)
        dh = grad @ self.W2.T
        dh[h <= 0] = 0
        dW1 = x.T @ dh
        db1 = dh.sum(axis=0)
        # 更新
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

# 准备数据
seq_len = 32
batch_size = 16
def get_batch():
    idx = random.randint(0, len(text) - seq_len - 1)
    seq = text[idx:idx+seq_len+1]
    x = [vocab.get(ch, 1) for ch in seq[:-1]]
    y = [vocab.get(ch, 1) for ch in seq[1:]]
    return np.array([emb[i] for i in x]), np.array(y)

# 训练
model = SimpleNN(d_in=34, d_hidden=64, d_out=len(vocab))
print("开始轻量训练...")
for step in range(1000):
    x_batch, y_batch = get_batch()
    loss = model.step(x_batch, y_batch, lr=0.01)
    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss:.4f}")

print("完成。保存权重...")
np.savez(os.path.join(data_dir, "simple_weights.npz"),
         W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
print(f"保存至 {data_dir}\\simple_weights.npz")