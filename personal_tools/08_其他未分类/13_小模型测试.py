import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# ---------------------- 1. 投喂资料：加载并处理自定义文本 ----------------------
def load_data(file_path):
    # 读取自己的文本资料
    with open(file_path, 'r', encoding='gbk',errors='ignore') as f:
        text = f.read()
    # 构建字符字典（核心：让模型认识资料里的字）
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    # 把文本转成数字序列（模型能识别的格式）
    text_idx = [char2idx[c] for c in text]
    return text_idx, char2idx, idx2char, vocab_size, text

# 替换成你的文本文件路径（比如自己写的小说、笔记）
text_idx, char2idx, idx2char, vocab_size, raw_text = load_data("mydata.txt")
seq_length = 20  # 输入序列长度
step = 3         # 步长
# 构建训练数据（X是输入字符，y是下一个字符）
X, y = [], []
for i in range(0, len(text_idx) - seq_length, step):
    X.append(text_idx[i:i+seq_length])
    y.append(text_idx[i+seq_length])
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# ---------------------- 2. 从零写小模型：简易LSTM ----------------------
class TinyTextGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out, hidden

# 初始化模型（参数量极小，轻薄本秒跑）
model = TinyTextGenerator(vocab_size, hidden_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------- 3. 训练模型：吃资料长大 ----------------------
epochs = 100  # 训练轮数，越多越像你的资料风格
batch_size = 32
model.train()
for epoch in range(epochs):
    # 分批训练（避免内存不足）
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        hidden = None  # 初始化隐藏状态
        optimizer.zero_grad()
        output, hidden = model(batch_X, hidden)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    # 每10轮打印进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------------- 4. 生成文本：基于你的资料风格创作 ----------------------
def generate_text(model, start_text, char2idx, idx2char, length=300):
    model.eval()
    # 用你指定的开头文字启动生成
    start_idx = [char2idx[c] for c in start_text if c in char2idx]
    if len(start_idx) < seq_length:
        start_idx = [char2idx[list(char2idx.keys())[0]]] * (seq_length - len(start_idx)) + start_idx
    else:
        start_idx = start_idx[-seq_length:]
    x = torch.tensor([start_idx], dtype=torch.long)
    hidden = None
    generated = start_text
    # 逐字符生成
    for _ in range(length):
        output, hidden = model(x, hidden)
        # 选概率最大的字符（也可以加随机度，让生成更灵活）
        pred_idx = torch.argmax(output, dim=1).item()
        generated_char = idx2char[pred_idx]
        generated += generated_char
        # 更新输入序列
        x = torch.cat([x[:, 1:], torch.tensor([[pred_idx]])], dim=1)
    return generated

# 生成文本（开头可以自己定，模型会模仿你的资料风格）
generated_text = generate_text(model, start_text="对酒当歌", char2idx=char2idx, idx2char=idx2char)
print("\n生成的文本：")
print(generated_text)