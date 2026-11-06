import torch
from nanotext18_model import create_model
import numpy as np
import json

# 加载嵌入权重
embeddings = np.load('embedding_matrix.npy')
embedding_weights = torch.tensor(embeddings, dtype=torch.float32)

# 创建模型
model = create_model(
    vocab_size=6170,
    n_embd=18,
    n_head=3,
    n_layer=12
),
    block_size=512,
    embedding_weights=embedding_weights
)

# 查看参数量
print(f"模型参数量: {model.get_num_params():,}")

# 编码并生成
# (详细见训练脚本)