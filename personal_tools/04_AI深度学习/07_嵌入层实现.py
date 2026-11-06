import torch
import torch.nn as nn
import json
import numpy as np

# 🔧 只改这里：填你真实的两个JSON文件名（和你合并成功的一致）
file_sym = "符号向量字库.json"   # 你的符号库
file_cn = "汉字向量字库.json"    # 你的汉字库

# 1. 加载已合并的核心数据（直接复用你成功的逻辑）
with open(file_sym, "r", encoding="utf-8") as f:
    data_sym = json.load(f)
with open(file_cn, "r", encoding="utf-8") as f:
    data_cn = json.load(f)

# 2. 提取字符+18维向量（和你之前一致的结构）
chars_sym = [item["char"] for item in data_sym["characters"]]
vecs_sym = np.array([item["vector"] for item in data_sym["characters"]], dtype=np.float32)
chars_cn = [item["char"] for item in data_cn["characters"]]
vecs_cn = np.array([item["vector"] for item in data_cn["characters"]], dtype=np.float32)

# 3. 合并去重（汉字优先，你已经成功的步骤）
all_chars = list(dict.fromkeys(chars_cn + chars_sym))
char_to_idx = {char: i for i, char in enumerate(all_chars)}

# 4. 构建最终18维向量矩阵
vec_matrix = []
for char in all_chars:
    if char in chars_cn:
        vec_matrix.append(vecs_cn[chars_cn.index(char)])
    else:
        vec_matrix.append(vecs_sym[chars_sym.index(char)])
vec_matrix = np.array(vec_matrix, dtype=np.float32)

# 5. 🔥 直接生成可用的嵌入层（核心！你要的就是这个）
embedding_layer = nn.Embedding.from_pretrained(
    torch.from_numpy(vec_matrix),  # 用合并好的18维向量初始化
    freeze=False                   # 训练时可更新，适配你的模型
)

# ✅ 成功提示（确认无误）
print("✅ 最终嵌入层生成成功！")
print(f"📚 总字符数：{len(all_chars)}")
print(f"📐 向量维度：{vec_matrix.shape[1]} 维（确认18维）")
print("🎯 嵌入层可直接用于模型训练/推理！")