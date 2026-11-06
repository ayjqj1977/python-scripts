import torch
import torch.nn as nn
import json
import numpy as np

# ================= 请修改这两个文件名 =================
file_symbols = "符号向量字库.json"   # 你的符号向量文件
file_chinese = "汉字向量字库.json"  # 你的汉字向量文件
# =====================================================

# 1. 加载两个 JSON 文件
with open(file_symbols, "r", encoding="utf-8") as f:
    data_sym = json.load(f)

with open(file_chinese, "r", encoding="utf-8") as f:
    data_cn = json.load(f)

# 2. 提取字符和对应的 18 维向量
# 假设 JSON 结构是 {"characters": [{"char": "字符", "vector": [18个数值]}, ...]}
chars_sym = [item["char"] for item in data_sym["characters"]]
vecs_sym = [np.array(item["vector"], dtype=np.float32) for item in data_sym["characters"]]

chars_cn = [item["char"] for item in data_cn["characters"]]
vecs_cn = [np.array(item["vector"], dtype=np.float32) for item in data_cn["characters"]]

# 3. 合并字符（去重），优先保留汉字库的向量，符号库补充
all_chars = list(dict.fromkeys(chars_cn + chars_sym))  # 汉字在前，符号在后，去重
char_to_idx = {char: i for i, char in enumerate(all_chars)}

# 4. 构建合并向量矩阵（18维）
vec_matrix = []
for char in all_chars:
    if char in chars_cn:
        # 优先使用汉字库向量
        idx = chars_cn.index(char)
        vec_matrix.append(vecs_cn[idx])
    else:
        # 用符号库向量补充
        idx = chars_sym.index(char)
        vec_matrix.append(vecs_sym[idx])

vec_matrix = np.array(vec_matrix, dtype=np.float32)

# 5. 创建嵌入层
embedding_layer = nn.Embedding.from_pretrained(
    torch.from_numpy(vec_matrix),
    freeze=False  # 可继续训练
)

print("✅ 两个 18 维字库合并完成！")
print(f"📚 总字符数：{len(all_chars)}")
print(f"📐 向量维度：{vec_matrix.shape[1]} 维（确认是 18 维）")

# ================= 快速测试（可选） =================
# test_chars = ["你", "好", "'"]  # 可以测试任意字符
# test_ids = torch.tensor([char_to_idx[c] for c in test_chars])
# print("\n🧪 测试嵌入层输出：")
# print(embedding_layer(test_ids))