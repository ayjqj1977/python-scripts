# 1. 词向量（用列表模拟，维度2）
word_vectors = [
    [0.2, 0.5],  # 第1位：“我”的词向量
    [0.3, 0.6]   # 第2位：“吃”的词向量
]

# 2. 位置向量（同样用列表，维度和词向量一致）
pos_vectors = [
    [0.1, 0.05],  # 第1位的位置向量
    [0.2, 0.1]    # 第2位的位置向量
]

# 3. 词向量 + 位置向量（逐元素相加）
final_vectors = []
for word_vec, pos_vec in zip(word_vectors, pos_vectors):
    fused = [w + p for w, p in zip(word_vec, pos_vec)]
    final_vectors.append(fused)

print("融合位置信息后的向量：")
for vec in final_vectors:
    print(vec)
