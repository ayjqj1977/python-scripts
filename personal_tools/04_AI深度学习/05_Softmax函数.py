# 1. 定义基础矩阵
I = [[1, 3], [2, 4]]  # 输入矩阵：2个词×2维向量
Wq = [[1, 0], [0, 1]] # 转换矩阵（Q）
Wk = [[1, 0], [0, 1]] # 转换矩阵（K）
Wv = [[1, 0], [0, 1]] # 转换矩阵（V）

# 2. 工具函数：矩阵乘法、转置、Softmax
def matrix_mult(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matrix_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def softmax(matrix):
    # 对矩阵的每一行单独做Softmax
    result = []
    for row in matrix:
        exp_row = [pow(2.71828, x) for x in row]  # e^x（近似值）
        exp_sum = sum(exp_row)
        softmax_row = [x / exp_sum for x in exp_row]
        result.append(softmax_row)
    return result

# 3. 生成Q/K/V
Q = matrix_mult(Wq, I)
K = matrix_mult(Wk, I)
V = matrix_mult(Wv, I)

# 4. 计算关联度矩阵 → 权重矩阵（Softmax）
K_T = matrix_transpose(K)
attention_scores = matrix_mult(Q, K_T)
attention_weights = softmax(attention_scores)  # 新增：Softmax标准化

# 5. 计算最终输出（权重矩阵 × V）
output = matrix_mult(attention_weights, V)

# 打印结果（标注Softmax的作用）
print("输入矩阵I:", I)
print("关联度矩阵:", attention_scores)
print("Softmax后的权重矩阵（每行和为1）:", attention_weights)
print("带Softmax的最终输出:", output)
