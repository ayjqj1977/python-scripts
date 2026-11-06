# 1. 定义基础矩阵（用列表模拟）
I = [[1, 3], [2, 4]]
Wq = [[1, 0], [0, 1]]
Wk = [[1, 0], [0, 1]]
Wv = [[1, 0], [0, 1]]

# 2. 矩阵乘法函数（原生Python实现）
def matrix_mult(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

# 3. 矩阵转置函数
def matrix_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

# 4. 生成Q/K/V
Q = matrix_mult(Wq, I)
K = matrix_mult(Wk, I)
V = matrix_mult(Wv, I)

# 5. 计算关联度矩阵
K_T = matrix_transpose(K)
attention_scores = matrix_mult(Q, K_T)

# 6. 计算输出
output = matrix_mult(attention_scores, V)

# 打印结果
print("输入矩阵I:", I)
print("Q矩阵:", Q)
print("关联度矩阵:", attention_scores)
print("最终输出:", output)
