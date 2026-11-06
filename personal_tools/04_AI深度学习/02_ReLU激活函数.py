# ReLU 函数核心实现（一行代码）
def relu(x):
    return max(0, x)  # 输入x≥0返回x，否则返回0

# 测试示例（覆盖正数、负数、0三种情况）
test_values = [3, -2, 0, 5.8, -4.3]
for x in test_values:
    print(f"ReLU({x}) = {relu(x)}")
