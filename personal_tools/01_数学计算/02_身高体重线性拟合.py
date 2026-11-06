# 数据：身高（厘米）、对应的体重（千克）
heights = [150, 160, 170, 180]
weights = [40, 45, 55, 65]

# 简单线性模型：weight = w * height + b（用最小二乘法算w和b）
n = len(heights)
sum_h = sum(heights)
sum_w = sum(weights)
sum_hw = sum(h * w for h, w in zip(heights, weights))
sum_h2 = sum(h**2 for h in heights)

# 计算参数w和b
w = (n * sum_hw - sum_h * sum_w) / (n * sum_h2 - sum_h**2)
b = (sum_w - w * sum_h) / n

# 预测175cm的体重
predict_weight = w * 175 + b
print("不缩放：")
print(f"参数w={w:.2f}，b={b:.2f}")
print(f"175cm对应的体重预测值：{predict_weight:.1f}千克\n")
