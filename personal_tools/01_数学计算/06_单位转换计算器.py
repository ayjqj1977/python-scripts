# 数据：身高（原单位厘米）→ 转成米
heights_cm = [150, 160, 170, 180]
heights_m = [h / 100 for h in heights_cm]  # 厘米→米，范围1.5-1.8
weights = [40, 45, 55, 65]
n = len(heights_m)

# 计算模型参数
sum_h = sum(heights_m)
sum_w = sum(weights)
sum_hw = sum(h * w for h, w in zip(heights_m, weights))
sum_h2 = sum(h**2 for h in heights_m)
w = (n * sum_hw - sum_h * sum_w) / (n * sum_h2 - sum_h**2)
b = (sum_w - w * sum_h) / n

# 预测175cm（即1.75米）的体重
predict_weight = w * 1.75 + b
print("单位转换（厘米→米）后：")
print(f"参数w={w:.2f}，b={b:.2f}")
print(f"175cm对应的体重预测值：{predict_weight:.1f}千克")
