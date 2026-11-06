# 1. 定义数据（确保heights变量提前声明）
heights = [150, 160, 170, 180]
weights = [40, 45, 55, 65]
n = len(heights)

# 2. 不做特征缩放的情况
sum_h = sum(heights)
sum_w = sum(weights)
sum_hw = sum(h * w for h, w in zip(heights, weights))
sum_h2 = sum(h**2 for h in heights)
w = (n * sum_hw - sum_h * sum_w) / (n * sum_h2 - sum_h**2)
b = (sum_w - w * sum_h) / n
predict_weight = w * 175 + b
print("不缩放：")
print(f"参数w={w:.2f}，b={b:.2f}")
print(f"175cm对应的体重预测值：{predict_weight:.1f}千克\n")

# 3. 做特征缩放的情况
h_min = min(heights)
h_max = max(heights)
scaled_heights = [(h - h_min) / (h_max - h_min) for h in heights]
sum_sh = sum(scaled_heights)
sum_shw = sum(sh * w for sh, w in zip(scaled_heights, weights))
sum_sh2 = sum(sh**2 for sh in scaled_heights)
w_scaled = (n * sum_shw - sum_sh * sum_w) / (n * sum_sh2 - sum_sh**2)
b_scaled = (sum_w - w_scaled * sum_sh) / n
scaled_175 = (175 - h_min) / (h_max - h_min)
predict_weight_scaled = w_scaled * scaled_175 + b_scaled
print("缩放后：")
print(f"参数w={w_scaled:.2f}，b={b_scaled:.2f}")
print(f"175cm对应的体重预测值：{predict_weight_scaled:.1f}千克")
