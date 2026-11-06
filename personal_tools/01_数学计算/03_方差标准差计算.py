def calculate_variance_and_std(data):
    """输入数据列表，返回方差和标准差"""
    n = len(data)
    mean = sum(data) / n  # 均值
    variance = sum((x - mean)**2 for x in data) / n  # 方差
    std = variance**0.5  # 标准差
    return round(variance, 2), round(std, 2)

# 示例：计算身高的方差和标准差
heights = [150, 160, 170, 180]
variance, std = calculate_variance_and_std(heights)
print(f"方差：{variance}，标准差：{std}")  # 输出：方差：125.0，标准差：11.18
