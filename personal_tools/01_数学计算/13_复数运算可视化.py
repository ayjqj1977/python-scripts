import matplotlib.pyplot as plt
import numpy as np

# 复数计算器（带复平面可视化）
print("=== 复数计算器（含可视化）===")
print("复数格式：z1 = a + b*i ， z2 = c + d*i")

# 1. 接收用户输入
a = float(input("请输入z1的实部a："))
b = float(input("请输入z1的虚部b："))
c = float(input("请输入z2的实部c："))
d = float(input("z2的虚部d："))

# 2. 计算加减乘除结果
# 加法
add_a = a + c
add_b = b + d
# 减法
sub_a = a - c
sub_b = b - d
# 乘法
mul_a = a * c - b * d
mul_b = a * d + b * c
# 除法
div_a, div_b = 0, 0
denominator = c**2 + d**2
if denominator == 0:
    div_result = "z2为0，无法除法"
else:
    div_a = (a * c + b * d) / denominator
    div_b = (b * c - a * d) / denominator
    div_result = f"{div_a:.2f} + {div_b:.2f}i"

# 3. 复平面可视化
plt.figure(figsize=(12, 8))
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # 实轴
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)  # 虚轴

# 定义点的样式和标注
points = {
    "z1": (a, b, "red", "o"),
    "z2": (c, d, "blue", "o"),
    "z1+z2": (add_a, add_b, "green", "s"),
    "z1-z2": (sub_a, sub_b, "orange", "s"),
    "z1×z2": (mul_a, mul_b, "purple", "^"),
}
# 除法单独处理（避免分母为0）
if denominator != 0:
    points["z1÷z2"] = (div_a, div_b, "brown", "^")

# 绘制点并标注
for name, (x, y, color, marker) in points.items():
    plt.scatter(x, y, color=color, marker=marker, s=100, label=name)
    plt.text(x + 0.1, y + 0.1, f"{name}\n({x:.2f},{y:.2f})", fontsize=9)

# 图像设置
plt.xlabel("实轴 (Real)")
plt.ylabel("虚轴 (Imaginary)")
plt.title("复数运算复平面可视化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")  # 等比例坐标轴，避免变形

# 4. 打印运算结果
print("\n=== 运算结果 ===")
print(f"z1 = {a} + {b}i")
print(f"z2 = {c} + {d}i")
print(f"z1 + z2 = {add_a} + {add_b}i")
print(f"z1 - z2 = {sub_a} + {sub_b}i")
print(f"z1 × z2 = {mul_a} + {mul_b}i")
print(f"z1 ÷ z2 = {div_result if denominator == 0 else f'{div_a:.2f} + {div_b:.2f}i'}")

# 显示图像
plt.show()