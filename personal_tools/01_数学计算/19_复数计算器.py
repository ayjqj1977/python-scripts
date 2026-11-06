# 复数计算器：输入z1(a+bi)和z2(c+di)，输出加减乘除结果
print("=== 复数计算器 ===")
print("复数格式：z1 = a + b*i ， z2 = c + d*i")

# 1. 接收用户输入（转成浮点数，兼容小数输入）
a = float(input("请输入z1的实部a："))
b = float(input("请输入z1的虚部b："))
c = float(input("请输入z2的实部c："))
d = float(input("请输入z2的虚部d："))

# 2. 定义两个复数的字符串格式（方便后续打印，优化显示效果）
z1 = f"{a} + {b}i" if b >= 0 else f"{a} - {abs(b)}i"
z2 = f"{c} + {d}i" if d >= 0 else f"{c} - {abs(d)}i"

# 3. 复数加减乘除运算（按公式计算，无模块依赖）
# 加法：(a+c) + (b+d)i
add_a = a + c
add_b = b + d
add_result = f"{add_a} + {add_b}i" if add_b >= 0 else f"{add_a} - {abs(add_b)}i"

# 减法：(a-c) + (b-d)i
sub_a = a - c
sub_b = b - d
sub_result = f"{sub_a} + {sub_b}i" if sub_b >= 0 else f"{sub_a} - {abs(sub_b)}i"

# 乘法：(a*c - b*d) + (a*d + b*c)i
mul_a = a * c - b * d
mul_b = a * d + b * c
mul_result = f"{mul_a} + {mul_b}i" if mul_b >= 0 else f"{mul_a} - {abs(mul_b)}i"

# 除法：先判断分母是否为0，再按公式计算
div_result = ""
denominator = c ** 2 + d ** 2  # 除法分母（实数）
if denominator == 0:
    div_result = "错误：z2为0，复数不能除以0！"
else:
    div_a = (a * c + b * d) / denominator  # 商的实部
    div_b = (b * c - a * d) / denominator  # 商的虚部
    div_result = f"{div_a:.2f} + {div_b:.2f}i" if div_b >= 0 else f"{div_a:.2f} - {abs(div_b):.2f}i"

# 4. 打印所有结果（格式清晰，方便查看）
print("\n=== 运算结果 ===")
print(f"z1 = {z1}")
print(f"z2 = {z2}")
print(f"z1 + z2 = {add_result}")
print(f"z1 - z2 = {sub_result}")
print(f"z1 × z2 = {mul_result}")
print(f"z1 ÷ z2 = {div_result}")
