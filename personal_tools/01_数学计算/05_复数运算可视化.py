import matplotlib.pyplot as plt

# 定义复数
a = -2 - 7j
b = 4 -8j
c = a + b

# 提取坐标
a_x, a_y = a.real, a.imag
b_x, b_y = b.real, b.imag
c_x, c_y = c.real, c.imag

plt.figure(figsize=(50, 50))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

# 1. 画向量a（原点→a）
plt.quiver(0, 0, a_x, a_y, angles='xy', scale_units='xy', scale=1, 
           color='blue', label=f'向量a: {a}')
# 2. 画向量b（原点→b）
plt.quiver(0, 0, b_x, b_y, angles='xy', scale_units='xy', scale=1, 
           color='green', label=f'向量b: {b}')
# 3. 画平行四边形的边（a→c，即a+ b）
plt.quiver(a_x, a_y, b_x, b_y, angles='xy', scale_units='xy', scale=1, 
           color='green', linestyle='--')
# 4. 画平行四边形的边（b→c，即b+ a）
plt.quiver(b_x, b_y, a_x, a_y, angles='xy', scale_units='xy', scale=1, 
           color='blue', linestyle='--')
# 5. 画和向量c（原点→c）
plt.quiver(0, 0, c_x, c_y, angles='xy', scale_units='xy', scale=1, 
           color='red', label=f'和向量a+b: {c}')

# 调整坐标轴范围（确保所有元素显示）
plt.xlim(min(a_x, b_x, c_x)-3, max(a_x, b_x, c_x)+3)
plt.ylim(min(a_y, b_y, c_y)-3, max(a_y, b_y, c_y)+3)
plt.xlabel('实部', fontsize=12)
plt.ylabel('虚部', fontsize=12)
plt.title('复平面复数加法（完整平行四边形）', fontsize=14)
plt.legend(fontsize=10)
plt.show()
