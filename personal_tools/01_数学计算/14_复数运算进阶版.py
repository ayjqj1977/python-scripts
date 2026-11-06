import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

class ComplexCalculator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口
        
        # 获取复数输入
        self.a = self.get_float_input("请输入z1实部a：", "z1实部输入")
        self.b = self.get_float_input("请输入z1虚部b：", "z1虚部输入")
        self.c = self.get_float_input("请输入z2实部c：", "z2实部输入")
        self.d = self.get_float_input("请输入z2虚部d：", "z2虚部输入")
        
        # 执行计算
        self.calculate()
        
        # 显示结果
        self.show_results()
        
        # 绘制图形
        self.plot_complex_numbers()
    
    def get_float_input(self, prompt, title):
        """获取浮点数输入"""
        while True:
            try:
                value = simpledialog.askfloat(title, prompt)
                if value is None:  # 用户点击了取消
                    exit()
                return value
            except Exception:
                prompt = "输入无效，请重新输入\n\n" + prompt.split('\n')[-1]  # 保留原始提示信息
    
    def calculate(self):
        """执行复数运算"""
        # 加法
        self.add_a, self.add_b = self.a + self.c, self.b + self.d
        # 减法
        self.sub_a, self.sub_b = self.a - self.c, self.b - self.d
        # 乘法
        self.mul_a, self.mul_b = self.a*self.c - self.b*self.d, self.a*self.d + self.b*self.c
        # 除法
        denominator = self.c**2 + self.d**2
        if denominator != 0:
            self.div_a, self.div_b = (self.a*self.c + self.b*self.d)/denominator, (self.b*self.c - self.a*self.d)/denominator
        else:
            self.div_a, self.div_b = None, None

    def show_results(self):
        """显示计算结果"""
        print("\n=== 运算结果 ===")
        print(f"z1 = {self.a} + {self.b}i")
        print(f"z2 = {self.c} + {self.d}i")
        print(f"z1 + z2 = {self.add_a} + {self.add_b}i")
        print(f"z1 - z2 = {self.sub_a} + {self.sub_b}i")
        print(f"z1 × z2 = {self.mul_a} + {self.mul_b}i")
        print(f"z1 ÷ z2 = {'无法除法（z2为0）' if self.c**2 + self.d**2 == 0 else f'{self.div_a:.2f} + {self.div_b:.2f}i'}")

    def plot_complex_numbers(self):
        """绘制复数向量图"""
        plt.figure(figsize=(12, 8))
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # 实轴
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)  # 虚轴

        # 绘制向量箭头
        plt.arrow(0, 0, self.a, self.b, color='red', head_width=0.2, head_length=0.3, linewidth=2)  # z1向量
        plt.arrow(0, 0, self.c, self.d, color='blue', head_width=0.2, head_length=0.3, linewidth=2)  # z2向量
        plt.arrow(0, 0, self.add_a, self.add_b, color='green', head_width=0.2, head_length=0.3, linewidth=2)  # 加法向量
        plt.arrow(0, 0, self.sub_a, self.sub_b, color='orange', head_width=0.2, head_length=0.3, linewidth=2)  # 减法向量
        plt.arrow(0, 0, self.mul_a, self.mul_b, color='purple', head_width=0.2, head_length=0.3, linewidth=2)  # 乘法向量
        if self.c**2 + self.d**2 != 0:
            plt.arrow(0, 0, self.div_a, self.div_b, color='brown', head_width=0.2, head_length=0.3, linewidth=2)  # 除法向量

        # 绘制点并标注
        plt.scatter(self.a, self.b, color='red', marker='o', s=100)
        plt.text(self.a+0.1, self.b+0.1, f'z1\n({self.a:.2f},{self.b:.2f})', fontsize=9)
        plt.scatter(self.c, self.d, color='blue', marker='o', s=100)
        plt.text(self.c+0.1, self.d+0.1, f'z2\n({self.c:.2f},{self.d:.2f})', fontsize=9)
        plt.scatter(self.add_a, self.add_b, color='green', marker='s', s=100)
        plt.text(self.add_a+0.1, self.add_b+0.1, f'z1+z2\n({self.add_a:.2f},{self.add_b:.2f})', fontsize=9)
        plt.scatter(self.sub_a, self.sub_b, color='orange', marker='s', s=100)
        plt.text(self.sub_a+0.1, self.sub_b+0.1, f'z1-z2\n({self.sub_a:.2f},{self.sub_b:.2f})', fontsize=9)
        plt.scatter(self.mul_a, self.mul_b, color='purple', marker='^', s=100)
        plt.text(self.mul_a+0.1, self.mul_b+0.1, f'z1×z2\n({self.mul_a:.2f},{self.mul_b:.2f})', fontsize=9)
        if self.c**2 + self.d**2 != 0:
            plt.scatter(self.div_a, self.div_b, color='brown', marker='^', s=100)
            plt.text(self.div_a+0.1, self.div_b+0.1, f'z1÷z2\n({self.div_a:.2f},{self.div_b:.2f})', fontsize=9)

        # 图像美化
        plt.xlabel('实轴（Real Part）')
        plt.ylabel('虚轴（Imaginary Part）')
        plt.title('复数加减乘除运算（向量可视化）')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')  # 等比例坐标轴，向量长度不畸变

        plt.show()
    
    def run(self):
        """运行程序"""
        self.root.mainloop()


if __name__ == "__main__":
    print("=== 复数计算器（向量可视化版）===")
    calculator = ComplexCalculator()