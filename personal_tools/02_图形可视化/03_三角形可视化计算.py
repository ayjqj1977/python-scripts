import tkinter as tk
from tkinter import messagebox, simpledialog
import math

def calculate_triangle_area():
    """
    通过对话框输入三角形三边长，自动计算三角形面积
    使用海伦公式计算面积
    """
    try:
        # 循环直到获得有效的边长输入
        while True:
            # 通过对话框获取三条边的长度
            a = simpledialog.askfloat("输入", "请输入第一条边长:", 
                                      minvalue=0.01, 
                                      initialvalue=1.0)
            
            # 如果用户取消输入，则退出程序
            if a is None:
                return
            
            b = simpledialog.askfloat("输入", "请输入第二条边长:", 
                                      minvalue=0.01, 
                                      initialvalue=1.0)
            
            if b is None:
                return
            
            c = simpledialog.askfloat("输入", "请输入第三条边长:", 
                                      minvalue=0.01, 
                                      initialvalue=1.0)
            
            if c is None:
                return
            
            # 检查是否能构成三角形（两边之和大于第三边）
            if a + b <= c or a + c <= b or b + c <= a:
                messagebox.showerror("错误", f"这三条边无法构成三角形！\n边长: {a}, {b}, {c}\n请重新输入。")
                continue  # 重新开始循环，再次请求输入
            
            # 计算半周长
            s = (a + b + c) / 2
            
            # 使用海伦公式计算面积
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
            # 显示结果，保留两位小数
            messagebox.showinfo("计算结果", f"三角形的面积为: {area:.2f}")
            break  # 计算完成，退出循环
            
    except Exception as e:
        # 处理可能的异常情况
        messagebox.showerror("错误", f"发生未知错误: {str(e)}")

# 只有在直接运行此脚本时才执行
if __name__ == "__main__":
    # 创建一个隐藏的根窗口，这是使用tkinter对话框所必需的
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    calculate_triangle_area()
    
    # 销毁根窗口
    root.destroy()
