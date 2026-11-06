import tkinter as tk
from tkinter import messagebox
import math

def calculate_triangle_area():
    """
    通过GUI界面输入三角形三边长，自动计算三角形面积
    使用海伦公式计算面积
    """
    def calculate():
        try:
            # 获取输入框中的值
            a_str = entry_a.get().strip()
            b_str = entry_b.get().strip()
            c_str = entry_c.get().strip()
            
            # 检查输入是否为空
            if not a_str or not b_str or not c_str:
                messagebox.showerror("错误", "请输入所有边长值！")
                return
            
            a = float(a_str)
            b = float(b_str)
            c = float(c_str)
            
            # 检查边长是否为正数
            if a <= 0 or b <= 0 or c <= 0:
                messagebox.showerror("错误", "边长必须为正数！")
                return
            
            # 检查是否能构成三角形
            if a + b <= c or a + c <= b or b + c <= a:
                messagebox.showerror("错误", f"这三条边无法构成三角形！\n边长: {a}, {b}, {c}")
                return
            
            # 计算半周长
            s = (a + b + c) / 2
            
            # 使用海伦公式计算面积
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
            # 显示结果
            result_label.config(text=f"三角形的面积为: {area:.2f}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字！")
    
    def on_entry_return_a(event):
        """当在第一条边输入框按下回车时，自动跳转到第二条边输入框"""
        entry_b.focus_set()
        
    def on_entry_return_b(event):
        """当在第二条边输入框按下回车时，自动跳转到第三条边输入框"""
        entry_c.focus_set()
        
    def on_entry_return_c(event):
        """当在第三条边输入框按下回车时，自动执行计算"""
        calculate()
    
    # 创建主窗口
    root = tk.Tk()
    root.title("三角形面积计算器")
    root.geometry("350x300")
    
    # 创建标题标签
    label_title = tk.Label(root, text="三角形面积计算器", font=("Arial", 14, "bold"))
    label_title.pack(pady=10)
    
    # 创建说明标签
    label_instruction = tk.Label(root, text="请输入三角形的三条边长：", font=("Arial", 10))
    label_instruction.pack(pady=5)
    
    # 第一条边
    frame_a = tk.Frame(root)
    frame_a.pack(pady=5)
    label_a = tk.Label(frame_a, text="第一条边长:", width=12, anchor='w')
    label_a.pack(side=tk.LEFT)
    entry_a = tk.Entry(frame_a, width=15)
    entry_a.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_a.bind('<Return>', on_entry_return_a)
    
    # 第二条边
    frame_b = tk.Frame(root)
    frame_b.pack(pady=5)
    label_b = tk.Label(frame_b, text="第二条边长:", width=12, anchor='w')
    label_b.pack(side=tk.LEFT)
    entry_b = tk.Entry(frame_b, width=15)
    entry_b.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_b.bind('<Return>', on_entry_return_b)
    
    # 第三条边
    frame_c = tk.Frame(root)
    frame_c.pack(pady=5)
    label_c = tk.Label(frame_c, text="第三条边长:", width=12, anchor='w')
    label_c.pack(side=tk.LEFT)
    entry_c = tk.Entry(frame_c, width=15)
    entry_c.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_c.bind('<Return>', on_entry_return_c)
    
    # 创建计算按钮
    calc_button = tk.Button(root, text="计算面积", command=calculate, 
                           bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), 
                           width=15, height=1)
    calc_button.pack(pady=15)
    
    # 创建结果显示标签
    result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="blue", 
                           wraplength=300)
    result_label.pack(pady=10)
    
    # 设置初始焦点到第一个输入框
    entry_a.focus_set()
    
    # 运行主循环
    root.mainloop()

# 只有在直接运行此脚本时才执行
if __name__ == "__main__":
    calculate_triangle_area()
import tkinter as tk
from tkinter import messagebox
import math

def calculate_triangle_area():
    """
    通过GUI界面输入三角形三边长，自动计算三角形面积
    使用海伦公式计算面积
    """
    def calculate():
        try:
            # 获取输入框中的值
            a_str = entry_a.get().strip()
            b_str = entry_b.get().strip()
            c_str = entry_c.get().strip()
            
            # 检查输入是否为空
            if not a_str or not b_str or not c_str:
                messagebox.showerror("错误", "请输入所有边长值！")
                return
            
            a = float(a_str)
            b = float(b_str)
            c = float(c_str)
            
            # 检查边长是否为正数
            if a <= 0 or b <= 0 or c <= 0:
                messagebox.showerror("错误", "边长必须为正数！")
                return
            
            # 检查是否能构成三角形
            if a + b <= c or a + c <= b or b + c <= a:
                messagebox.showerror("错误", f"这三条边无法构成三角形！\n边长: {a}, {b}, {c}")
                return
            
            # 计算半周长
            s = (a + b + c) / 2
            
            # 使用海伦公式计算面积，确保被开方数非负（处理浮点精度问题）
            discriminant = s * (s - a) * (s - b) * (s - c)
            if discriminant < 0:
                discriminant = 0  # 防止由于浮点精度导致的负数开方
            area = math.sqrt(discriminant)
            
            # 显示结果
            try:
                result_label.config(text=f"三角形的面积为: {area:.2f}")
            except tk.TclError:
                # 处理GUI组件可能已被销毁的情况
                pass
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字！")
        except Exception as e:
            messagebox.showerror("错误", f"发生未知错误: {str(e)}")
    
    def on_entry_return_a(event):
        """当在第一条边输入框按下回车时，自动跳转到第二条边输入框"""
        try:
            entry_b.focus_set()
        except tk.TclError:
            pass
        
    def on_entry_return_b(event):
        """当在第二条边输入框按下回车时，自动跳转到第三条边输入框"""
        try:
            entry_c.focus_set()
        except tk.TclError:
            pass
        
    def on_entry_return_c(event):
        """当在第三条边输入框按下回车时，自动执行计算"""
        calculate()
    
    # 创建主窗口
    root = tk.Tk()
    root.title("三角形面积计算器")
    root.geometry("350x300")
    
    # 创建标题标签
    label_title = tk.Label(root, text="三角形面积计算器", font=("Arial", 14, "bold"))
    label_title.pack(pady=10)
    
    # 创建说明标签
    label_instruction = tk.Label(root, text="请输入三角形的三条边长：", font=("Arial", 10))
    label_instruction.pack(pady=5)
    
    # 第一条边
    frame_a = tk.Frame(root)
    frame_a.pack(pady=5)
    label_a = tk.Label(frame_a, text="第一条边长:", width=12, anchor='w')
    label_a.pack(side=tk.LEFT)
    entry_a = tk.Entry(frame_a, width=15)
    entry_a.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_a.bind('<Return>', on_entry_return_a)
    
    # 第二条边
    frame_b = tk.Frame(root)
    frame_b.pack(pady=5)
    label_b = tk.Label(frame_b, text="第二条边长:", width=12, anchor='w')
    label_b.pack(side=tk.LEFT)
    entry_b = tk.Entry(frame_b, width=15)
    entry_b.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_b.bind('<Return>', on_entry_return_b)
    
    # 第三条边
    frame_c = tk.Frame(root)
    frame_c.pack(pady=5)
    label_c = tk.Label(frame_c, text="第三条边长:", width=12, anchor='w')
    label_c.pack(side=tk.LEFT)
    entry_c = tk.Entry(frame_c, width=15)
    entry_c.pack(side=tk.LEFT)
    # 绑定回车事件
    entry_c.bind('<Return>', on_entry_return_c)
    
    # 创建计算按钮
    calc_button = tk.Button(root, text="计算面积", command=calculate, 
                           bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), 
                           width=15, height=1)
    calc_button.pack(pady=15)
    
    # 创建结果显示标签
    result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="blue", 
                           wraplength=300)
    result_label.pack(pady=10)
    
    # 设置初始焦点到第一个输入框
    entry_a.focus_set()
    
    # 运行主循环
    root.mainloop()