import tkinter as tk
from tkinter import messagebox
import math  # 用于计算平方根的实部和虚部

# 计算逻辑（新增复数解处理）
def calculate():
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        c = float(entry_c.get())
        if a == 0:
            messagebox.showerror("错误", "a不能为0，这不是二次方程！")
            return
        
        delta = b**2 - 4*a*c
        # 区分实数解和复数解
        if delta >= 0:
            x1 = (-b + math.sqrt(delta)) / (2*a)
            x2 = (-b - math.sqrt(delta)) / (2*a)
            result.set(f"实数解：x1={x1:.2f}  x2={x2:.2f}")
        else:
            real_part = -b / (2*a)  # 复数的实部
            imaginary_part = math.sqrt(-delta) / (2*a)  # 复数的虚部
            # 格式化复数解，用i表示虚数单位
            x1 = f"{real_part:.2f}+{imaginary_part:.2f}i"
            x2 = f"{real_part:.2f}-{imaginary_part:.2f}i"
            result.set(f"复数解：x1={x1}  x2={x2}")
        
        messagebox.showinfo("计算完成", "方程解已算出，查看结果吧～")
    
    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字！")

# 窗口初始化
root = tk.Tk()
root.title("二次方程求解器（支持复数）")
root.geometry("350x220")

# 变量绑定
var_a = tk.StringVar()
var_b = tk.StringVar()
var_c = tk.StringVar()
result = tk.StringVar()

# 界面控件
tk.Label(root, text="二次项系数a：").grid(row=0, column=0, padx=10, pady=10)
entry_a = tk.Entry(root, textvariable=var_a, width=25)
entry_a.grid(row=0, column=1)

tk.Label(root, text="一次项系数b：").grid(row=1, column=0, padx=10, pady=5)
entry_b = tk.Entry(root, textvariable=var_b, width=25)
entry_b.grid(row=1, column=1)

tk.Label(root, text="常数项系数c：").grid(row=2, column=0, padx=10, pady=5)
entry_c = tk.Entry(root, textvariable=var_c, width=25)
entry_c.grid(row=2, column=1)

# 按钮（带颜色）
tk.Button(root, text="计算", command=calculate, bg="#4CAF50", fg="white").grid(row=3, column=0, padx=10, pady=10)
tk.Button(root, text="退出", command=root.quit, bg="#f44336", fg="white").grid(row=3, column=1)

# 结果显示
tk.Label(root, textvariable=result, fg="red", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2)

# 窗口循环
root.mainloop()