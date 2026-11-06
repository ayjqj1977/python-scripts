import tkinter as tk
from tkinter import messagebox

# 计算逻辑（和之前一样，只是把输入输出绑到窗口控件）
def calculate():
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        c = float(entry_c.get())
        delta = b**2 - 4*a*c
        if delta < 0:
            result.set("无实数解")
        else:
            x1 = (-b + delta**0.5)/(2*a)
            x2 = (-b - delta**0.5)/(2*a)
            result.set(f"x1={x1:.2f}  x2={x2:.2f}")
    except ValueError:
        messagebox.showerror("错误", "请输入正确数字！")  # 输错弹窗提示
    except ZeroDivisionError:
        messagebox.showerror("错误", "a不能为0！")  # a=0弹窗提示

# 窗口初始化
root = tk.Tk()
root.title("二次方程求解器")
root.geometry("300x200")  # 窗口大小

# 变量绑定（用于获取输入框内容、显示结果）
var_a = tk.StringVar()
var_b = tk.StringVar()
var_c = tk.StringVar()
result = tk.StringVar()

# 界面控件（标签+输入框+按钮，按顺序排列）
tk.Label(root, text="二次项系数a：").grid(row=0, column=0, padx=10, pady=10)
entry_a = tk.Entry(root, textvariable=var_a)
entry_a.grid(row=0, column=1)

tk.Label(root, text="一次项系数b：").grid(row=1, column=0, padx=10, pady=5)
entry_b = tk.Entry(root, textvariable=var_b)
entry_b.grid(row=1, column=1)

tk.Label(root, text="常数项系数c：").grid(row=2, column=0, padx=10, pady=5)
entry_c = tk.Entry(root, textvariable=var_c)
entry_c.grid(row=2, column=1)

# 计算按钮（点击触发calculate函数）
tk.Button(root, text="计算", command=calculate).grid(row=3, column=0, padx=10, pady=10)
# 退出按钮（点击关闭窗口）
tk.Button(root, text="退出", command=root.quit).grid(row=3, column=1)

# 结果显示
tk.Label(root, textvariable=result, fg="red").grid(row=4, column=0, columnspan=2)

# 窗口循环（保持窗口显示）
root.mainloop()