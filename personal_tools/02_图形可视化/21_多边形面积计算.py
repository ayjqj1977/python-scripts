import tkinter as tk
from tkinter import messagebox
import math

# 全局变量
side_lengths = {}
result = None
entry_list = []

# 获取边数并生成输入框
def get_sides_num():
    global entry_list
    try:
        num = int(entry_num.get())
        if num not in [3,4,5,6]:
            messagebox.showerror("错误", "仅支持3/4/5/6边形！")
            return
        
        for widget in frame_sides.winfo_children():
            widget.destroy()
        side_lengths.clear()
        entry_list = []
        
        for i in range(num):
            var = tk.StringVar()
            side_lengths[f"side{i+1}"] = var
            tk.Label(frame_sides, text=f"第{i+1}条边长：").grid(row=i, column=0, padx=5, pady=3)
            entry = tk.Entry(frame_sides, textvariable=var, width=15)
            entry.grid(row=i, column=1)
            entry_list.append(entry)
            
            if i < num - 1:
                entry.bind('<Return>', lambda e, idx=i+1: entry_list[idx].focus())
            else:
                entry.bind('<Return>', lambda e: calculate_area())
        
        btn_calc.config(state=tk.NORMAL)
        entry_list[0].focus()
    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字（3/4/5/6）！")

# 计算面积
def calculate_area(event=None):
    try:
        num = int(entry_num.get())
        side = float(list(side_lengths.values())[0].get())
        if side <= 0:
            messagebox.showerror("错误", "边长必须大于0！")
            return
        
        area = (num * side ** 2) / (4 * math.tan(math.pi / num))
        result.set(f"{num}边形面积：{area:.2f}（平方单位）")
    except ValueError:
        messagebox.showerror("错误", "请输入有效的边长数字！")
    except Exception as e:
        messagebox.showerror("错误", f"计算出错：{str(e)}")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    root.title("正多边形面积计算器（回车跳转）")
    # 调大窗口高度，适配6边形显示（宽度300→320，高度350→420）
    root.geometry("320x420")
    result = tk.StringVar()

    # 边数输入区域
    frame_num = tk.Frame(root)
    frame_num.pack(pady=8)
    tk.Label(frame_num, text="输入边数（3/4/5/6）：").grid(row=0, column=0, padx=5)
    entry_num = tk.Entry(frame_num, width=10)
    entry_num.grid(row=0, column=1)
    entry_num.bind('<Return>', lambda e: get_sides_num())
    btn_confirm = tk.Button(frame_num, text="确认边数", command=get_sides_num, bg="#2196F3", fg="white")
    btn_confirm.grid(row=0, column=2, padx=5)

    # 边长输入区域（用pack+fill让区域自适应）
    frame_sides = tk.Frame(root)
    frame_sides.pack(pady=5, fill=tk.X, padx=10)

    # 按钮区域（用frame包裹，避免排版混乱）
    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=8)
    btn_calc = tk.Button(frame_btn, text="计算面积", command=calculate_area, bg="#4CAF50", fg="white", state=tk.DISABLED)
    btn_calc.grid(row=0, column=0, padx=10)
    btn_quit = tk.Button(frame_btn, text="退出", command=root.quit, bg="#f44336", fg="white")
    btn_quit.grid(row=0, column=1, padx=10)

    # 结果显示区域（单独frame，保证始终可见）
    frame_result = tk.Frame(root)
    frame_result.pack(pady=10, fill=tk.X, padx=10)
    tk.Label(frame_result, textvariable=result, fg="red", font=("Arial", 10, "bold")).pack()

    root.mainloop()