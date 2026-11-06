import tkinter as tk
from tkinter import messagebox
import math

# 全局变量
point_entries = []
result = None
canvas = None
current_points = 3

# 获取顶点数并生成坐标输入框
def get_point_num():
    global point_entries, current_points
    try:
        num = int(entry_num.get())
        if num < 3:
            messagebox.showerror("错误", "多边形至少需要3个顶点！")
            return
        current_points = num
        
        for widget in frame_points.winfo_children():
            widget.destroy()
        point_entries.clear()
        
        # 生成顶点坐标输入框
        for i in range(num):
            frame_i = tk.Frame(frame_points)
            frame_i.grid(row=i, column=0, padx=5, pady=3, sticky=tk.W)
            # 显示顶点字母（A、B、C、D...）
            point_char = chr(ord('A') + i)
            tk.Label(frame_i, text=f"顶点{i+1}（{point_char}）：").grid(row=0, column=0)
            
            var_x = tk.StringVar()
            entry_x = tk.Entry(frame_i, textvariable=var_x, width=8)
            entry_x.grid(row=0, column=1, padx=2)
            tk.Label(frame_i, text="x:").grid(row=0, column=2)
            
            var_y = tk.StringVar()
            entry_y = tk.Entry(frame_i, textvariable=var_y, width=8)
            entry_y.grid(row=0, column=3, padx=2)
            tk.Label(frame_i, text="y:").grid(row=0, column=4)
            
            point_entries.append((var_x, var_y))
            var_x.trace_add('write', lambda *args: draw_polygon())
            var_y.trace_add('write', lambda *args: draw_polygon())
        
        btn_calc.config(state=tk.NORMAL)
        if point_entries:
            point_entries[0][0].get()
            entry_x.focus()
        draw_polygon()
    except ValueError:
        messagebox.showerror("错误", "请输入有效的顶点数（≥3）！")

# 绘制多边形+顶点字母标注
def draw_polygon():
    global canvas
    canvas.delete("all")
    points = []
    try:
        for var_x, var_y in point_entries:
            x = float(var_x.get()) if var_x.get() else 0
            y = float(var_y.get()) if var_y.get() else 0
            scale = 10
            offset_x = 100
            offset_y = 100
            draw_x = x * scale + offset_x
            draw_y = -y * scale + offset_y
            points.append((draw_x, draw_y))
        
        if len(points) >= 3:
            canvas.create_polygon(points, outline="#2196F3", fill="#E3F2FD", width=2)
            # 绘制顶点和字母标注
            for idx, (x, y) in enumerate(points):
                # 顶点红点
                canvas.create_oval(x-3, y-3, x+3, y+3, fill="#f44336")
                # 字母标注（偏移一点避免覆盖红点）
                char = chr(ord('A') + idx)
                canvas.create_text(x+8, y-8, text=char, font=("Arial", 10, "bold"), fill="#000000")
    except ValueError:
        pass

# 鞋带公式计算面积
def calculate_area(event=None):
    try:
        coords = []
        for var_x, var_y in point_entries:
            x = float(var_x.get())
            y = float(var_y.get())
            coords.append((x, y))
        
        if len(coords) < 3:
            messagebox.showerror("错误", "至少需要3个顶点！")
            return
        
        n = len(coords)
        area = 0.0
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i+1)%n]
            area += (x1 * y2) - (x2 * y1)
        area = abs(area) / 2.0
        result.set(f"多边形面积：{area:.2f}（平方单位）")
        draw_polygon()
    except ValueError:
        messagebox.showerror("错误", "请输入有效的坐标数字！")
    except Exception as e:
        messagebox.showerror("错误", f"计算出错：{str(e)}")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    root.title("不规则多边形面积计算器（带顶点标注）")
    root.geometry("500x550")
    result = tk.StringVar()

    # 顶点数输入区域
    frame_num = tk.Frame(root)
    frame_num.pack(pady=10)
    tk.Label(frame_num, text="输入顶点数（≥3）：").grid(row=0, column=0, padx=5)
    entry_num = tk.Entry(frame_num, width=10)
    entry_num.grid(row=0, column=1)
    entry_num.bind('<Return>', lambda e: get_point_num())
    btn_confirm = tk.Button(frame_num, text="确认顶点数", command=get_point_num, bg="#2196F3", fg="white")
    btn_confirm.grid(row=0, column=2, padx=5)

    # 坐标输入区域
    frame_points = tk.Frame(root)
    frame_points.pack(pady=5, fill=tk.X, padx=10)

    # 绘图画布区域
    frame_canvas = tk.Frame(root)
    frame_canvas.pack(pady=10)
    canvas = tk.Canvas(frame_canvas, width=200, height=200, bg="white", bd=1, relief=tk.SUNKEN)
    canvas.pack()

    # 按钮区域
    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=8)
    btn_calc = tk.Button(frame_btn, text="计算面积", command=calculate_area, bg="#4CAF50", fg="white", state=tk.DISABLED)
    btn_calc.grid(row=0, column=0, padx=10)
    btn_quit = tk.Button(frame_btn, text="退出", command=root.quit, bg="#f44336", fg="white")
    btn_quit.grid(row=0, column=1, padx=10)

    # 结果显示区域
    frame_result = tk.Frame(root)
    frame_result.pack(pady=10)
    tk.Label(frame_result, textvariable=result, fg="red", font=("Arial", 10, "bold")).pack()

    root.mainloop()