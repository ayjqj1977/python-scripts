import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import date, datetime, timedelta

def date_converter():
    # 基准日期：你的出生日期
    BIRTH_DATE = date(1977, 8, 24)
    messagebox.showinfo("信息", f"🎂 基准日期：{BIRTH_DATE}（你的出生日期）")

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    while True:
        # 功能菜单
        choice = simpledialog.askstring(
            "请选择功能", 
            "请选择功能：\n\n"
            "1. 输入天数 → 生成【出生日期±天数】对应的日期（正数往后算，负数往前算）\n"
            "2. 输入任意日期 → 计算距离出生日期的天数差\n\n"
            "输入功能编号（1/2）："
        )

        if choice is None:  # 用户取消操作
            break

        # 功能1：天数转日期（支持正负天数）
        if choice.strip() == "1":
            try:
                days = simpledialog.askinteger(
                    "输入天数", 
                    "请输入天数（正数往后、负数往前）：\n\n"
                    "（点击确定继续）"
                )
                
                if days is None:  # 用户取消操作
                    continue
                    
                target_date = BIRTH_DATE + timedelta(days=days)
                messagebox.showinfo(
                    "计算结果", 
                    f"✅ {BIRTH_DATE} {'+' if days>=0 else ''}{days} 天 = {target_date}"
                )
            except Exception as e:
                messagebox.showerror("错误", f"输入错误！请输入整数天数（比如1000、-500）\n\n错误详情：{str(e)}")

        # 功能2：日期转天数
        elif choice.strip() == "2":
            try:
                date_str = simpledialog.askstring(
                    "输入日期", 
                    "请输入日期（格式：年-月-日，比如2025-12-09）：\n\n"
                    "（点击确定继续）"
                )
                
                if date_str is None:  # 用户取消操作
                    continue
                    
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                days_diff = (target_date - BIRTH_DATE).days
                messagebox.showinfo(
                    "计算结果", 
                    f"✅ {target_date} 距离出生日期共 {days_diff} 天"
                )
            except ValueError:
                messagebox.showerror(
                    "错误", 
                    "❌ 日期格式错误！请按「年-月-日」输入（比如2025-12-09，单月单日不用补0）"
                )
            except Exception as e:
                messagebox.showerror("错误", f"输入错误！\n\n错误详情：{str(e)}")

        # 无效输入处理
        else:
            messagebox.showerror("错误", "❌ 功能编号错误！请只输入1或2")
            continue

        # 询问是否继续
        result = messagebox.askyesno("继续使用", "是否继续使用？\n\n点击“是”继续，点击“否”退出")
        if not result:
            messagebox.showinfo("\n", "👋 工具已退出，下次见～")
            break

    root.destroy()

if __name__ == "__main__":
    date_converter()