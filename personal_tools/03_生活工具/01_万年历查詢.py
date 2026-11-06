import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import calendar

class CalendarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("班次万年历")
        self.root.geometry("800x600")
        
        # 定义排班循环：早班、晚班、中班，各3天
        self.shift_cycle = ['早班', '晚班', '中班']
        self.cycle_days = 3  # 每个班次的天数
        self.total_cycle = len(self.shift_cycle) * self.cycle_days  # 总循环周期：9天
        
        # 基准日期：2025-12-01，是第二个早班
        self.base_date = datetime(2025, 12, 1)
        self.base_cycle_pos = 1  # 基准日在总循环中的位置（0开始：早班1=0，早班2=1，早班3=2，晚班1=3...）
        
        self.setup_ui()
        self.current_date = datetime.now()
        self.update_calendar()
        
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部控制区域
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 月份导航
        ttk.Button(top_frame, text="上一月", command=self.prev_month).pack(side=tk.LEFT, padx=(0, 10))
        self.month_label = ttk.Label(top_frame, font=("Arial", 16, "bold"))
        self.month_label.pack(side=tk.LEFT, expand=True)
        
        # 年份选择
        ttk.Button(top_frame, text="上一年", command=self.prev_year).pack(side=tk.LEFT, padx=(10, 5))
        self.year_var = tk.StringVar(value=str(self.current_date.year))
        year_combo = ttk.Combobox(top_frame, textvariable=self.year_var, width=6, state="readonly")
        year_combo['values'] = [str(year) for year in range(2020, 2031)]
        year_combo.bind('<<ComboboxSelected>>', self.on_year_change)
        year_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="下一年", command=self.next_year).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(top_frame, text="下一月", command=self.next_month).pack(side=tk.LEFT, padx=(10, 0))
        
        # 日期信息显示区域
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 12))
        self.info_label.pack(anchor=tk.W)
        
        # 日历显示区域
        self.calendar_frame = ttk.Frame(main_frame)
        self.calendar_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建日历标题
        weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        for i, day in enumerate(weekdays):
            label = ttk.Label(self.calendar_frame, text=day, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
        
        # 配置网格权重
        for i in range(7):
            self.calendar_frame.columnconfigure(i, weight=1)
    
    def get_shift_info(self, target_date):
        """根据日期计算班次信息"""
        days_diff = (target_date - self.base_date).days
        target_cycle_pos = (self.base_cycle_pos + days_diff) % self.total_cycle
        if target_cycle_pos < 0:
            target_cycle_pos += self.total_cycle
        shift_index = target_cycle_pos // self.cycle_days
        shift_type = self.shift_cycle[shift_index]
        shift_num = (target_cycle_pos % self.cycle_days) + 1
        return shift_type, shift_num
    
    def update_calendar(self):
        # 清除现有的日历日期
        for widget in self.calendar_frame.winfo_children()[7:]:  # 保留标题行
            widget.destroy()
        
        # 获取当前月份的信息
        year = self.current_date.year
        month = self.current_date.month
        month_name = f"{year}年{month}月"
        self.month_label.config(text=month_name)
        
        # 获取该月第一天是星期几（0=周一, 6=周日）
        first_day_weekday = calendar.monthrange(year, month)[0]
        # 获取该月天数
        days_in_month = calendar.monthrange(year, month)[1]
        
        # 计算日历网格的行数
        rows_needed = (days_in_month + first_day_weekday + 6) // 7
        
        # 创建日历网格
        day = 1
        for row in range(rows_needed):
            for col in range(7):
                if row == 0 and col < first_day_weekday:
                    # 前置空格
                    label = ttk.Label(self.calendar_frame, text="", relief="raised", padding=5)
                    label.grid(row=row+1, column=col, sticky="nsew", padx=1, pady=1)
                elif day <= days_in_month:
                    # 创建日期标签
                    date = datetime(year, month, day)
                    
                    # 获取班次信息
                    shift_type, shift_num = self.get_shift_info(date)
                    shift_info = f"{shift_type}{shift_num}"
                    
                    # 创建文本内容
                    text = f"{day}\n{shift_info}"
                    
                    # 设置颜色
                    fg_color = "black"
                    if date.weekday() == 5 or date.weekday() == 6:  # 周六周日
                        fg_color = "red"
                    
                    label = tk.Label(
                        self.calendar_frame,
                        text=text,
                        relief="raised",
                        padx=5,
                        pady=5,
                        bg="white",
                        fg=fg_color,
                        font=("Arial", 10),
                        wraplength=100,
                        justify=tk.LEFT
                    )
                    
                    # 绑定点击事件显示详细信息
                    label.bind("<Button-1>", lambda e, d=date: self.show_date_info(d))
                    
                    label.grid(row=row+1, column=col, sticky="nsew", padx=1, pady=1)
                    
                    day += 1
                else:
                    # 后置空格
                    label = ttk.Label(self.calendar_frame, text="", relief="raised", padding=5)
                    label.grid(row=row+1, column=col, sticky="nsew", padx=1, pady=1)
    
    def prev_month(self):
        """切换到上一个月"""
        year = self.current_date.year
        month = self.current_date.month
        if month == 1:
            self.current_date = datetime(year-1, 12, 1)
        else:
            self.current_date = datetime(year, month-1, 1)
        self.update_calendar()
    
    def next_month(self):
        """切换到下一个月"""
        year = self.current_date.year
        month = self.current_date.month
        if month == 12:
            self.current_date = datetime(year+1, 1, 1)
        else:
            self.current_date = datetime(year, month+1, 1)
        self.update_calendar()
    
    def prev_year(self):
        """切换到上一年"""
        year = self.current_date.year
        if year > 2020:
            self.current_date = datetime(year-1, self.current_date.month, 1)
            self.update_calendar()
    
    def next_year(self):
        """切换到下一年"""
        year = self.current_date.year
        if year < 2030:
            self.current_date = datetime(year+1, self.current_date.month, 1)
            self.update_calendar()
    
    def on_year_change(self, event):
        """年份选择变化事件"""
        selected_year = int(self.year_var.get())
        current_month = self.current_date.month
        self.current_date = datetime(selected_year, current_month, 1)
        self.update_calendar()
    
    def show_date_info(self, date):
        """显示选中日期的详细信息"""
        date_str = date.strftime("%Y-%m-%d")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.weekday()]
        shift_type, shift_num = self.get_shift_info(date)
        
        info_text = f"阳历: {date_str} ({weekday}) | 班次: {shift_type}{shift_num}"
        self.info_label.config(text=info_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = CalendarApp(root)
    root.mainloop()
