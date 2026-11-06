from datetime import datetime
import re

# 定义排班循环：早班、晚班、中班，各3天
shift_cycle = ['早班', '晚班', '中班']
cycle_days = 3  # 每个班次的天数
total_cycle = len(shift_cycle) * cycle_days  # 总循环周期：9天

# 基准日期：2025-12-01，是第二个早班
base_date = datetime(2025, 12, 1)
base_cycle_pos = 1  # 基准日在总循环中的位置（0开始：早班1=0，早班2=1，早班3=2，晚班1=3...）

def parse_date(input_str):
    """解析输入的日期，支持空格/横线分隔、1和01等效，校验年份≥2020"""
    # 替换空格为-，统一格式
    date_str = re.sub(r'\s+', '-', input_str.strip())
    parts = date_str.split('-')
    if len(parts) != 3:
        raise ValueError("日期需包含年、月、日")
    year = parts[0].zfill(4)
    month = parts[1].zfill(2)
    day = parts[2].zfill(2)
    if int(year) < 2020:
        raise ValueError("仅支持查询2020年及以后的日期")
    return datetime(int(year), int(month), int(day))

def get_shift_info(target_date):
    """根据日期计算班次信息"""
    days_diff = (target_date - base_date).days
    target_cycle_pos = (base_cycle_pos + days_diff) % total_cycle
    if target_cycle_pos < 0:
        target_cycle_pos += total_cycle
    shift_index = target_cycle_pos // cycle_days
    shift_type = shift_cycle[shift_index]
    shift_num = (target_cycle_pos % cycle_days) + 1
    return shift_type, shift_num

# 循环查询主逻辑
print("排班查询工具（支持2020年及以后日期，输入'q'退出）")
while True:
    user_input = input("\n请输入查询日期（格式：年 月 日 或 年-月-日）：")
    # 输入q/quit退出程序
    if user_input.strip().lower() in ['q', 'quit', '退出']:
        print("程序已退出！")
        break
    try:
        target_date = parse_date(user_input)
        shift_type, shift_num = get_shift_info(target_date)
        print(f"{target_date.strftime('%Y年%m月%d日')} 的班次：{shift_type}{shift_num}")
    except ValueError as e:
        print(f"输入错误：{e}，请重新输入！")
    except Exception as e:
        print(f"日期解析失败：{e}，示例输入：2025 12 1 或 2020-01-01")