import time
import random
from colorama import Fore, Style

# 改进版彩色数字金字塔，输出更规整
def print_colorful_number_pyramid():
    # 打印金字塔的上半部分（1到5）
    for i in range(1, 6):
        # 随机选择颜色
        color = random.choice([Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA])
        spaces = ' ' * (5 - i)  # 计算前导空格数
        digits = ' '.join([str(i)] * i)  # 用空格连接重复的数字
        print(color + spaces + digits + Style.RESET_ALL)
        time.sleep(0.5)
    
    # 打印金字塔的下半部分（6到9）
    for i in range(6, 10):
        # 随机选择颜色
        color = random.choice([Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA])
        spaces = ' ' * (i - 5)  # 计算前导空格数
        digits = ' '.join([str(i)] * (10 - i))  # 用空格连接重复的数字
        print(color + spaces + digits + Style.RESET_ALL)
        time.sleep(0.5)

# 调用函数打印彩色数字金字塔
print_colorful_number_pyramid()