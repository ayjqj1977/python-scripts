import time

# 改进版数字金字塔，输出更规整，带延时效果
def print_number_pyramid_with_delay():
    # 打印金字塔的上半部分（1到5）
    for i in range(1, 6):
        spaces = ' ' * (5 - i)  # 计算前导空格数
        digits = ' '.join([str(i)] * i)  # 用空格连接重复的数字
        print(spaces + digits)
        time.sleep(1)
    
    # 打印金字塔的下半部分（6到9）
    for i in range(6, 10):
        spaces = ' ' * (i - 5)  # 计算前导空格数
        digits = ' '.join([str(i)] * (10 - i))  # 用空格连接重复的数字
        print(spaces + digits)
        time.sleep(1)

# 调用函数打印数字金字塔
print_number_pyramid_with_delay()