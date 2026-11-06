# 改进版数字金字塔，输出更规整
def print_number_pyramid():
    # 打印金字塔的上半部分（1到5）
    for i in range(1, 6):
        spaces = ' ' * (5 - i)  # 计算前导空格数
        digits = str(i) * i     # 重复当前数字i次
        print(spaces + digits)
    
    # 打印金字塔的下半部分（6到9）
    for i in range(6, 10):
        spaces = ' ' * (i - 5)  # 计算前导空格数
        digits = str(i) * (10 - i)  # 重复数字次数递减
        print(spaces + digits)

# 调用函数打印数字金字塔
print_number_pyramid()