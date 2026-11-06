import math

while True:
    print("科学计算（输入：sqrt(25)、sin(90)、pi 等，输入q退出）：")
    expr = input("输入：")
    if expr.lower() == 'q':
        break
    # 替换快捷指令
    expr = expr.replace("sqrt(", "math.sqrt(")
    expr = expr.replace("sin(", "math.sin(math.radians(")
    expr = expr.replace("cos(", "math.cos(math.radians(")
    expr = expr.replace("tan(", "math.tan(math.radians(")
    expr = expr.replace("pi", "math.pi")
    expr = expr.replace("e", "math.e")
    try:
        result = eval(expr)
        print(f"结果：{result}")
    except:
        print("输入错误，请重新输入～")
