import turtle

def draw_centered_color_squares():
    # 设置turtle模式
    turtle.colormode(255)
    square_turtle = turtle.Turtle()
    square_turtle.speed(0)  # 最快速度

    # 定义4种不同颜色（RGB值），对应4个正方形
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0)   # 黄色
    ]

    # 设置方块大小和间距
    square_size = 100
    gap = 50
    move_distance = square_size + gap

    # 计算起始位置，使整个2x2方块组居中
    start_x = -move_distance / 2
    start_y = move_distance / 2

    # 移动到起始位置
    square_turtle.penup()
    square_turtle.goto(start_x, start_y)
    square_turtle.pendown()

    # 外层循环控制"行"：2行（i=0是第一行，i=1是第二行）
    for row in range(2):
        # 保存当前行的起始y坐标
        current_y = start_y - row * move_distance
        
        # 内层循环控制"列"：2列（j=0是第一列，j=1是第二列）
        for col in range(2):
            # 计算当前正方形的索引（0→1→2→3），对应颜色列表
            index = row * 2 + col
            square_turtle.fillcolor(colors[index])  # 用列表里的独特色彩
            square_turtle.pencolor(0, 0, 0)  # 黑色边框，更清晰
            
            # 计算当前方块的x坐标
            current_x = start_x + col * move_distance
            
            # 移动到当前方块位置
            square_turtle.penup()
            square_turtle.goto(current_x, current_y)
            square_turtle.pendown()
            
            # 画正方形（4条边）
            square_turtle.begin_fill()
            for _ in range(4):
                square_turtle.forward(square_size)
                square_turtle.right(90)
            square_turtle.end_fill()

    # 完成绘制
    turtle.done()

# 调用函数绘制居中的彩色方块
draw_centered_color_squares()
