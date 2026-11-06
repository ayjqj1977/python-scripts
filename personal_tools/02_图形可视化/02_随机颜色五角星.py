import turtle
import random

def draw_random_color_pentagram():
    # 设置turtle
    t = turtle.Turtle()
    t.speed(5)
    t.pensize(3)
    
    # 设置颜色模式为RGB
    turtle.colormode(255)
    
    # 随机生成颜色
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    # 设置填充颜色和画笔颜色
    t.fillcolor(r, g, b)
    t.pencolor(r, g, b)
    
    # 绘制五角星 - 大小是原来的两倍（200代替100）
    t.penup()
    t.goto(-100, 40)  # 起始位置也相应调整
    t.pendown()
    
    t.begin_fill()
    for _ in range(5):
        t.forward(200)  # 原来是100，现在是200（两倍大小）
        t.right(144)
    t.end_fill()
    
    turtle.done()

# 调用函数绘制随机颜色的五角星
draw_random_color_pentagram()