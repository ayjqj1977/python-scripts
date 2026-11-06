import turtle

# 设置绘图参数
t = turtle.Turtle()
screen = turtle.Screen()
screen.title("方波动画")
screen.setup(width=800, height=400)
t.speed(3)  # 放慢速度，方便看绘制过程
t.penup()
t.goto(-350, 0)
t.pendown()

# 绘制方波（2个周期）
period = 200  # 周期长度
amplitude = 100  # 振幅
for _ in range(2):
    # 上升到高电平
    t.goto(t.xcor(), amplitude/2)
    # 保持高电平
    t.goto(t.xcor() + period/2, amplitude/2)
    # 下降到低电平
    t.goto(t.xcor(), -amplitude/2)
    # 保持低电平
    t.goto(t.xcor() + period/2, -amplitude/2)

# 回到中线
t.goto(t.xcor(), 0)

turtle.done()
