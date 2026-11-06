# 1. 导入海龟绘图模块（核心工具库，就像当年Pascal的基础库）
import turtle

# 2. 初始化绘图窗口：setup(宽, 高, 窗口x坐标, 窗口y坐标)，后面两个参数默认居中
turtle.setup(600, 400)  # 设置窗口宽600像素、高400像素，居中显示

# 3. 创建画笔对象（相当于拿起画笔，后续操作都通过这个对象执行）
pen = turtle.Turtle()
pen.speed(3)  # 设置绘图速度：1最慢，10最快，3适合新手观察

# 4. 绘制酒杯（核心用到 pendown() 落笔绘图）
pen.penup()  # 抬笔：移动时不画线（对应 pendown() 的反向操作）
pen.goto(-50, 0)  # 移动到起始位置（x=-50, y=0），类似当年的坐标定位
pen.pendown()  # 落笔：开始画线
pen.circle(30, 180)  # 画半圆（酒杯杯身）：半径30，角度180度
pen.penup()
pen.goto(-50, 0)
pen.pendown()
pen.goto(-50, 100)  # 画酒杯杯柄（垂直线）
pen.goto(-20, 110)  # 画酒杯杯口（水平短线）
pen.goto(20, 110)
pen.goto(50, 100)
pen.goto(50, 0)
pen.circle(-30, 180)  # 画另一侧半圆，完成杯身

# 5. 绘制香酥馋豆（小圆形）
pen.penup()
pen.goto(100, 30)
pen.pendown()
pen.color("brown")  # 设置画笔颜色（馋豆的棕色）
pen.begin_fill()  # 开始填充颜色
pen.circle(10)  # 画圆形（半径10）
pen.end_fill()  # 结束填充

# 6. 绘制五香花生（椭圆形，用两个半圆模拟）
pen.penup()
pen.goto(130, 30)
pen.pendown()
pen.color("goldenrod")  # 花生的金黄色
pen.begin_fill()
pen.circle(12, 180)  # 上半圆
pen.penup()
pen.goto(130, 30)
pen.pendown()
pen.circle(-12, 180)  # 下半圆
pen.end_fill()

# 7. 隐藏画笔，保持窗口显示（避免画完直接关闭）
pen.hideturtle()
turtle.done()  # 固定窗口，直到手动关闭
