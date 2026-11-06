import turtle

# 模板：画笔名 = 新建画笔 → 设置属性 → 执行画图
画笔名 = turtle.Turtle()
画笔名.hideturtle()  # 可选：隐藏箭头
画笔名.pencolor("颜色关键词")  # 替换：red/yellow/green/blue等
画笔名.pensize(数字)  # 替换：1/2/5等（数字越大线越粗）
画笔名.要做的事()  # 替换：goto(100,0)/circle(30)/dot(8)等

turtle.done()
