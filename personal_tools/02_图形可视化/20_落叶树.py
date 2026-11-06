import turtle
import random
import time

t = turtle.Turtle()
t.speed(0)
t.left(90)
t.penup()
t.goto(0, -200)
t.pendown()
leaf_positions = []

def draw_tree(length, depth):
    if depth == 0:
        leaf_positions.append(t.pos())
        return
    t.pencolor(random.choice(["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]))
    t.forward(length)
    t.right(30)
    draw_tree(length*0.7, depth-1)
    t.left(60)
    draw_tree(length*0.7, depth-1)
    t.right(30)
    t.backward(length)

# 画树
draw_tree(100, 6)

# 落叶暴雨效果
t.penup()
# 循环多次掉落，每个枝头掉5片叶子
for _ in range(5):
    for pos in leaf_positions:
        # 随机选2-3个枝头同时掉
        if random.randint(1,3) <= 2:
            t.goto(pos)
            t.pencolor(random.choice(["#e67e22", "#f39c12", "#d35400", "#f1c40f"]))
            t.begin_fill()
            t.circle(2)
            t.end_fill()
            # 快速左右摆动下落
            current_y = t.ycor()
            while current_y > -200:
                t.goto(t.xcor() + random.randint(-2,2), current_y - 1)
                current_y = t.ycor()
                time.sleep(0.01)
    # 每轮落叶后停顿
    time.sleep(0.3)

turtle.done()
