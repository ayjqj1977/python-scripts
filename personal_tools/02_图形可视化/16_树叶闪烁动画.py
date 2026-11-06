# 导入需要的库：turtle（画图）、random（随机）、time（控制时间）
import turtle
import random
import time

# 创建画图的“画笔”对象t，用来画树
tree_t = turtle.Turtle()
# 创建专门画叶子的画笔，避免影响树
leaf_t = turtle.Turtle()
# 设置两个画笔的速度为最快
tree_t.speed(0)
leaf_t.speed(0)
# 把画树的画笔方向向左转90度，让树垂直向上生长
tree_t.left(90)
# 抬起画树的画笔，移动时不留下痕迹
tree_t.penup()
# 把画树的画笔移到(0, -200)的位置，作为树的根部
tree_t.goto(0, -200)
# 放下画树的画笔，准备开始画树
tree_t.pendown()
# 用来存储所有叶子的位置
leaf_positions = []

# 定义画树的函数：length是当前树枝长度，depth是递归深度（控制树枝层数）
def draw_tree(length, depth):
    # 如果递归深度为0，说明到了树枝顶端
    if depth == 0:
        # 把当前位置存入叶子位置列表
        leaf_positions.append(tree_t.pos())
        # 结束当前分支的绘制
        return
    # 给树枝随机选一个颜色（绿色、蓝色、紫色、红色系）
    tree_t.pencolor(random.choice(["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]))
    # 向前画一段树枝
    tree_t.forward(length)
    # 向右转30度，准备画右侧的小树枝
    tree_t.right(30)
    # 递归画右侧的小树枝：长度是当前的0.7倍，深度减1
    draw_tree(length*0.7, depth-1)
    # 向左转60度（抵消之前的右转30度，再转30度向左），准备画左侧的小树枝
    tree_t.left(60)
    # 递归画左侧的小树枝
    draw_tree(length*0.7, depth-1)
    # 向右转30度，回到原来的方向
    tree_t.right(30)
    # 向后退一段距离，回到树枝的起点
    tree_t.backward(length)

# 调用画树函数：初始长度100，递归深度6（控制树的复杂度）
draw_tree(100, 6)

# 把画叶子的画笔抬起，移动时不留下痕迹
leaf_t.penup()
# 开始“星星落叶”效果，树会一直保留
while True:
    # 随机选一个枝头
    pos = random.choice(leaf_positions)
    # 随机选一个星星颜色
    color = random.choice(["#f1c40f", "#ffffff", "#3498db", "#e74c3c"])
    # 把画叶子的画笔移到枝头位置
    leaf_t.goto(pos)
    # 设置星星颜色
    leaf_t.pencolor(color)
    # 开始填充
    leaf_t.begin_fill()
    # 画一个小圆圈作为星星
    leaf_t.circle(2)
    # 结束填充
    leaf_t.end_fill()
    # 暂停0.1秒，让星星闪烁
    time.sleep(0.1)
    # 只清空当前星星，不影响树
    leaf_t.clear()
    # 让星星慢慢下落一点
    pos = (pos[0] + random.randint(-1,1), pos[1] - 0.5)

# 保持画图窗口不关闭，方便查看效果
turtle.done()
