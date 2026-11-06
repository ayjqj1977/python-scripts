import turtle

# 1. 初始化（调整窗口布局）
turtle.setup(width=800, height=700)
board = turtle.Turtle()
board.speed(0)
board.hideturtle()
grid_size = 30
edge = 9 * grid_size
current_color = "black"
stones = []
button_texts = []
occupied_positions = set()

# 2. 绘制棋盘（和按钮完全分离）
def draw_board():
    # 木质底色
    board.penup()
    board.goto(-edge - 10, -edge - 10 + 50)  # 下移棋盘，给按钮留空间
    board.pendown()
    board.color("#E6D8B8")
    board.begin_fill()
    for _ in range(2):
        board.forward(2*edge + 20)
        board.left(90)
        board.forward(2*edge + 20)
        board.left(90)
    board.end_fill()
    # 网格线
    board.color("#8B4513")
    for y in range(-edge, edge + 1, grid_size):
        board.penup()
        board.goto(-edge, y + 50)
        board.pendown()
        board.goto(edge, y + 50)
    for x in range(-edge, edge + 1, grid_size):
        board.penup()
        board.goto(x, -edge + 50)
        board.pendown()
        board.goto(x, edge + 50)
    # 星位
    board.color("black")
    star_positions = [(-6*grid_size, -6*grid_size), (6*grid_size, 6*grid_size),
                      (-6*grid_size, 6*grid_size), (6*grid_size, -6*grid_size), (0, 0)]
    for (x, y) in star_positions:
        board.penup()
        board.goto(x, y + 50)
        board.pendown()
        board.circle(3)
    # 按钮竖排放在右上角（远离棋盘）
    draw_button(320, 200, "悔棋", "blue")
    draw_button(320, 140, "清空棋盘", "red")

# 3. 按钮绘制（竖排小尺寸）
def draw_button(x, y, text, color):
    btn = turtle.Turtle()
    btn.speed(0)
    btn.hideturtle()
    btn.penup()
    btn.goto(x - 35, y - 15)
    btn.pendown()
    btn.color(color)
    btn.begin_fill()
    for _ in range(2):
        btn.forward(70)
        btn.left(90)
        btn.forward(30)
        btn.left(90)
    btn.end_fill()
    text_turtle = turtle.Turtle()
    text_turtle.speed(0)
    text_turtle.hideturtle()
    text_turtle.penup()
    text_turtle.goto(x, y - 10)
    text_turtle.color("white")
    text_turtle.write(text, align="center", font=("SimHei", 10, "bold"))
    button_texts.append(text_turtle)

# 4. 落子功能（所有格点都可用）
def place_stone(x, y):
    global current_color
    # 只响应棋盘区域
    if x > edge + 10 or x < -edge -10 or y > edge + 60 or y < -edge + 40:
        return
    aligned_x = round(x / grid_size) * grid_size
    aligned_y = round((y - 50) / grid_size) * grid_size  # 匹配棋盘下移后的坐标
    if (aligned_x, aligned_y) in occupied_positions:
        return
    # 绘制棋子
    stone = turtle.Turtle()
    stone.speed(0)
    stone.hideturtle()
    stone.penup()
    stone.goto(aligned_x, aligned_y + 50 - grid_size/2)  # 匹配棋盘下移
    stone.color(current_color)
    stone.begin_fill()
    stone.circle(grid_size/2 - 1)
    stone.end_fill()
    stones.append((stone, current_color, (aligned_x, aligned_y)))
    occupied_positions.add((aligned_x, aligned_y))
    current_color = "white" if current_color == "black" else "black"

# 5. 悔棋+清空功能
def undo_last_stone(x, y):
    if 285 < x < 355 and 185 < y < 215 and stones:
        last_stone, last_color, last_pos = stones.pop()
        last_stone.clear()
        occupied_positions.remove(last_pos)
        global current_color
        current_color = last_color

def clear_board(x, y):
    if 285 < x < 355 and 125 < y < 155:
        global stones, current_color, button_texts, occupied_positions
        for stone, _, _ in stones:
            stone.clear()
        for text_turtle in button_texts:
            text_turtle.clear()
        button_texts = []
        stones = []
        occupied_positions.clear()
        current_color = "black"
        draw_button(320, 200, "悔棋", "blue")
        draw_button(320, 140, "清空棋盘", "red")

# 6. 绑定事件
draw_board()
turtle.onscreenclick(place_stone)
turtle.onscreenclick(undo_last_stone, add=True)
turtle.onscreenclick(clear_board, add=True)
turtle.done()
