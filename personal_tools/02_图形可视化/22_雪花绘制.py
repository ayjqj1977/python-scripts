import turtle

t = turtle.Turtle()
t.speed(0)
t.pencolor("blue")

def draw_fractal(length, depth):
    if depth == 0:
        t.forward(length)
    else:
        draw_fractal(length/3, depth-1)
        t.left(60)
        draw_fractal(length/3, depth-1)
        t.right(120)
        draw_fractal(length/3, depth-1)
        t.left(60)
        draw_fractal(length/3, depth-1)

t.penup()
t.goto(-200, 0)
t.pendown()
draw_fractal(400, 4)
turtle.done()
