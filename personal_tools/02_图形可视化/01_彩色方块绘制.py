import turtle
def draw_square():
    a=turtle.Turtle()
    a.pensize(5)
    a.speed(4)
    a.pendown()
    a.goto(0,100)
    a.goto(100,100)
    a.goto(100,0)
    a.goto(0,0)