import turtle
import random

t = turtle.Turtle()
t.speed(0)
screen = turtle.Screen()
screen.colormode(255)

for _ in range(100):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    t.pencolor(r, g, b)
    t.forward(_ * 2)
    t.right(89)

turtle.done()
