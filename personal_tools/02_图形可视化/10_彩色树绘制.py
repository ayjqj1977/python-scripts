import turtle
import random

t = turtle.Turtle()
t.speed(0)
t.left(90)
t.penup()
t.goto(0, -200)
t.pendown()

def draw_tree(length, depth):
    if depth == 0:
        return
    t.pencolor(random.choice(["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]))
    t.forward(length)
    t.right(30)
    draw_tree(length*0.7, depth-1)
    t.left(60)
    draw_tree(length*0.7, depth-1)
    t.right(30)
    t.backward(length)

draw_tree(100, 6)
turtle.done()
