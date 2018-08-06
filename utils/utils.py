import tkinter as tk
from past.utils import old_div


GOAL_TYPE = "apple"
MOB_TYPE = "Endermite"
ARENA_WIDTH = 60
ARENA_BREADTH = 60

# Display parameters:
CANVAS_BORDER = 20
CANVAS_WIDTH = 400
CANVAS_HEIGHT = CANVAS_BORDER + (
            (CANVAS_WIDTH - CANVAS_BORDER) * ARENA_BREADTH / ARENA_WIDTH)
CANVAS_SCALEX = old_div((CANVAS_WIDTH - CANVAS_BORDER), ARENA_WIDTH)
CANVAS_SCALEY = old_div((CANVAS_HEIGHT - CANVAS_BORDER), ARENA_BREADTH)
CANVAS_ORGX = old_div(-ARENA_WIDTH, CANVAS_SCALEX)
CANVAS_ORGY = old_div(-ARENA_BREADTH, CANVAS_SCALEY)


root = tk.Tk()
root.wm_title("Collect the " + GOAL_TYPE + "s, dodge the " + MOB_TYPE + "s!")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                   borderwidth=0, highlightthickness=0, bg="black")
canvas.pack()
root.update()


def canvasX(x):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(x, float(ARENA_WIDTH))) * (
                       CANVAS_WIDTH - CANVAS_BORDER)


def canvasY(y):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(y, float(ARENA_BREADTH))) * (
                       CANVAS_HEIGHT - CANVAS_BORDER)


def drawMobs(entities):
    canvas.delete("all")
    canvas.create_rectangle(canvasX(old_div(-ARENA_WIDTH, 2)),
                            canvasY(old_div(-ARENA_BREADTH, 2)),
                            canvasX(old_div(ARENA_WIDTH, 2)),
                            canvasY(old_div(ARENA_BREADTH, 2)), fill="#888888")
    for ent in entities:
        if ent["name"] == MOB_TYPE:
            canvas.create_oval(canvasX(ent["x"]) - 2, canvasY(ent["z"]) - 2,
                               canvasX(ent["x"]) + 2, canvasY(ent["z"]) + 2,
                               fill="#ff2244")
        elif ent["name"] == GOAL_TYPE:
            canvas.create_oval(canvasX(ent["x"]) - 3, canvasY(ent["z"]) - 3,
                               canvasX(ent["x"]) + 3, canvasY(ent["z"]) + 3,
                               fill="#4422ff")
        else:
            canvas.create_oval(canvasX(ent["x"]) - 4, canvasY(ent["z"]) - 4,
                               canvasX(ent["x"]) + 4, canvasY(ent["z"]) + 4,
                               fill="#22ff44")
    root.update()