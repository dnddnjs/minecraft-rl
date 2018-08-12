import tkinter as tk
from past.utils import old_div
import torch
import math
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image, ImageDraw
import random

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


def init_map():
    root = tk.Tk()
    root.wm_title("Collect the " + GOAL_TYPE + "s, dodge the " + MOB_TYPE + "s!")

    canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                       borderwidth=0, highlightthickness=0, bg="black")
    canvas.pack()
    root.update()
    return root, canvas


def canvasX(x):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(x, float(ARENA_WIDTH))) * (
                       CANVAS_WIDTH - CANVAS_BORDER)


def canvasY(y):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(y, float(ARENA_BREADTH))) * (
                       CANVAS_HEIGHT - CANVAS_BORDER)


def drawMobs(entities):
    img = Image.new("RGB", (CANVAS_WIDTH, int(CANVAS_HEIGHT)), "black")
    img_pil = ImageDraw.Draw(img)
    img_pil.rectangle(((canvasX(old_div(-ARENA_WIDTH, 2)),
                        canvasY(old_div(-ARENA_BREADTH, 2))),
                       (canvasX(old_div(ARENA_WIDTH, 2)),
                        canvasY(old_div(ARENA_BREADTH, 2)))), fill="#888888")
    for ent in entities:
        if ent["name"] == MOB_TYPE:
            img_pil.ellipse((canvasX(ent["x"]) - 2, canvasY(ent["z"]) - 2,
                            canvasX(ent["x"]) + 2, canvasY(ent["z"]) + 2),
                            fill="#ff2244")
        elif ent["name"] == GOAL_TYPE:
            img_pil.ellipse((canvasX(ent["x"]) - 3, canvasY(ent["z"]) - 3,
                            canvasX(ent["x"]) + 3, canvasY(ent["z"]) + 3),
                            fill="#4422ff")
        else:
            img_pil.ellipse((canvasX(ent["x"]) - 4, canvasY(ent["z"]) - 4,
                            canvasX(ent["x"]) + 4, canvasY(ent["z"]) + 4),
                            fill="#22ff44")

    '''
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
    
    '''
    return img


def pre_process(image):
    image = np.array(image)
    image = resize(image, (84, 84, 3))
    image = rgb2gray(image)
    # img = np.uint8(image * 255)
    # img = Image.fromarray(image.astype('uint8'), mode='L')
    # img.show()
    return image


def to_tensor_long(numpy_array):
    if torch.cuda.is_available():
        variable = torch.LongTensor(numpy_array).cuda()
    else:
        variable = torch.LongTensor(numpy_array).cpu()
    return variable


def to_tensor(numpy_array):
    if torch.cuda.is_available():
        variable = torch.Tensor(numpy_array).cuda()
    else:
        variable = torch.Tensor(numpy_array).cpu()
    return variable


def get_action(epsilon, q_valu, num_action):
    if np.random.rand() <= epsilon:
        return random.randrange(num_action)
    else:
        _, action = torch.max(q_value, 1)
        return action


# weight xavier initialize
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)


def update_target_model(model, target_model):
    target_model.load_state_dict(model.state_dict())
