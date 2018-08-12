import tkinter as tk
from past.utils import old_div
import torch
import math
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image, ImageDraw


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


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)
