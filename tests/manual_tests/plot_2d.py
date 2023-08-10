import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch

from auto_LiRPA.bound_ops import (BoundMul, BoundTanh, BoundSigmoid, BoundSin,
                                  BoundCos, BoundAtan, BoundTan, BoundPow)

function = 'pow3'
range_l = -5.
range_u = 5.
x_l = -math.pi
x_u = math.pi


if function == 'x^2':
    def get_bound(x_l, x_u):
        a_l, _, b_l, a_u, _, b_u = BoundMul.get_bound_square(torch.tensor(x_l), torch.tensor(x_u))
        return a_l, b_l, a_u, b_u

    def f(x):
        return x * x
else:
    if function == 'tanh':
        bound = BoundTanh({}, [], 0, {})
        def f(x):
            return np.tanh(x)
    elif function == 'sigmoid':
        bound = BoundSigmoid({}, [], 0, {})
        def f(x):
            return 1 / (1 + np.exp(-x))
    elif function == 'sin':
        range_l = -1. * math.pi
        range_u = 1. * math.pi
        bound = BoundSin({}, [], 0, {})
        def f(x):
            return np.sin(x)
    elif function == 'cos':
        range_l = -4. * math.pi
        range_u = 4. * math.pi
        bound = BoundCos({}, [], 0, {})
        def f(x):
            return np.cos(x)
    elif function == 'arctan':
        range_l = - math.pi
        range_u = math.pi
        bound = BoundAtan({}, [], 0, {})
        def f(x):
            return np.arctan(x)
    elif function == 'pow3':
        range_l = -5.
        range_u = 5.
        bound = BoundPow({}, [], 0, {})
        bound.exponent = 3
        bound.precompute_relaxation(bound.act_func, bound.d_act_func)
        def f(x):
            return x**3
    elif function == 'tan':
        period = -1 * torch.pi
        range_l = -0.5 * torch.pi + 0.1 + period
        range_u = 0.5 * torch.pi - 0.1 + period
        x_l = -1. + period
        x_u = 1. + period
        bound = BoundTan({}, [], 0, {})
        def f(x):
            return np.tan(x)
    else:
        raise NotImplementedError(function)

    def get_bound(x_l, x_u):
        class Input(object):
            def __init__(self, x_l, x_u):
                self.lower = torch.tensor([[x_l]], dtype=torch.float32)
                self.upper = torch.tensor([[x_u]], dtype=torch.float32)
        # Create a fake input object with lower and upper bounds.
        i = Input(x_l, x_u)
        bound.bound_relax(i, init=True)
        return bound.lw.item(), bound.lb.item(), bound.uw.item(), bound.ub.item()

def fu(x, a, b):
    return a * x + b

def fl(x, a, b):
    return a * x + b

# Get initial values.
a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
fig = plt.figure()
# Leave some space below for sliders.
plt.subplots_adjust(bottom=0.25)
ax = fig.gca()
x = np.linspace(range_l, range_u, 1001)
# Plot main function.
plt.plot(x, f(x), color='skyblue', linewidth=1)
y_l, y_u = ax.get_ylim()
# Plot upper and lower bounds.
l_p, = plt.plot(x, fl(x, a_l, b_l), color='olive', linewidth=1, label="lb")
u_p, = plt.plot(x, fu(x, a_u, b_u), color='red', linewidth=1, label="ub")
# Plot two straight lines for ub and lb.
l_pl, = plt.plot(x_l * np.ones_like(x), np.linspace(y_l, y_u, 1001), color='blue', linestyle='dashed', linewidth=1)
u_pl, = plt.plot(x_u * np.ones_like(x), np.linspace(y_l, y_u, 1001), color='blue', linestyle='dashed', linewidth=1)
plt.ylim(y_l, y_u)
plt.legend()

# Create sliders.
axcolor = 'lightgoldenrodyellow'
ax_xl = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_xu = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
s_xl = Slider(ax_xl, 'lb', range_l, range_u, valinit=x_l)
s_xu = Slider(ax_xu, 'ub', range_l, range_u, valinit=x_u)

def update_xu(val):
    # Update upper bound value, and update figure.
    global x_u
    x_u = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    u_pl.set_xdata(x_u * np.ones_like(x))
    fig.canvas.draw_idle()

def update_xl(val):
    # Update lower bound value, and update figure.
    global x_l
    x_l = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

s_xl.on_changed(update_xl)
s_xu.on_changed(update_xu)

plt.show()
