import math
import numpy as np
import torch_optimizer as optim
import torch
from hyperopt import fmin, tpe, hp
import matplotlib.pyplot as plt
from test_functions import *

plt.style.use('seaborn-white')


def ackley(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    f = (
        -20 * lib.exp(-0.2 * lib.sqrt(0.5 * (x * x + y * y)))
        - lib.exp(0.5 * lib.cos(2 * math.pi * x) + lib.cos(2 * math.pi * y))
        + math.e + 20
    )
    return f


def execute_steps(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 2.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


# def objective_rastrigin(params):
#     lr = params['lr']
#     optimizer_class = params['optimizer_class']
#     initial_state = (-2.0, 3.5)
#     minimum = (0, 0)
#     optimizer_config = dict(lr=lr)
#     num_iter = 100
#     steps = execute_steps(
#         rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
#     )
#     return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2
#
#
# def objective_rosenbrok(params):
#     lr = params['lr']
#     optimizer_class = params['optimizer_class']
#     minimum = (1.0, 1.0)
#     initial_state = (-2.0, 2.0)
#     optimizer_config = dict(lr=lr)
#     num_iter = 100
#     steps = execute_steps(
#         rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
#     )
#     return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2
#
#
# def objective_ackley(params):
#     lr = params['lr']
#     optimizer_class = params['optimizer_class']
#     minimum = (0, 0)
#     initial_state = (-2.0, 4.0)
#     optimizer_config = dict(lr=lr)
#     num_iter = 100
#     steps = execute_steps(
#         ackley, initial_state, optimizer_class, optimizer_config, num_iter
#     )
#     return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective(params, func):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    minimum = func.minimum
    initial_state = func.initial_state
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        ackley, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


# def plot_rastrigin(grad_iter, optimizer_name, lr):
#     x = np.linspace(-4.5, 4.5, 250)
#     y = np.linspace(-4.5, 4.5, 250)
#     minimum = (0, 0)
#
#     X, Y = np.meshgrid(x, y)
#     Z = rastrigin([X, Y], lib=np)
#
#     iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
#
#     fig = plt.figure(figsize=(8, 8))
#
#     ax = fig.add_subplot(1, 1, 1)
#     ax.contour(X, Y, Z, 20, cmap='jet')
#     ax.plot(iter_x, iter_y, color='r', marker='x')
#     ax.set_title(
#         f'Rastrigin func: {optimizer_name} with '
#         f'{len(iter_x)} iterations, lr={lr:.6}'
#     )
#     plt.plot(*minimum, 'gD')
#     plt.plot(iter_x[-1], iter_y[-1], 'rD')
#     plt.savefig(f'docs/rastrigin_{optimizer_name}.png')
#
#
# def plot_rosenbrok(grad_iter, optimizer_name, lr):
#     x = np.linspace(-2, 2, 250)
#     y = np.linspace(-1, 3, 250)
#     minimum = (1.0, 1.0)
#
#     X, Y = np.meshgrid(x, y)
#     Z = rosenbrock([X, Y])
#
#     iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
#
#     fig = plt.figure(figsize=(8, 8))
#
#     ax = fig.add_subplot(1, 1, 1)
#     ax.contour(X, Y, Z, 90, cmap='jet')
#     ax.plot(iter_x, iter_y, color='r', marker='x')
#
#     ax.set_title(
#         f'Rosenbrock func: {optimizer_name} with {len(iter_x)} '
#         f'iterations, lr={lr:.6}'
#     )
#     plt.plot(*minimum, 'gD')
#     plt.plot(iter_x[-1], iter_y[-1], 'rD')
#     plt.savefig(f'docs/rosenbrock_{optimizer_name}.png')


def plot(func, grad_iter, optimizer_name, lr):
    x = torch.linspace(func.x_domain[0], func.x_domain[1], func.num_pt)
    y = torch.linspace(func.y_domain[0], func.y_domain[1], func.num_pt)
    minimum = func.minimum

    Y, X = torch.meshgrid(x, y)
    Z = func([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, func.levels, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')

    func_name = func.__name__()
    ax.set_title(
        f'{func_name} func: {optimizer_name} with {len(iter_x)} '
        f'iterations, lr={lr:.6}'
    )
    plt.plot(*minimum, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'rD')
    plt.savefig(f'docs/{func_name}_{optimizer_name}.png')


def execute_experiments(
    optimizers, func, seed=1
):
    seed = seed
    for item in optimizers:
        optimizer_class, lr_low, lr_hi = item
        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
        }
        best = fmin(
            fn=lambda x: objective(x, func),
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.RandomState(seed),
        )
        print(best['lr'], optimizer_class)

        steps = execute_steps(
            func,
            func.initial_state,
            optimizer_class,
            {'lr': best['lr']},
            num_iter=500,
        )
        plot(func, steps, optimizer_class.__name__, best['lr'])


if __name__ == '__main__':
    # python examples/viz_optimizers.py

    # Each optimizer has tweaked search space to produce better plots and
    # help to converge on better lr faster.
    optimizers = [
        # Adam based
        (optim.AdaBound, -8, 0.3),
        (optim.AdaMod, -8, 0.2),
        (optim.DiffGrad, -8, 0.4),
        (optim.Lamb, -8, -2.9),
        (optim.NovoGrad, -8, -1.7),
        (optim.RAdam, -8, 0.5),
        (optim.Yogi, -8, 0.1),
        # SGD/Momentum based
        (optim.AccSGD, -8, -1.4),
        (optim.SGDW, -8, -1.5),
        (optim.PID, -8, -1.0),
    ]

    for test_func in [Rastrigin(), Rosenbrock()]:
        execute_experiments(
            optimizers,
            test_func,
        )