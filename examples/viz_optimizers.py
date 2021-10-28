import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import fmin, hp, tpe

import torch_optimizer as optim

plt.style.use('seaborn-white')

NUM_ITER: int = 500
NUM_ITER_HPARAM: int = 200


def rosenbrock(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f


def execute_steps(
    func, initial_state, optimizer_class, optimizer_config, num_iter=NUM_ITER
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


def objective_rastrigin(params):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    kwargs = params['kwargs']
    initial_state = (-2.0, 3.5)
    minimum = (0, 0)
    optimizer_config = dict(lr=lr, **kwargs)
    num_iter = NUM_ITER_HPARAM
    steps = execute_steps(
        rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_rosenbrok(params):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    kwargs = params['kwargs']
    minimum = (1.0, 1.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr, **kwargs)
    num_iter = NUM_ITER_HPARAM
    steps = execute_steps(
        rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_rastrigin(grad_iters, optimizer_name, lr):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)
    assert len(grad_iters) <= 3, "Cannot handle more than three states"
    l = None
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    plt.contour(X, Y, Z, 20, cmap='jet', alpha=0.75)
    for grad_iter, color in zip(grad_iters, ['r', 'm', 'c']):
        iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
        if l is None:
            l = len(iter_x)
        ax.plot(iter_x, iter_y, color=color, marker=None, alpha=0.75)
        for px, py, pdx, pdy in zip(
            iter_x[:-1],
            iter_y[:-1],
            iter_x[1:] - iter_x[:-1],
            iter_y[1:] - iter_y[:-1],
        ):
            ax.arrow(
                x=px,
                y=py,
                dx=pdx,
                dy=pdy,
                overhang=0.5,
                width=0.001,
                head_width=0.08,
                length_includes_head=True,
                color=color,
                visible=True,
            )
        # Starting point
        ax.plot(
            iter_x[0],
            iter_y[0],
            marker="s",
            markersize=11,
            markeredgecolor="black",
            markerfacecolor=color,
            markeredgewidth=2,
        )
        # Ending point
        ax.plot(
            iter_x[-1],
            iter_y[-1],
            marker="P",
            markersize=11,
            markeredgecolor="black",
            markerfacecolor=color,
            markeredgewidth=2,
        )
    plt.title(
        'Rastrigin func: {} with '
        '{} iterations, lr={:.6}'.format(optimizer_name, l, lr)
    )
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.plot(*minimum, 'X', color="green", markersize=11)
    plt.savefig('docs/rastrigin_{}.png'.format(optimizer_name))
    plt.close()


def plot_rosenbrok(grad_iters, optimizer_name, lr):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)
    assert len(grad_iters) <= 3, "Cannot handle more than three states"

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    l = None
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet', alpha=0.75)
    for grad_iter, color in zip(grad_iters, ['r', 'm', 'c']):
        iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
        if l is None:
            l = len(iter_x)
        ax.plot(iter_x, iter_y, color=color, marker=None, alpha=0.75)

        for px, py, pdx, pdy in zip(
            iter_x[:-1],
            iter_y[:-1],
            iter_x[1:] - iter_x[:-1],
            iter_y[1:] - iter_y[:-1],
        ):
            ax.arrow(
                x=px,
                y=py,
                dx=pdx,
                dy=pdy,
                overhang=0.5,
                width=0.001,
                head_width=0.0375,
                length_includes_head=True,
                color=color,
                visible=True,
            )
        # Starting point
        ax.plot(
            iter_x[0],
            iter_y[0],
            marker="s",
            markersize=11,
            markeredgecolor="black",
            markerfacecolor=color,
            markeredgewidth=2,
        )
        # Ending point
        ax.plot(
            iter_x[-1],
            iter_y[-1],
            marker="P",
            markersize=11,
            markeredgecolor="black",
            markerfacecolor=color,
            markeredgewidth=2,
        )
    plt.title(
        'Rosenbrock func: {} with {} '
        'iterations, lr={:.6}'.format(optimizer_name, l, lr)
    )
    plt.plot(*minimum, 'X', color="green", markersize=11)
    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.savefig('docs/rosenbrock_{}.png'.format(optimizer_name))
    plt.close()


def execute_experiments(
    optimizers,
    objective,
    func,
    plot_func,
    initial_states: List[Tuple[float, float]],
    seed=1,
):
    seed = seed
    for item in optimizers:
        optimizer_class, lr_low, lr_hi, kwargs, extra_desc = item
        extra_desc_str = '' if not extra_desc else f'_{extra_desc}'
        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
            'kwargs': kwargs,
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.RandomState(seed),
        )
        print(best['lr'], optimizer_class)
        steps_lst = []
        for initial_state in initial_states:
            steps = execute_steps(
                func,
                initial_state,
                optimizer_class,
                {'lr': best['lr'], **kwargs},
                num_iter=NUM_ITER,
            )
            steps_lst.append(steps)
        plot_func(
            steps_lst,
            f'{optimizer_class.__name__}{extra_desc_str}',
            best['lr'],
        )


def LookaheadYogi(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)


if __name__ == '__main__':
    # python examples/viz_optimizers.py

    # Each optimizer has tweaked search space to produce better plots and
    # help to converge on better lr faster.
    optimizers = [
        # baselines
        (torch.optim.Adam, -8, 0.5, {}, None),
        (torch.optim.SGD, -8, -1.0, {}, None),
        # Adam based
        (optim.Adam, -8, 0.5, {}, 'internal'),
        (
            optim.Adam,
            -8,
            0.5,
            {'adamd_bias_correction': True},
            'internal_adamD',
        ),
        (optim.AdamW, -8, 0.5, {}, 'internal'),
        (
            optim.AdamW,
            -8,
            0.5,
            {'adamd_bias_correction': True},
            'internal_adamD',
        ),
        (optim.AdaBound, -8, 0.3, {}, None),
        (optim.AdaBound, -8, 0.3, {'adamd_bias_correction': True}, 'adamD'),
        # TODO
        (optim.Adahessian, -1, 8, {}, None),
        (optim.Adahessian, -1, 8, {'adamd_bias_correction': True}, 'adamD'),
        (optim.AdaMod, -8, 0.2, {}, None),
        (optim.AdaMod, -8, 0.2, {'adamd_bias_correction': True}, 'adamD'),
        (optim.AdamP, -8, 0.2, {}, None),
        (optim.AdamP, -8, 0.2, {'adamd_bias_correction': True}, 'adamD'),
        (optim.DiffGrad, -8, 0.4, {}, None),
        (optim.DiffGrad, -8, 0.4, {'adamd_bias_correction': True}, 'adamD'),
        (optim.Lamb, -8, -2.9, {}, None),
        (
            optim.Lamb,
            -8,
            -2.9,
            {'debias': True, 'adamd_bias_correction': True},
            'adamD',
        ),
        (optim.MADGRAD, -8, 0.5, {}, None),
        (optim.NovoGrad, -8, -1.7, {}, None),
        (optim.Yogi, -8, 0.1, {}, None),
        (optim.Yogi, -8, 0.1, {'adamd_bias_correction': True}, 'adamD'),
        # SGD/Momentum based
        (optim.AccSGD, -8, -1.4, {}, None),
        (optim.SGDW, -8, -1.5, {}, None),
        (optim.SGDP, -8, -1.5, {}, None),
        (optim.PID, -8, -1.0, {}, None),
        (optim.QHM, -6, -0.2, {}, None),
        (optim.QHAdam, -8, 0.1, {}, None),
        (optim.Ranger, -8, 0.1, {}, None),
        (optim.RangerQH, -8, 0.1, {}, None),
        (optim.RangerVA, -8, 0.1, {}, None),
        (optim.Shampoo, -8, 0.1, {}, None),
        (LookaheadYogi, -8, 0.1, {}, None),
        (optim.AggMo, -8, -1.5, {}, None),
        (optim.SWATS, -8, -1.5, {}, None),
        (optim.SWATS, -8, -1.5, {'adamd_bias_correction': True}, 'adamD'),
        (optim.Adafactor, -8, 0.5, {}, None),
        (optim.A2GradUni, -8, 0.1, {}, None),
        (optim.A2GradInc, -8, 0.1, {}, None),
        (optim.A2GradExp, -8, 0.1, {}, None),
        (optim.AdaBelief, -8, 0.1, {}, None),
        (optim.AdaBelief, -8, 0.1, {'adamd_bias_correction': True}, 'adamD'),
        (optim.Apollo, -8, 0.1, {}, None),
    ]
    execute_experiments(
        optimizers,
        objective_rastrigin,
        rastrigin,
        plot_rastrigin,
        [(-2.0, 3.5), (1.0, -2.0)],
    )

    execute_experiments(
        optimizers,
        objective_rosenbrok,
        rosenbrock,
        plot_rosenbrok,
        [(-2.0, 2.0), (-0.5, 2.75)],
    )
