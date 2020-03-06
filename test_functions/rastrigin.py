import torch
import math

from .test_function import TestFunction


class Rastrigin(TestFunction):
    r"""Rastrigin test function.

    Example:
    >>> import test_functions
    >>> rastrigin = test_functions.Rastrigin()
    >>> x = torch.linspace(
            rastrigin.x_domain[0],
            rastrigin.x_domain[1],
            rastrigin.num_pt
        )
    >>> y = torch.linspace(
            rastrigin.y_domain[0],
            rastrigin.y_domain[1],
            rastrigin.num_pt
        )
    >>> Y, X = torch.meshgrid(x, y)
    >>> Z = rastrigin([X, Y])

    __ https://en.wikipedia.org/wiki/Test_functions_for_optimization
    __ https://en.wikipedia.org/wiki/Rastrigin_function
    """
    def __init__(self):
        super(Rastrigin, self).__init__(
            x_domain=(-4.5, 4.5),
            y_domain=(-4.5, 4.5),
            minimum=(0, 0),
            initial_state=(-2.0, 3.5),
            levels=20
        )

    def __call__(self, tensor, lib=torch):
        x, y = tensor
        A = 10
        f = (
                A * 2
                + (x ** 2 - A * lib.cos(x * math.pi * 2))
                + (y ** 2 - A * lib.cos(y * math.pi * 2))
        )
        return f


