import torch

from .test_function import TestFunction


class Rosenbrock(TestFunction):
    r"""Rosenbrock test function.

    Example:
    >>> import test_functions
    >>> rosenbrock = test_functions.Rosenbrock()
    >>> x = torch.linspace(
            rosenbrock.x_domain[0], 
            rosenbrock.x_domain[1],
            rosenbrock.num_pt
        )
    >>> y = torch.linspace(
            rosenbrock.y_domain[0],
            rosenbrock.y_domain[1],
            rosenbrock.num_pt
        )
    >>> Y, X = torch.meshgrid(x, y)
    >>> Z = rosenbrock([X, Y])

    __ https://en.wikipedia.org/wiki/Test_functions_for_optimization
    __ https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    def __init__(self):
        super(Rosenbrock, self).__init__(
            x_domain=(-2.0, 2.0),
            y_domain=(-1.0, 3.0),
            minimum=(1.0, 1.0),
            initial_state=(-2.0, 2.0),
            levels=90,
        )

    def __call__(self, tensor, lib=torch):
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
