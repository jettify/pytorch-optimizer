import torch
import pytest

from torch_optimizer import DiffGrad, AdaMod
from torch.autograd import Variable


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2


def quadratic(tensor):
    x, y = tensor
    a = 1.0
    b = 1.0
    return (x ** 2) / a + (y ** 2) / b


def beale(tensor):
    x, y = tensor
    f = (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y ** 2) ** 2
        + (2.625 - x + x * y ** 3) ** 2
    )
    return f


cases = [
    (rosenbrock, (1.5, 1.5), (1, 1)),
    (quadratic, (1.5, 1.5), (0, 0)),
    (beale, (1.5, 1.5), (3, 0.5)),
]


def ids(v):
    n = f'{v[0].__name__} {v[1:]}'
    return n


optimizers = [(DiffGrad, 0.5), (AdaMod, 1.9)]


@pytest.mark.parametrize('case', cases, ids=ids)
@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_rosenbrock(case, optimizer_config):
    func, initial_state, min_loc = case
    x = Variable(torch.Tensor(initial_state), requires_grad=True)
    x_min = torch.Tensor(min_loc)
    optimizer_class, lr = optimizer_config
    optimizer = optimizer_class([x], lr=lr)
    for _ in range(800):
        optimizer.zero_grad()
        f = func(x)
        f.backward(retain_graph=True)
        optimizer.step()
    assert torch.allclose(x, x_min, atol=0.00001)
