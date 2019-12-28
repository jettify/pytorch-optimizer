import torch
from torch_optimizer import Lookahead
from torch.autograd import Variable
from torch.optim import Adam


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2


def quadratic(tensor):
    x, y = tensor
    a = 1.0
    b = 1.0
    return (x ** 2) / a + (y ** 2) / b


def saddle(tensor):
    x, y = tensor
    a = 1.0
    b = 1.0
    return (x ** 2) / a - (y ** 2) / b


def test_rosenbrock():
    X = Variable(torch.Tensor([1.5, 1.5]), requires_grad=True)
    optimizer = Lookahead(Adam([X], lr=0.01))
    optimizer = Adam([X], lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        f = saddle(X)
        f.backward(retain_graph=True)
        optimizer.step()
        print(f, X)
