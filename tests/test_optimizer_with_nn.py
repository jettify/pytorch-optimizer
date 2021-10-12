import numpy as np
import pytest
import torch
from torch import nn

import torch_optimizer as optim


def make_dataset(seed=42):
    rng = np.random.RandomState(seed)
    N = 100
    D = 2

    X = rng.randn(N, D) * 2

    # center the first N/2 points at (-2,-2)
    mid = N // 2
    X[:mid, :] = X[:mid, :] - 2 * np.ones((mid, D))

    # center the last N/2 points at (2, 2)
    X[mid:, :] = X[mid:, :] + 2 * np.ones((mid, D))

    # labels: first N/2 are 0, last N/2 are 1
    Y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    return x, y


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.linear2(output)
        y_pred = torch.sigmoid(output)
        return y_pred


def ids(v):
    return '{} {}'.format(v[0].__name__, v[1:])


def build_lookahead(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)


optimizers = [
    (build_lookahead, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.A2GradExp, {'lips': 2.0, 'beta': 1e-3}, 500),
    (optim.A2GradInc, {'lips': 5.0, 'beta': 1e-3}, 200),
    (optim.A2GradUni, {'lips': 5.0, 'beta': 1e-3}, 500),
    (optim.AccSGD, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.AdaBelief, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.AdaBound, {'lr': 1.5, 'gamma': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.AdaMod, {'lr': 2.0, 'weight_decay': 1e-3}, 200),
    (optim.Adafactor, {'lr': 0.004466, 'weight_decay': 1e-3}, 1500),
    (optim.AdamP, {'lr': 0.045, 'weight_decay': 1e-3}, 800),
    (optim.AggMo, {'lr': 0.17059, 'weight_decay': 1e-3}, 1000),
    (optim.Apollo, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.DiffGrad, {'lr': 0.5, 'weight_decay': 1e-3}, 200),
    (
        optim.LARS,
        {'lr': 1.0, 'weight_decay': 1e-3, 'trust_coefficient': 0.01},
        200,
    ),
    (optim.Lamb, {'lr': 0.0151, 'weight_decay': 1e-3}, 1000),
    (optim.MADGRAD, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.NovoGrad, {'lr': 0.01, 'weight_decay': 1e-3}, 200),
    (optim.PID, {'lr': 0.01, 'weight_decay': 1e-3, 'momentum': 0.1}, 200),
    (optim.QHAdam, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.QHM, {'lr': 0.1, 'weight_decay': 1e-5, 'momentum': 0.2}, 200),
    (optim.Ranger, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.RangerQH, {'lr': 0.0124, 'weight_decay': 1e-3}, 1100),
    (optim.RangerVA, {'lr': 0.2214, 'weight_decay': 1e-3}, 500),
    (optim.SGDP, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.SGDW, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.SWATS, {'lr': 0.703, 'weight_decay': 1e-3}, 600),
    (
        optim.Shampoo,
        {'lr': 0.279, 'weight_decay': 1e-3, 'momentum': 0.05},
        1600,
    ),
    (optim.Yogi, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.Adahessian, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(optimizer_config):
    torch.manual_seed(42)
    x_data, y_data = make_dataset()
    model = LogisticRegression()

    loss_fn = nn.BCELoss()
    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)
    init_loss = None
    for _ in range(iterations):
        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)
        if init_loss is None:
            init_loss = loss
        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()
    assert init_loss.item() > 2.0 * loss.item()
