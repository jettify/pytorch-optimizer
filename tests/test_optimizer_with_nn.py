import numpy as np
import pytest
import torch
import torch_optimizer as optim

from torch import nn


def make_dataset(seed=42):
    rng = np.random.RandomState(seed)
    N = 100
    D = 2

    X = rng.randn(N, D) * 2

    # center the first N/2 points at (-2,-2)
    mid = N // 2
    X[: mid, :] = X[: mid, :] - 2 * np.ones((mid, D))

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
    return f'{v[0].__name__} {v[1:]}'


optimizers = [
    (optim.NovoGrad, {'lr': 0.01, 'weight_decay': 1e-3}, 200),
    (optim.Lamb, {'lr': 0.01, 'weight_decay': 1e-3}, 200),
    (optim.SGDW, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.DiffGrad, {'lr': 0.5, 'weight_decay': 1e-3}, 200),
    (optim.AdaMod, {'lr': 2.0, 'weight_decay': 1e-3}, 200),
    (optim.AdaBound, {'lr': 1.1, 'weight_decay': 1e-3}, 200),
    (optim.Yogi, {'lr': 0.1, 'weight_decay': 1e-3}, 200),
    (optim.RAdam, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
    (optim.AccSGD, {'lr': 1.0, 'weight_decay': 1e-3}, 200),
]


@pytest.mark.parametrize('optimizer_config', optimizers, ids=ids)
def test_basic_nn_modeloptimizer_config(optimizer_config):
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
        loss.backward()
        optimizer.step()
    assert init_loss.item() > 2.0 * loss.item()
