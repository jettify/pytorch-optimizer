from io import BytesIO

import pytest
import torch

from torch_optimizer import MADGRAD


def step_quadratic(optimizer):
    optimizer.zero_grad()
    loss = sum(
        parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize(
    "default_momentum, group_momentum",
    [(0, 0.9), (0.9, 0), (0.9, 0.5), (0, 0)],
)
def test_madgrad_initializes_from_group_momentum(
    default_momentum, group_momentum, weight_decay
):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = MADGRAD(
        [{"params": [parameter], "momentum": group_momentum}],
        lr=0.02,
        momentum=default_momentum,
        weight_decay=weight_decay,
    )
    independent = MADGRAD(
        [reference],
        lr=0.02,
        momentum=group_momentum,
        weight_decay=weight_decay,
    )
    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(independent)
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("added_lr", [None, 0.1])
def test_madgrad_updates_added_parameter_groups(momentum, added_lr):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = MADGRAD([first], lr=0.02, momentum=momentum)
    step_quadratic(optimizer)
    added = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    reference = torch.nn.Parameter(added.detach().clone())
    group = {"params": [added]}
    if added_lr is not None:
        group["lr"] = added_lr
    optimizer.add_param_group(group)
    independent = MADGRAD(
        [reference],
        lr=0.02 if added_lr is None else added_lr,
        momentum=momentum,
    )
    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(independent)
        torch.testing.assert_close(added, reference, rtol=0, atol=0)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_madgrad_added_groups_survive_checkpoint_roundtrip(momentum):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = MADGRAD([first], lr=0.02, momentum=momentum)
    step_quadratic(optimizer)
    added = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer.add_param_group({"params": [added], "lr": 0.1})
    for _ in range(3):
        step_quadratic(optimizer)
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = MADGRAD(
        [
            {"params": [torch.nn.Parameter(first.detach().clone())]},
            {"params": [torch.nn.Parameter(added.detach().clone())]},
        ]
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(restored)
        for expected, actual in zip(
            optimizer.param_groups, restored.param_groups
        ):
            torch.testing.assert_close(
                actual["params"][0], expected["params"][0], rtol=0, atol=0
            )


def test_madgrad_added_embedding_accepts_sparse_gradients():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = MADGRAD([first], lr=0.02, momentum=0)
    step_quadratic(optimizer)
    embedding = torch.nn.Embedding(4, 2, sparse=True)
    reference = torch.nn.Embedding(4, 2, sparse=True)
    reference.load_state_dict(embedding.state_dict())
    optimizer.add_param_group({"params": [embedding.weight]})
    independent = MADGRAD(reference.parameters(), lr=0.02, momentum=0)
    for indices in [[0, 2, 2], [1, 2], [0, 3]]:
        optimizer.zero_grad()
        independent.zero_grad()
        embedding(torch.tensor(indices)).square().sum().backward()
        reference(torch.tensor(indices)).square().sum().backward()
        optimizer.step()
        independent.step()
        torch.testing.assert_close(
            embedding.weight, reference.weight, rtol=0, atol=0
        )
