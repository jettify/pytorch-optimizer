from copy import deepcopy
from io import BytesIO

import pytest
import torch

from torch_optimizer import AdaBound


def step_quadratic(optimizer):
    optimizer.zero_grad()
    loss = sum(
        50 * parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("amsbound", [False, True])
@pytest.mark.parametrize("added_lr", [None, 0.2])
def test_adabound_updates_added_parameter_groups(amsbound, added_lr):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    optimizer = AdaBound([first], lr=0.05, amsbound=amsbound)
    step_quadratic(optimizer)

    added = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    reference = torch.nn.Parameter(added.detach().clone())
    group = {"params": [added]}
    if added_lr is not None:
        group["lr"] = added_lr
    optimizer.add_param_group(group)
    reference_optimizer = AdaBound(
        [reference],
        lr=0.05 if added_lr is None else added_lr,
        amsbound=amsbound,
    )

    initial = added.detach().clone()
    for _ in range(3):
        step_quadratic(optimizer)
        step_quadratic(reference_optimizer)
        torch.testing.assert_close(added, reference, rtol=0, atol=0)
    assert not torch.equal(added, initial)


@pytest.mark.parametrize("amsbound", [False, True])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_adabound_checkpoint_preserves_bound_scaling(amsbound, num_groups):
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        for _ in range(num_groups)
    ]
    groups = [
        {"params": [parameter], "lr": 0.05 * (index + 1)}
        for index, parameter in enumerate(parameters)
    ]
    optimizer = AdaBound(
        groups, lr=0.05, final_lr=0.005, gamma=0.4, amsbound=amsbound
    )
    for _ in range(3):
        step_quadratic(optimizer)
    for group in optimizer.param_groups:
        group["lr"] *= 0.2

    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in parameters
    ]
    restored = AdaBound(
        [{"params": [parameter]} for parameter in restored_parameters],
        lr=0.7,
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(restored)
        for expected, actual in zip(parameters, restored_parameters):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("amsbound", [False, True])
def test_adabound_loads_legacy_checkpoint(amsbound):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    optimizer = AdaBound(
        [parameter], lr=0.05, final_lr=0.005, gamma=0.4, amsbound=amsbound
    )
    for _ in range(3):
        step_quadratic(optimizer)
    optimizer.param_groups[0]["lr"] *= 0.2
    checkpoint = deepcopy(optimizer.state_dict())
    for group in checkpoint["param_groups"]:
        group.pop("base_lr", None)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = AdaBound([restored_parameter], lr=0.05)
    restored.load_state_dict(checkpoint)

    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(restored)
        torch.testing.assert_close(
            restored_parameter, parameter, rtol=0, atol=0
        )
    assert "base_lr" not in checkpoint["param_groups"][0]


@pytest.mark.parametrize("amsbound", [False, True])
def test_adabound_deepcopy_continues_after_learning_rate_change(amsbound):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    optimizer = AdaBound(
        [parameter], lr=0.05, final_lr=0.005, gamma=0.4, amsbound=amsbound
    )
    step_quadratic(optimizer)
    optimizer.param_groups[0]["lr"] *= 0.2
    copied = deepcopy(optimizer)

    for _ in range(3):
        step_quadratic(optimizer)
        step_quadratic(copied)
        torch.testing.assert_close(
            copied.param_groups[0]["params"][0], parameter, rtol=0, atol=0
        )
