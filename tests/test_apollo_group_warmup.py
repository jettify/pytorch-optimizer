from io import BytesIO

import pytest
import torch

from torch_optimizer import Apollo


def step_quadratic(optimizer):
    optimizer.zero_grad()
    sum(
        parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ).backward()
    optimizer.step()


@pytest.mark.parametrize("rates", [(0.01, 0.1), (0.1, 0.01), (0.02, 0.02)])
@pytest.mark.parametrize("warmup", [1, 3])
@pytest.mark.parametrize("as_generator", [False, True])
def test_apollo_group_warmup_matches_independent_optimizers(
    rates, warmup, as_generator
):
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        for _ in rates
    ]
    references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    groups = [{"params": [p], "lr": lr} for p, lr in zip(parameters, rates)]
    optimizer = Apollo(
        iter(groups) if as_generator else groups,
        lr=0.02,
        warmup=warmup,
        init_lr=0.001,
    )
    independent = [
        Apollo([p], lr=lr, warmup=warmup, init_lr=0.001)
        for p, lr in zip(references, rates)
    ]
    for _ in range(6):
        step_quadratic(optimizer)
        for reference_optimizer in independent:
            step_quadratic(reference_optimizer)
        for actual, expected in zip(parameters, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("added_lr", [None, 0.1])
def test_apollo_added_group_uses_its_own_warmup_rate(added_lr):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = Apollo([first], lr=0.02, warmup=3, init_lr=0.001)
    for _ in range(4):
        step_quadratic(optimizer)
    added = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    reference = torch.nn.Parameter(added.detach().clone())
    group = {"params": [added]}
    if added_lr is not None:
        group["lr"] = added_lr
    optimizer.add_param_group(group)
    independent = Apollo(
        [reference],
        lr=0.02 if added_lr is None else added_lr,
        warmup=3,
        init_lr=0.001,
    )
    for _ in range(5):
        step_quadratic(optimizer)
        step_quadratic(independent)
        torch.testing.assert_close(added, reference, rtol=0, atol=0)


def test_apollo_explicit_base_rate_survives_checkpoint():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = Apollo(
        [{"params": [parameter], "lr": 0.2, "base_lr": 0.1}],
        lr=0.01,
        warmup=4,
        init_lr=0.001,
    )
    for _ in range(2):
        step_quadratic(optimizer)
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = Apollo([restored_parameter], lr=0.5)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(restored)
        torch.testing.assert_close(
            restored_parameter, parameter, rtol=0, atol=0
        )
    assert optimizer.param_groups[0]["base_lr"] == 0.1
