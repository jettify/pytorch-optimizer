from copy import deepcopy

import pytest
import torch

from torch_optimizer import RAdam


@pytest.mark.parametrize("as_generator", [False, True])
@pytest.mark.parametrize("different_betas", [False, True])
@pytest.mark.parametrize("rates", [(0.01, 0.1), (0.1, 0.01), (0.02, 0.02)])
def test_group_cache_matches_independent_optimizers(
    as_generator, different_betas, rates
):
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0])) for _ in rates]
    references = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in parameters
    ]
    betas = [(0.9, 0.999), (0.8, 0.99) if different_betas else (0.9, 0.999)]
    groups = [
        {"params": [parameter], "lr": lr, "betas": beta}
        for parameter, lr, beta in zip(parameters, rates, betas)
    ]
    optimizer = RAdam(iter(groups) if as_generator else groups)
    independent = [
        RAdam([parameter], lr=lr, betas=beta)
        for parameter, lr, beta in zip(references, rates, betas)
    ]

    for step in range(16):
        gradient = torch.tensor([0.3 + 0.01 * step, -0.7])
        for parameter, reference in zip(parameters, references):
            parameter.grad = gradient.clone()
            reference.grad = gradient.clone()
        optimizer.step()
        for reference_optimizer in independent:
            reference_optimizer.step()
        for actual, expected in zip(parameters, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_delayed_gradient_uses_current_learning_rate():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    delayed = torch.nn.Parameter(first.detach().clone())
    reference = torch.nn.Parameter(first.detach().clone())
    optimizer = RAdam([first, delayed], lr=0.1)
    independent = RAdam([reference], lr=0.01)
    first.grad = torch.tensor([0.3, -0.7])
    optimizer.step()
    optimizer.param_groups[0]["lr"] = 0.01
    first.grad = None
    delayed.grad = torch.tensor([0.3, -0.7])
    reference.grad = delayed.grad.clone()

    optimizer.step()
    independent.step()

    torch.testing.assert_close(delayed, reference, rtol=0, atol=0)


def test_legacy_integer_cache_keys_are_recomputed_on_resume():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = RAdam([parameter], lr=0.1)
    parameter.grad = torch.tensor([0.3, -0.7])
    optimizer.step()
    checkpoint = deepcopy(optimizer.state_dict())
    # Old releases cached the learning-rate-dependent step size by step only.
    checkpoint["param_groups"][0]["buffer"][1] = [1, 1.0, 1.0]
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = RAdam([restored_parameter])
    restored.load_state_dict(checkpoint)
    for _ in range(12):
        parameter.grad = torch.tensor([0.3, -0.7])
        restored_parameter.grad = parameter.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(
            restored_parameter, parameter, rtol=0, atol=0
        )
