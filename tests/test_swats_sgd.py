from io import BytesIO

import pytest
import torch

from torch_optimizer import SWATS


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize(
    "momentum,nesterov",
    [(0.0, False), (0.5, False), (0.9, False), (0.5, True), (0.9, True)],
)
def test_swats_sgd_phase_matches_normalized_momentum_sgd(
    momentum, nesterov, weight_decay
):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], dtype=torch.float64)
    )
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = SWATS(
        [parameter],
        lr=0.1,
        betas=(momentum, 0.999),
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    optimizer.param_groups[0]["phase"] = "SGD"
    expected_optimizer = torch.optim.SGD(
        [reference],
        lr=0.1 * (1 - momentum),
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    for step in range(5):
        gradient = torch.tensor([1.0 + 0.2 * step, -0.5], dtype=torch.float64)
        parameter.grad = gradient.clone()
        reference.grad = gradient.clone()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(
            parameter, reference, rtol=1e-14, atol=1e-14
        )


@pytest.mark.parametrize("nesterov", [False, True])
def test_swats_automatic_switch_and_checkpoint_keep_sgd_trajectory(nesterov):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = SWATS(
        [parameter], lr=0.1, betas=(0.5, 0.9), eps=0, nesterov=nesterov
    )
    for _ in range(2):
        optimizer.zero_grad()
        parameter.sum().backward()
        optimizer.step()
    assert optimizer.param_groups[0]["phase"] == "SGD"
    reference = torch.nn.Parameter(parameter.detach().clone())
    expected_optimizer = torch.optim.SGD(
        [reference],
        lr=optimizer.param_groups[0]["lr"] * 0.5,
        momentum=0.5,
        nesterov=nesterov,
    )
    for _ in range(4):
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        parameter.square().sum().backward()
        reference.square().sum().backward()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(
            parameter, reference, rtol=1e-14, atol=1e-14
        )

    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = SWATS([restored_parameter])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for _ in range(3):
        restored.zero_grad()
        expected_optimizer.zero_grad()
        restored_parameter.square().sum().backward()
        reference.square().sum().backward()
        restored.step()
        expected_optimizer.step()
        torch.testing.assert_close(
            restored_parameter, reference, rtol=1e-14, atol=1e-14
        )
