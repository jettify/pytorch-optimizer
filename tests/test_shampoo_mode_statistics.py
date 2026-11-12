from io import BytesIO

import pytest
import torch

from torch_optimizer import Shampoo


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 3, 2), (2, 2, 2, 2)])
@pytest.mark.parametrize("update_freq", [1, 2])
def test_shampoo_uses_the_same_raw_gradient_for_every_mode(shape, update_freq):
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))
    optimizer = Shampoo(
        [parameter], lr=0.02, epsilon=0.5, update_freq=update_freq
    )
    expected = parameter.detach().clone()
    statistics = [0.5 * torch.eye(size, dtype=torch.float64) for size in shape]
    factors = [None for _ in shape]
    for step in range(4):
        gradient = torch.linspace(
            -0.2, 0.3, parameter.numel(), dtype=torch.float64
        )
        gradient = gradient.reshape(shape) + 0.05 * step
        optimizer.zero_grad()
        (parameter * gradient).sum().backward()

        for axis in range(len(shape)):
            contracted = [
                other for other in range(len(shape)) if other != axis
            ]
            statistics[axis] += torch.tensordot(
                gradient, gradient, dims=(contracted, contracted)
            )
            if step % update_freq == 0:
                eigenvalues, eigenvectors = torch.linalg.eigh(statistics[axis])
                # Keep the implementation's existing exponent; this test
                # isolates which gradient enters each mode's statistic.
                factors[axis] = (
                    eigenvectors * eigenvalues.pow(-1 / len(shape))
                ) @ eigenvectors.t()
        direction = gradient
        for axis, factor in enumerate(factors):
            direction = torch.tensordot(factor, direction, dims=([1], [axis]))
            direction = direction.movedim(0, axis)
        expected.add_(direction, alpha=-0.02)

        optimizer.step()

        for axis, statistic in enumerate(statistics):
            torch.testing.assert_close(
                optimizer.state[parameter]["precond_{}".format(axis)],
                statistic,
                rtol=1e-12,
                atol=1e-12,
            )
        torch.testing.assert_close(parameter, expected, rtol=1e-10, atol=1e-12)


def test_shampoo_mode_statistics_survive_checkpoint_roundtrip():
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    )
    optimizer = Shampoo([parameter], lr=0.02, epsilon=0.5, update_freq=2)
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = Shampoo([restored_parameter])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for _ in range(3):
        optimizer.zero_grad()
        restored.zero_grad()
        parameter.square().sum().backward()
        restored_parameter.square().sum().backward()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(
            restored_parameter, parameter, rtol=0, atol=0
        )
