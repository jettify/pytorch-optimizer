import pytest
import torch

from torch_optimizer import Lookahead


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("momentum", [0.0, 0.7])
def test_lookahead_synchronizes_after_each_k_fast_steps(k, alpha, momentum):
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    inner = torch.optim.SGD([parameter], lr=0.1, momentum=momentum)
    optimizer = Lookahead(inner, k=k, alpha=alpha)
    # Model weights may be loaded after constructing the optimizer.
    initial = torch.tensor([2.0, -1.0], dtype=torch.float64)
    with torch.no_grad():
        parameter.copy_(initial)
    gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)

    def cumulative_velocity(steps):
        return (
            steps / (1 - momentum)
            - momentum * (1 - momentum**steps) / (1 - momentum) ** 2
        )

    for step in range(1, 13):
        optimizer.zero_grad()
        (parameter * gradient).sum().backward()
        optimizer.step()
        synchronized = step - step % k
        displacement = (
            alpha * cumulative_velocity(synchronized)
            + cumulative_velocity(step)
            - cumulative_velocity(synchronized)
        )
        expected = initial - 0.1 * displacement * gradient
        torch.testing.assert_close(parameter, expected, rtol=1e-12, atol=1e-12)


def test_lookahead_closure_returns_loss_and_uses_initial_slow_weights():
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = Lookahead(torch.optim.SGD([parameter], lr=0.1), k=2, alpha=0.5)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    first_loss = optimizer.step(closure)
    second_loss = optimizer.step(closure)
    torch.testing.assert_close(
        first_loss, torch.tensor(4.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        second_loss, torch.tensor(2.56, dtype=torch.float64)
    )
    torch.testing.assert_close(
        parameter, torch.tensor([1.64], dtype=torch.float64)
    )
