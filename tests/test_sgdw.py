"""SGDW decay is a scale of the parameter, as in arXiv 1711.05101."""
import torch
import torch_optimizer as optim


def _step(theta, grad, lr, weight_decay, momentum=0.0):
    param = torch.nn.Parameter(torch.as_tensor(theta, dtype=torch.float64).clone())
    opt = optim.SGDW(
        [param], lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    param.grad = torch.as_tensor(grad, dtype=torch.float64).clone()
    opt.step()
    return param.detach()


def test_zero_gradient_scales_the_parameter():
    # θ = 4, μ = 0.1, λ = 0.2. The old update subtracted λμ and landed
    # on 3.98. The paper scale (1 − λμ) θ is 3.92.
    got = _step([4.0, -2.0], [0.0, 0.0], lr=0.1, weight_decay=0.2)
    assert torch.allclose(got, torch.tensor([3.92, -1.96], dtype=torch.float64))


def test_decay_precedes_the_gradient_step():
    # Scale first, then subtract μ g: 4*(1-0.02) - 0.1*1.5 = 3.77.
    got = _step([4.0], [1.5], lr=0.1, weight_decay=0.2)
    assert torch.allclose(got, torch.tensor([3.77], dtype=torch.float64))


def test_no_decay_matches_plain_sgd():
    theta = torch.tensor([1.5, -0.25, 3.0], dtype=torch.float64)
    grad = torch.tensor([0.4, -1.0, 0.0], dtype=torch.float64)
    got = _step(theta, grad, lr=0.1, weight_decay=0.0, momentum=0.9)
    # one momentum step from an empty buffer stores g and subtracts μ g
    assert torch.allclose(got, theta - 0.1 * grad)
