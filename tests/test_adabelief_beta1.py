"""beta1 = 0 must not collapse the AdaBelief second moment to eps."""
import torch
import torch_optimizer as optim


def test_positive_beta1_uses_the_residual():
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    opt = optim.AdaBelief(
        [param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        rectify=False,
        weight_decouple=False,
    )
    grad = torch.tensor([0.4, -0.2])
    param.grad = grad.clone()
    opt.step()
    exp_avg = opt.state[param]["exp_avg"]
    residual = grad - exp_avg
    # s = (1 - beta2) residual^2, then eps is added into the state
    expected = (1 - 0.999) * residual * residual + 1e-8
    assert torch.allclose(opt.state[param]["exp_avg_var"], expected)


def test_beta1_zero_stays_finite():
    param = torch.nn.Parameter(torch.tensor([0.5]))
    opt = optim.AdaBelief(
        [param],
        lr=1e-2,
        betas=(0.0, 0.999),
        eps=1e-16,
        rectify=False,
        weight_decouple=False,
        weight_decay=0.0,
    )
    for _ in range(30):
        param.grad = (2 * param.detach()).clone()
        opt.step()
    assert torch.isfinite(param.detach()).all()
    assert param.detach().abs().item() < 0.5
