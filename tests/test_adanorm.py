"""AdaNorm optimizer: matches the reference mathematics, guards zero norm."""
import math

import pytest
import torch
import torch_optimizer as optim


def test_matches_reference_one_step():
    torch.manual_seed(21)
    g = torch.randn(5)
    p = torch.nn.Parameter(torch.randn(5))
    opt = optim.AdaNorm([p], lr=1e-3, weight_decay=0.0)
    before = p.detach().clone()
    p.grad = g.clone()
    opt.step()
    b1, b2 = 0.9, 0.999
    gn = g.norm().item()
    egn = 0.05 * gn
    gc = g * (egn / gn) if egn > gn else g
    m = (1 - b1) * gc
    v = (1 - b2) * g * g
    denom = (v + 1e-8).sqrt() / math.sqrt(1 - b2) + 1e-8
    assert torch.allclose(
        p.detach(),
        before - (1e-3 / (1 - b1)) * m / denom,
        atol=1e-6,
    )


def test_zero_gradient_is_finite():
    # exp_grad_norm > 0 from history, current grad exactly zero: the guard
    # must avoid 0/0, and the parameter must stay finite.
    p = torch.nn.Parameter(torch.randn(4))
    opt = optim.AdaNorm([p], lr=1e-3, weight_decay=0.0)
    p.grad = torch.ones(4)
    opt.step()
    p.grad = torch.zeros(4)
    opt.step()
    assert torch.isfinite(p).all()
    assert torch.isfinite(opt.state[p]["exp_avg"]).all()


def test_rectify_paths():
    p = torch.nn.Parameter(torch.randn(3))
    opt = optim.AdaNorm([p], lr=1e-3, rectify=True)
    p.grad = torch.randn(3)
    opt.step()
    # step 1 with beta2=0.999: rho_t = 1999 - 1998 = 1 <= 4 -> SGD-style
    assert opt.state[p]["rho_t"] <= 4
    for _ in range(60):
        p.grad = torch.randn(3)
        opt.step()
    assert opt.state[p]["rho_t"] > 4


def test_decoupled_decay_variants_move():
    for kw in ({"weight_decouple": True}, {"weight_decouple": True, "fixed_decay": True}):
        p = torch.nn.Parameter(torch.ones(3))
        opt = optim.AdaNorm([p], lr=1e-2, weight_decay=0.5, **kw)
        p.grad = torch.zeros(3)
        before = p.detach().clone()
        opt.step()
        assert not torch.equal(p, before)


def test_reset_clears_state():
    p = torch.nn.Parameter(torch.randn(3))
    opt = optim.AdaNorm([p])
    p.grad = torch.randn(3)
    opt.step()
    assert opt.state[p]["step"] == 1
    opt.reset()
    assert opt.state[p]["step"] == 0
    assert torch.equal(opt.state[p]["exp_avg"], torch.zeros_like(p))
