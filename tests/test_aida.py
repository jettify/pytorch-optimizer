"""Aida optimizer: matches the reference mathematics, no eps drift."""
import math

import torch
import torch_optimizer as optim


def _reference_step(p, g, state, lr, beta1, beta2, eps, wd, k, xi):
    m, v, step = state
    step += 1
    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step
    g = g + wd * p
    m = beta1 * m + (1 - beta1) * g
    pg, pm = g.clone(), m.clone()
    for _ in range(k):
        inner = (pg * pm).sum()
        pg = pg * (inner / ((pg**2).sum() + xi))
        pm = pm * (inner / ((pm**2).sum() + xi))
    r = pm - pg
    v = beta2 * v + (1 - beta2) * r**2
    denom = (v + eps).sqrt() / math.sqrt(bc2)
    p = p - (lr / bc1) * m / denom
    return p, (m, v, step)


def test_matches_reference_two_steps():
    torch.manual_seed(9)
    for k in (1, 2, 3):
        p = torch.nn.Parameter(torch.randn(6))
        opt = optim.Aida([p], lr=1e-3, weight_decay=1e-2, k=k)
        state = (
            torch.zeros(6),
            torch.zeros(6),
            0,
        )
        ref_p = p.detach().clone()
        for _ in range(2):
            g = torch.randn(6)
            p.grad = g.clone()
            opt.step()
            ref_p, state = _reference_step(
                ref_p, g, state, 1e-3, 0.9, 0.999, 1e-8, 1e-2, k, 1e-20
            )
            assert torch.allclose(p.detach(), ref_p, atol=1e-6)


def test_eps_does_not_accumulate_in_state():
    # After one step from zero state, exp_avg_var must be exactly
    # (1-b2) * residual^2 with no eps mixed in. The reference added eps
    # into the state in place, so eps would already pollute it here.
    torch.manual_seed(11)
    g = torch.randn(4)
    p = torch.nn.Parameter(torch.zeros(4))
    opt = optim.Aida([p], lr=1e-3, eps=0.5)
    p.grad = g.clone()
    opt.step()
    m = (1 - 0.9) * g
    pg, pm = g.clone(), m.clone()
    for _ in range(2):
        inner = (pg * pm).sum()
        pg = pg * (inner / ((pg**2).sum() + 1e-20))
        pm = pm * (inner / ((pm**2).sum() + 1e-20))
    expected = (1 - 0.999) * (pm - pg) ** 2
    assert torch.allclose(opt.state[p]["exp_avg_var"], expected, atol=1e-9)


def test_reset_clears_state():
    p = torch.nn.Parameter(torch.randn(4))
    opt = optim.Aida([p])
    p.grad = torch.randn(4)
    opt.step()
    assert opt.state[p]["step"] == 1
    opt.reset()
    assert opt.state[p]["step"] == 0
    assert torch.equal(
        opt.state[p]["exp_avg"], torch.zeros_like(p)
    )


def test_invalid_k_xi_rejected():
    p = torch.nn.Parameter(torch.ones(2))
    import pytest

    with pytest.raises(ValueError):
        optim.Aida([p], k=0)
    with pytest.raises(ValueError):
        optim.Aida([p], xi=0.0)
