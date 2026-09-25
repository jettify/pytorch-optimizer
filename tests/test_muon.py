"""Muon optimizer: matches the reference update, 1D fallback works."""
import sys

import torch
import torch_optimizer as optim
from torch_optimizer.muon import muon_update, zeropower_via_newtonschulz5

sys.path.insert(0, "/tmp/Muon-upstream")
import muon as muon_ref


def _explicit_fp32_ns(matrix, steps=5):
    # The same quintic in float32, written out step by step.
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.float()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
    return x


def _explicit_fp32_update(grad, beta=0.95, steps=5):
    # The full reference update rewritten step by step in float32:
    # momentum lerp, nesterov mix, quintic, rectangular scale.
    a, b, c = (3.4445, -4.7750, 2.0315)
    m = torch.zeros_like(grad).float()
    m = m.lerp(grad.float(), 1 - beta)
    u = grad.float().lerp(m, beta)
    x = u / (u.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
    return x * max(1, u.size(-2) / u.size(-1)) ** 0.5


def test_matches_reference_update():
    torch.manual_seed(71)
    g = torch.randn(8, 4)
    m1 = torch.zeros_like(g)
    m2 = torch.zeros_like(g)
    got = muon_update(g.clone(), m1, beta=0.95, ns_steps=5, nesterov=True)
    ref = muon_ref.muon_update(g.clone(), m2, beta=0.95, ns_steps=5,
                               nesterov=True)
    # Tight: the port equals the explicit float32 update.
    assert torch.allclose(
        got.float(), _explicit_fp32_update(g), atol=1e-5
    )
    assert torch.equal(m1, m2)
    # Loose: the bfloat16 reference agrees up to working-precision drift.
    assert torch.allclose(got.float(), ref.float(), atol=0.15)


def test_update_is_near_orthogonal():
    # The quintic does not produce exact UV^T but US'V^T with
    # S'_ii in ~(0.5, 1.5) per the reference docstring.
    torch.manual_seed(72)
    g = torch.randn(16, 16)
    update = muon_update(g, torch.zeros_like(g))
    s = torch.linalg.svdvals(update.float())
    assert bool(((s > 0.3) & (s < 1.7)).all())


def test_1d_fallback_is_momentum_sgd():
    torch.manual_seed(73)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    opt = optim.Muon([p], lr=0.02, momentum=0.9, weight_decay=0.0)
    p.grad = torch.ones(4)
    opt.step()
    # buf = 0 + 0.1 * (1 - 0); update = g.lerp(buf, 0.9) = 1 + 0.9*(0.1-1)
    nesterov_update = torch.full((4,), 1 + 0.9 * (0.1 - 1))
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]) - 0.02 * nesterov_update
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_1d_does_not_raise():
    p = torch.nn.Parameter(torch.ones(4))
    opt = optim.Muon([p], lr=0.02)
    p.grad = torch.ones(4)
    opt.step()


def test_dtype_preserved_on_cpu():
    torch.manual_seed(74)
    g = torch.randn(8, 4)
    out = zeropower_via_newtonschulz5(g, steps=5)
    assert out.dtype == torch.float32
