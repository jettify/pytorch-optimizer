"""Sophia optimizer: matches the reference update, Hessian refresh works."""
import torch
import torch_optimizer as optim


def test_matches_reference_update():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(3, 4))
    opt = optim.Sophia([p], lr=1e-3, betas=(0.9, 0.99), rho=0.04,
                       weight_decay=0.0)
    g = torch.randn(3, 4)
    h = torch.rand(3, 4) + 0.5
    p.grad = g.clone()
    # seed state as update_hessian would after one refresh
    opt.update_hessian()
    opt.state[p]["hessian"].copy_(h)
    opt.state[p]["exp_avg"].zero_()
    before = p.detach().clone()
    opt.step()
    beta1, bs, rho, lr = 0.9, 5120, 0.04, 1e-3
    m = (1 - beta1) * g
    ratio = (m.abs() / (rho * bs * h + 1e-15)).clamp(None, 1)
    expected = before * (1 - lr * 0.0) - lr * m.sign() * ratio
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_update_hessian_accumulates():
    torch.manual_seed(2)
    p = torch.nn.Parameter(torch.randn(4))
    opt = optim.Sophia([p])
    p.grad = torch.ones(4) * 2.0
    opt.update_hessian()
    h1 = opt.state[p]["hessian"].clone()
    assert torch.allclose(h1, torch.full((4,), (1 - 0.99) * 4.0))
    opt.update_hessian()
    h2 = opt.state[p]["hessian"].clone()
    assert torch.allclose(h2, h1 * 0.99 + (1 - 0.99) * 4.0)


def test_step_without_hessian_update_is_signed_step():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.zeros(4))
    opt = optim.Sophia([p], lr=0.01, weight_decay=0.0)
    p.grad = torch.tensor([1.0, -2.0, 0.5, -0.25])
    opt.step()
    # hessian is zero -> ratio clamps to 1 -> pure sign step
    assert torch.allclose(
        p.detach(), -0.01 * torch.tensor([1.0, -1.0, 1.0, -1.0])
    )


def test_sparse_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = optim.Sophia([p])
    p.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]), torch.tensor([1.0, 2.0]), (4,)
    )
    try:
        opt.step()
    except RuntimeError:
        return
    raise AssertionError("sparse gradient was not rejected")


def test_maximize_flips_sign():
    torch.manual_seed(4)
    kw = dict(lr=0.01, weight_decay=0.0)
    a = torch.nn.Parameter(torch.zeros(4))
    b = torch.nn.Parameter(torch.zeros(4))
    oa = optim.Sophia([a], **kw)
    ob = optim.Sophia([b], maximize=True, **kw)
    g = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a.grad = g.clone()
    b.grad = g.clone()
    oa.step()
    ob.step()
    assert torch.allclose(a.detach(), -b.detach())
