"""debias=True corrects the moments inside the trust ratio."""
import torch
import torch_optimizer as optim


def test_default_debias_is_the_uncorrected_step():
    param = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    opt = optim.Lamb([param], lr=0.1, weight_decay=0.0, betas=(0.9, 0.999))
    grad = torch.tensor([1.0, 0.5])
    param.grad = grad.clone()
    opt.step()
    # m = (1-0.9) g, v = (1-0.999) g^2, step = m / sqrt(v)
    direction = (0.1 * grad) / (0.001 * grad * grad).sqrt()
    weight_norm = torch.tensor([2.0, -1.0]).norm()
    trust = weight_norm / direction.norm()
    expected = torch.tensor([2.0, -1.0]) - 0.1 * trust * direction
    assert torch.allclose(param.detach(), expected, atol=1e-5)


def test_debias_corrects_adam_norm():
    # One step, scalar parameter. Stored moments must stay the raw EMA.
    param = torch.nn.Parameter(torch.tensor([2.0]))
    opt = optim.Lamb(
        [param], lr=0.1, weight_decay=0.1, betas=(0.9, 0.999), debias=True
    )
    param.grad = torch.tensor([1.0])
    opt.step()
    state = opt.state[param]
    assert torch.allclose(state["exp_avg"], torch.tensor([0.1]))
    assert torch.allclose(state["exp_avg_sq"], torch.tensor([0.001]))
    m_hat = 0.1 / (1 - 0.9)
    v_hat = 0.001 / (1 - 0.999)
    direction = m_hat / (v_hat ** 0.5) + 0.1 * 2.0
    # scalar norms are absolute values; trust ratio is |p| / |direction|
    updated = 2.0 - 0.1 * (2.0 / abs(direction)) * direction
    assert torch.allclose(param.detach(), torch.tensor([updated]), atol=1e-5)
    # Scaling the learning rate and leaving adam_norm uncorrected gives
    # about 1.937 on this step, not the corrected 1.8.
    assert abs(param.detach().item() - 1.936754) > 0.05
