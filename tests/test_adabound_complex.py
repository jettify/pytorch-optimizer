"""Complex AdaBound is the real algorithm on both components."""
import torch
import torch_optimizer as optim

BASE = torch.tensor(
    [
        [0.1, -0.2, 0.3],
        [0.0, 1.5, -0.4],
        [2.0, -1.0, 0.5],
        [0.7, 0.7, -0.7],
    ]
)
GRAD = torch.tensor(
    [
        [0.2, 0.0, -0.1],
        [1.0, -0.5, 0.3],
        [0.0, 0.4, -0.2],
        [0.1, 0.1, 0.1],
    ]
)


def test_real_path_is_unchanged():
    param = torch.nn.Parameter(BASE.clone())
    opt = optim.AdaBound([param], lr=1e-3, weight_decay=1e-2)
    param.grad = GRAD.clone()
    opt.step()
    param.grad = (-0.5 * GRAD).clone()
    opt.step()
    expected = torch.tensor(
        [
            [0.09872793406248093, -0.19800041615962982, 0.30123090744018555],
            [-0.0012663287343457341, 1.501231074333191, -0.4012507498264313],
            [1.998000144958496, -1.0012370347976685, 0.5012369155883789],
            [0.6986559629440308, 0.6986559629440308, -0.7011821269989014],
        ]
    )
    assert torch.equal(param.detach(), expected)


def _complex_matches(amsbound):
    base = torch.tensor([1.0 + 0.5j, -0.25 + 0.0j, 0.3 - 0.8j])
    grad = torch.tensor([0.4 - 0.1j, 0.2 + 0.2j, -0.5 + 0.3j])
    param = torch.nn.Parameter(base.clone())
    opt = optim.AdaBound(
        [param], lr=1e-2, betas=(0.9, 0.999), final_lr=0.1, amsbound=amsbound
    )
    param.grad = grad.clone()
    opt.step()
    real_param = torch.nn.Parameter(torch.view_as_real(base).clone())
    real_opt = optim.AdaBound(
        [real_param],
        lr=1e-2,
        betas=(0.9, 0.999),
        final_lr=0.1,
        amsbound=amsbound,
    )
    real_param.grad = torch.view_as_real(grad).clone()
    real_opt.step()
    assert torch.equal(torch.view_as_real(param.detach()), real_param.detach())


def test_complex_matches_real_algorithm_on_both_parts():
    _complex_matches(False)


def test_amsbound_complex_matches_real_algorithm():
    _complex_matches(True)
