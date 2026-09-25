"""Complex Adahessian estimates curvature on both components."""
import torch
import torch_optimizer as optim


def test_real_path_is_unchanged():
    torch.manual_seed(1)
    param = torch.nn.Parameter(torch.tensor([0.5, -0.2, 0.3]))
    opt = optim.Adahessian([param], lr=0.05)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = (param ** 2).sum()
        loss.backward(create_graph=True)
        opt.step()
    expected = torch.tensor(
        [0.43060994148254395, -0.17224396765232086, 0.2583659589290619]
    )
    assert torch.allclose(param.detach(), expected)


def test_complex_matches_the_real_algorithm():
    start = torch.tensor([0.5 + 0.1j, -0.2 + 0.3j])
    torch.manual_seed(2)
    param = torch.nn.Parameter(start.clone())
    opt = optim.Adahessian([param], lr=0.05)
    opt.zero_grad(set_to_none=True)
    loss = (param.real ** 2 + param.imag ** 2).sum()
    loss.backward(create_graph=True)
    opt.step()

    torch.manual_seed(2)
    real = torch.nn.Parameter(torch.view_as_real(start).clone())
    real_opt = optim.Adahessian([real], lr=0.05)
    real_opt.zero_grad(set_to_none=True)
    real_loss = (real ** 2).sum()
    real_loss.backward(create_graph=True)
    real_opt.step()
    assert torch.equal(torch.view_as_real(param.detach()), real.detach())


def test_complex_matrix_steps():
    torch.manual_seed(3)
    param = torch.nn.Parameter(
        torch.tensor([[0.2 + 0.1j, -0.3 + 0.0j], [0.4 - 0.2j, 0.1 + 0.5j]])
    )
    before = param.detach().clone()
    opt = optim.Adahessian([param], lr=0.05)
    opt.zero_grad(set_to_none=True)
    loss = (param.real ** 2 + param.imag ** 2).sum()
    loss.backward(create_graph=True)
    opt.step()
    assert param.dtype == torch.complex64
    assert not torch.equal(param.detach(), before)
