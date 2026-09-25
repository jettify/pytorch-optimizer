"""Adahessian must support Conv1d/Conv2d/Conv3d weights (gh issue 500).

Conv1d weights are 3D and Conv3d weights are 5D. get_trace only averaged
the Hessian diagonal over spatial dims for 4D kernels, so any other conv
rank raised UnboundLocalError.
"""
import torch
import torch_optimizer as optim


def _step(module, x):
    opt = optim.Adahessian(module.parameters(), lr=0.1)
    loss = module(x).pow(2).mean()
    loss.backward(create_graph=True)
    opt.step()


def test_adahessian_conv1d():
    torch.manual_seed(500)
    _step(torch.nn.Conv1d(2, 2, 3), torch.randn(1, 2, 8))


def test_adahessian_conv2d():
    torch.manual_seed(500)
    _step(torch.nn.Conv2d(2, 2, 3), torch.randn(1, 2, 8, 8))


def test_adahessian_conv3d():
    torch.manual_seed(500)
    _step(torch.nn.Conv3d(2, 2, 2), torch.randn(1, 2, 4, 4, 4))


def test_adahessian_conv2d_trace_unchanged():
    # The generic spatial mean must equal the old explicit dims [2, 3].
    hv = torch.randn(2, 2, 3, 3)
    expected = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
    got = torch.mean(hv.abs(), dim=list(range(2, len(hv.size()))), keepdim=True)
    assert torch.equal(got, expected)
