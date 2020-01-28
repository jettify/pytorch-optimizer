import torch
import pytest
import torch_optimizer as optim


def assert_sparse_not_supported(optimizer_class, err_msg=None):
    param = torch.randn(1, 1).to_sparse().requires_grad_(True)
    grad = torch.randn(1, 1).to_sparse()
    param.grad = grad
    optimizer = optimizer_class([param])
    optimizer.zero_grad()
    with pytest.raises(RuntimeError) as ctx:
        optimizer.step()

    msg = err_msg or 'does not support sparse gradients'
    assert msg in str(ctx.value)


def assert_lr_validation(optimizer_class):
    lr = -0.01
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=-0.01)
    msg = f'Invalid learning rate: {lr}'
    assert msg in str(ctx.value)


optimizers = [
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.RAdam,
    optim.SGDW,
    optim.Yogi,
]


@pytest.mark.parametrize('optimizer_class', optimizers)
def test_sparse_not_supported(optimizer_class):
    assert_sparse_not_supported(optimizer_class)


@pytest.mark.parametrize('optimizer_class', optimizers)
def test_learning_rate(optimizer_class):
    assert_lr_validation(optimizer_class)
