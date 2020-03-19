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


no_sparse_optimizers = [
    optim.AdaBound,
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.NovoGrad,
    optim.RAdam,
    optim.Yogi,
]


@pytest.mark.parametrize('optimizer_class', no_sparse_optimizers)
def test_sparse_not_supported(optimizer_class):
    assert_sparse_not_supported(optimizer_class)


optimizers = [
    optim.AccSGD,
    optim.AdaBound,
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.NovoGrad,
    optim.PID,
    optim.QHAdam,
    optim.QHM,
    optim.RAdam,
    optim.SGDW,
    optim.Yogi,
]


@pytest.mark.parametrize('optimizer_class', optimizers)
def test_learning_rate(optimizer_class):
    lr = -0.01
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=-0.01)
    msg = 'Invalid learning rate: {}'.format(lr)
    assert msg in str(ctx.value)


eps_optimizers = [
    optim.AdaBound,
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.NovoGrad,
    optim.QHAdam,
    optim.RAdam,
    optim.Yogi,
]


@pytest.mark.parametrize('optimizer_class', eps_optimizers)
def test_eps_validation(optimizer_class):
    eps = -0.1
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=0.1, eps=eps)
    msg = 'Invalid epsilon value: {}'.format(eps)
    assert msg in str(ctx.value)


weight_decay_optimizers = [
    optim.AccSGD,
    optim.AdaBound,
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.PID,
    optim.QHAdam,
    optim.QHM,
    optim.RAdam,
    optim.SGDW,
    optim.Yogi,
]


@pytest.mark.parametrize('optimizer_class', optimizers)
def test_weight_decay_validation(optimizer_class):
    weight_decay = -0.1
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=0.1, weight_decay=weight_decay)
    msg = 'Invalid weight_decay value: {}'.format(weight_decay)
    assert msg in str(ctx.value)


betas_optimizers = [
    optim.AdaBound,
    optim.AdaMod,
    optim.DiffGrad,
    optim.Lamb,
    optim.NovoGrad,
    optim.RAdam,
    optim.Yogi,
    optim.QHAdam,
]


@pytest.mark.parametrize('optimizer_class', eps_optimizers)
def test_betas_validation(optimizer_class):
    betas = (-1, 0.999)
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=0.1, betas=(-1, 0.999))
    msg = 'Invalid beta parameter at index 0: {}'.format(betas[0])
    assert msg in str(ctx.value)

    betas = (0.9, -0.999)
    with pytest.raises(ValueError) as ctx:
        optimizer_class(None, lr=0.1, betas=betas)
    msg = 'Invalid beta parameter at index 1: {}'.format(betas[1])
    assert msg in str(ctx.value)
