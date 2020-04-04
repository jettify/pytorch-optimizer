import torch
import torch_optimizer as optim
import pytest


@pytest.mark.parametrize('name_optim_tuple', optim.NAME_OPTIM_MAP.items())
def test_returns_optimizer_cls(name_optim_tuple):
    optimizer_cls = optim.get(name_optim_tuple[0])
    assert optimizer_cls == name_optim_tuple[1]
    assert torch.optim.Optimizer in optimizer_cls.__bases__


@pytest.mark.parametrize('should_raise', [
    'not_an_optimizer_str', int(), torch.optim.Optimizer
])
def test_raises(should_raise):
    with pytest.raises(ValueError):
        optim.get(should_raise)
