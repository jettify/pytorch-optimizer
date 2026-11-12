import pytest
import torch

from torch_optimizer import SGDP, AdamP


@pytest.mark.parametrize("optimizer_class", [AdamP, SGDP])
@pytest.mark.parametrize("layout", ["transpose", "channels_last"])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("delta", [0.0, 2.0])
def test_projection_optimizers_accept_strided_parameters(
    optimizer_class, layout, nesterov, delta
):
    if layout == "transpose":
        values = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4).t()
    else:
        values = torch.arange(1, 25, dtype=torch.float64).reshape(2, 3, 2, 2)
        values = values.to(memory_format=torch.channels_last)
    assert not values.is_contiguous()
    parameter = torch.nn.Parameter(values)
    reference = torch.nn.Parameter(values.contiguous().clone())
    options = dict(lr=0.02, weight_decay=0.1, nesterov=nesterov, delta=delta)
    if optimizer_class is SGDP:
        options["momentum"] = 0.9
    optimizer = optimizer_class([parameter], **options)
    reference_optimizer = optimizer_class([reference], **options)
    original_stride = parameter.stride()
    for step in range(4):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        target = torch.arange(parameter.numel(), dtype=parameter.dtype)
        target = target.reshape(parameter.shape).remainder(3) - step
        (parameter - target).square().sum().backward()
        (reference - target).square().sum().backward()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(
            parameter, reference, rtol=1e-12, atol=1e-12
        )
        assert parameter.stride() == original_stride
