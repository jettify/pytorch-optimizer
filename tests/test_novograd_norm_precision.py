from io import BytesIO

import pytest
import torch

from torch_optimizer import NovoGrad


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("grad_averaging", [False, True])
@pytest.mark.parametrize(
    "dtype,scale",
    [
        (torch.float64, 1e20),
        (torch.float64, 1e-30),
        (torch.float16, 2000.0),
        (torch.float16, 1e-4),
        (torch.bfloat16, 2000.0),
        (torch.float32, 2000.0),
    ],
)
def test_novograd_normalization_preserves_norm_precision(
    dtype, scale, amsgrad, grad_averaging
):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype))
    gradient = torch.tensor([scale, -2 * scale], dtype=dtype)
    norm = gradient.double().square().sum()
    update = gradient.double() / norm.sqrt()
    if grad_averaging:
        update *= 0.1
    expected = (parameter.detach().double() - 0.1 * update).to(dtype)
    optimizer = NovoGrad(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=0,
        amsgrad=amsgrad,
        grad_averaging=grad_averaging,
    )
    parameter.grad = gradient.clone()

    optimizer.step()

    assert torch.isfinite(parameter).all()
    torch.testing.assert_close(parameter, expected)
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg_sq"].double(),
        norm,
        rtol=1e-6,
        atol=0,
    )
    if amsgrad:
        torch.testing.assert_close(
            optimizer.state[parameter]["max_exp_avg_sq"].double(),
            norm,
            rtol=1e-6,
            atol=0,
        )


@pytest.mark.parametrize("amsgrad", [False, True])
def test_novograd_float64_checkpoint_continues_exactly(amsgrad):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    optimizer = NovoGrad(
        [parameter], lr=0.02, betas=(0.9, 0.99), amsgrad=amsgrad
    )
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = NovoGrad([restored_parameter])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for _ in range(4):
        optimizer.zero_grad()
        restored.zero_grad()
        parameter.square().sum().backward()
        restored_parameter.square().sum().backward()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(
            restored_parameter, parameter, rtol=0, atol=0
        )
