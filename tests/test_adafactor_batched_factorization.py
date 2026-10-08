"""Factored Adafactor statistics are independent across leading batch axes."""

import copy
import math

import pytest
import torch

import torch_optimizer as optim


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 2, 4), (2, 3, 2, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_rank_one_matrices_have_closed_form_unit_preconditioned_update(
    shape, dtype, noncontiguous
):
    if noncontiguous:
        storage = torch.zeros(*shape[:-1], 2 * shape[-1], dtype=dtype)
        parameter = torch.nn.Parameter(storage[..., ::2])
        assert not parameter.is_contiguous()
    else:
        parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    gradient = torch.ones(shape, dtype=dtype)
    # Every matrix is g_ij=s*u_i*v_j, with strictly positive factors.
    leading_count = 1
    for size in shape[:-2]:
        leading_count *= size
    for index in range(leading_count):
        matrix = gradient.reshape(-1, shape[-2], shape[-1])[index]
        for row in range(shape[-2]):
            for column in range(shape[-1]):
                matrix[row, column] = (index + 1) * (row + 1) * (column + 1)
    parameter.grad = gradient
    optimizer = optim.Adafactor(
        [parameter],
        lr=0.01,
        eps2=(0.0, 1e-3),
        relative_step=False,
        scale_parameter=False,
    )
    optimizer.step()
    # First-step Vhat_ij=R_i*C_j/mean(R)=g_ij**2 for rank-one g**2.
    # Hence g/sqrt(Vhat)=1 and the default RMS clipping leaves it unchanged.
    assert torch.allclose(
        parameter, torch.full_like(parameter, -0.01), rtol=2e-6, atol=2e-8
    )
    assert optimizer.state[parameter]["exp_avg_sq_row"].shape == shape[:-1]
    assert optimizer.state[parameter]["exp_avg_sq_col"].shape == (
        shape[:-2] + shape[-1:]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("beta1", [None, 0.6])
def test_multistep_factorization_matches_independent_scalar_moments(
    dtype, beta1
):
    shape = (2, 2, 3)
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    optimizer = optim.Adafactor(
        [parameter],
        lr=0.01,
        eps2=(0.0, 1e-3),
        clip_threshold=1e6,
        decay_rate=-0.8,
        beta1=beta1,
        relative_step=False,
        scale_parameter=False,
        weight_decay=0.07,
    )
    row_moments = [[0.0, 0.0] for _ in range(2)]
    column_moments = [[0.0, 0.0, 0.0] for _ in range(2)]
    expected = [[[0.0] * 3 for _ in range(2)] for _ in range(2)]
    first_moments = [[[0.0] * 3 for _ in range(2)] for _ in range(2)]
    for step in range(1, 5):
        values = [
            [
                [
                    0.5
                    + 0.1 * step
                    + 0.05 * batch
                    + 0.03 * row
                    + 0.07 * column
                    for column in range(3)
                ]
                for row in range(2)
            ]
            for batch in range(2)
        ]
        parameter.grad = torch.tensor(values, dtype=dtype)
        beta2 = 1.0 - step**-0.8
        for batch in range(2):
            for row in range(2):
                square_mean = (
                    sum(values[batch][row][column] ** 2 for column in range(3))
                    / 3
                )
                row_moments[batch][row] = (
                    beta2 * row_moments[batch][row] + (1 - beta2) * square_mean
                )
            for column in range(3):
                square_mean = (
                    sum(values[batch][row][column] ** 2 for row in range(2))
                    / 2
                )
                column_moments[batch][column] = (
                    beta2 * column_moments[batch][column]
                    + (1 - beta2) * square_mean
                )
            mass = sum(row_moments[batch]) / 2
            for row in range(2):
                for column in range(3):
                    variance = (
                        row_moments[batch][row]
                        * column_moments[batch][column]
                        / mass
                    )
                    update = (
                        0.01 * values[batch][row][column] / math.sqrt(variance)
                    )
                    if beta1 is not None:
                        first_moments[batch][row][column] = (
                            beta1 * first_moments[batch][row][column]
                            + (1 - beta1) * update
                        )
                        update = first_moments[batch][row][column]
                    expected[batch][row][column] = (
                        expected[batch][row][column] * (1 - 0.07 * 0.01)
                        - update
                    )
        optimizer.step()
        assert torch.allclose(
            parameter,
            torch.tensor(expected, dtype=dtype),
            rtol=2e-6,
            atol=2e-8,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_leading_axis_permutation_does_not_mix_row_statistics(dtype):
    gradient = torch.tensor(
        [[[1, 2, 3], [2, 4, 6]], [[8, 16, 24], [16, 32, 48]]],
        dtype=dtype,
    )
    parameters = [
        torch.nn.Parameter(torch.zeros_like(gradient)),
        torch.nn.Parameter(torch.zeros_like(gradient)),
    ]
    for parameter, supplied in zip(parameters, [gradient, gradient.flip(0)]):
        optimizer = optim.Adafactor(
            [parameter],
            lr=0.01,
            eps2=(0.0, 1e-3),
            relative_step=False,
            scale_parameter=False,
        )
        parameter.grad = supplied.clone()
        optimizer.step()
    assert torch.allclose(
        parameters[0], parameters[1].flip(0), rtol=2e-6, atol=2e-8
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_moments_resume_from_real_optimizer_checkpoint(dtype):
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 4, dtype=dtype))
    options = dict(
        lr=0.01,
        eps2=(0.0, 1e-3),
        relative_step=False,
        scale_parameter=False,
        beta1=0.6,
    )
    optimizer = optim.Adafactor([parameter], **options)
    for step in range(2):
        parameter.grad = (
            torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4) + step
        )
        optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(parameter.detach().clone())
    resumed = optim.Adafactor([restored], **options)
    resumed.load_state_dict(copy.deepcopy(checkpoint))
    for supplied in (
        torch.ones_like(parameter),
        torch.full_like(parameter, 2),
    ):
        parameter.grad = supplied.clone()
        restored.grad = supplied.clone()
        optimizer.step()
        resumed.step()
        assert torch.equal(parameter, restored)
        for key in ("exp_avg", "exp_avg_sq_row", "exp_avg_sq_col"):
            assert torch.equal(
                optimizer.state[parameter][key], resumed.state[restored][key]
            )
    assert checkpoint["state"][0]["step"] == 2
