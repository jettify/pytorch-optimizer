from copy import deepcopy
from io import BytesIO

import pytest
import torch

from torch_optimizer import Lamb


def step_quadratic(optimizer):
    optimizer.zero_grad()
    loss = sum(
        (parameter - parameter.new_tensor([0.5, -1.0])).square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()


def serialized_state(optimizer):
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    return torch.load(checkpoint, weights_only=True)


def assert_continuation_matches(optimizer, restored):
    for _ in range(4):
        step_quadratic(optimizer)
        step_quadratic(restored)
        for original_group, restored_group in zip(
            optimizer.param_groups, restored.param_groups
        ):
            for expected, actual in zip(
                original_group["params"], restored_group["params"]
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"clamp_value": 0.1},
        {"adam": True},
        {"debias": True},
        {"clamp_value": 0.1, "adam": True, "debias": True},
    ],
)
def test_lamb_checkpoint_preserves_update_options(num_groups, options):
    parameters = [
        torch.nn.Parameter(
            torch.tensor([5.0 + index, -3.0], dtype=torch.float64)
        )
        for index in range(num_groups)
    ]
    optimizer = Lamb(
        [
            {"params": [parameter], "lr": 0.02 * (index + 1)}
            for index, parameter in enumerate(parameters)
        ],
        weight_decay=0.1,
        **options,
    )
    for _ in range(3):
        step_quadratic(optimizer)

    restored = Lamb(
        [
            {"params": [torch.nn.Parameter(parameter.detach().clone())]}
            for parameter in parameters
        ]
    )
    restored.load_state_dict(serialized_state(optimizer))
    assert_continuation_matches(optimizer, restored)
    assert restored.clamp_value == optimizer.clamp_value
    assert restored.adam == optimizer.adam
    assert restored.debias == optimizer.debias


def test_lamb_checkpoint_overrides_constructor_options():
    parameter = torch.nn.Parameter(
        torch.tensor([5.0, -3.0], dtype=torch.float64)
    )
    optimizer = Lamb([parameter], lr=0.02, weight_decay=0.1)
    for _ in range(3):
        step_quadratic(optimizer)
    restored = Lamb(
        [torch.nn.Parameter(parameter.detach().clone())],
        clamp_value=0.1,
        adam=True,
        debias=True,
    )
    restored.load_state_dict(serialized_state(optimizer))
    assert_continuation_matches(optimizer, restored)


@pytest.mark.parametrize("adam", [False, True])
def test_lamb_legacy_checkpoint_preserves_constructor_options(adam):
    options = {"clamp_value": 0.1, "adam": adam, "debias": True}
    parameter = torch.nn.Parameter(
        torch.tensor([5.0, -3.0], dtype=torch.float64)
    )
    optimizer = Lamb([parameter], lr=0.02, weight_decay=0.1, **options)
    for _ in range(3):
        step_quadratic(optimizer)
    checkpoint = serialized_state(optimizer)
    checkpoint.pop("lamb_options", None)
    restored = Lamb(
        [torch.nn.Parameter(parameter.detach().clone())], **options
    )
    restored.load_state_dict(checkpoint)
    assert_continuation_matches(optimizer, restored)
    assert "lamb_options" not in checkpoint


def test_lamb_checkpoint_uses_current_global_options():
    parameter = torch.nn.Parameter(
        torch.tensor([5.0, -3.0], dtype=torch.float64)
    )
    optimizer = Lamb([parameter], lr=0.02, weight_decay=0.1)
    step_quadratic(optimizer)
    optimizer.clamp_value = 0.1
    optimizer.adam = True
    optimizer.debias = True
    checkpoint = serialized_state(optimizer)
    original_options = deepcopy(checkpoint.get("lamb_options"))
    restored = Lamb([torch.nn.Parameter(parameter.detach().clone())])
    restored.load_state_dict(checkpoint)
    assert_continuation_matches(optimizer, restored)
    assert checkpoint.get("lamb_options") == original_options
