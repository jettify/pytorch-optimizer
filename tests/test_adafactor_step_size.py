"""Adafactor relative-step constants are configurable (gh issue 535)."""
import math

import pytest
import torch
import torch_optimizer as optim


def _lr(opt, step, rms=1.0):
    group = opt.param_groups[0]
    state = {"step": step, "RMS": rms}
    return opt._get_lr(group, state)


def test_defaults_reproduce_old_schedule():
    p = torch.nn.Parameter(torch.ones(4))
    opt = optim.Adafactor([p], warmup_init=True)
    assert _lr(opt, 100) == min(1e-6 * 100, 1.0 / math.sqrt(100))
    opt2 = optim.Adafactor([p], warmup_init=False)
    assert _lr(opt2, 100) == min(1e-2, 1.0 / math.sqrt(100))


def test_custom_warmup_rate_scales_early_steps():
    p = torch.nn.Parameter(torch.ones(4))
    opt = optim.Adafactor([p], warmup_init=True, warmup_rate=2e-6)
    assert _lr(opt, 100) == min(2e-6 * 100, 1.0 / math.sqrt(100))
    assert _lr(opt, 100) == 2 * min(1e-6 * 100, 1.0 / math.sqrt(100))


def test_custom_min_step_size():
    p = torch.nn.Parameter(torch.ones(4))
    opt = optim.Adafactor([p], warmup_init=False, min_step_size=5e-3)
    assert _lr(opt, 100) == min(5e-3, 1.0 / math.sqrt(100))


def test_old_checkpoint_without_new_keys():
    p = torch.nn.Parameter(torch.ones(4))
    opt = optim.Adafactor([p])
    del opt.param_groups[0]["warmup_rate"]
    del opt.param_groups[0]["min_step_size"]
    assert _lr(opt, 100) == min(1e-2, 1.0 / math.sqrt(100))


def test_negative_values_rejected():
    p = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError):
        optim.Adafactor([p], warmup_rate=-1e-6)
    with pytest.raises(ValueError):
        optim.Adafactor([p], min_step_size=-1e-2)


def test_end_to_end_step():
    torch.manual_seed(535)
    model = torch.nn.Linear(4, 2)
    opt = optim.Adafactor(
        model.parameters(), warmup_init=True, warmup_rate=2e-6
    )
    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    opt.step()
