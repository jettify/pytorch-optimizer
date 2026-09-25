"""The first PID step must remember its gradient."""
import torch
import torch_optimizer as optim


def test_second_step_uses_the_previous_gradient():
    param = torch.nn.Parameter(torch.tensor([0.0]))
    opt = optim.PID(
        [param],
        lr=1.0,
        momentum=0.5,
        dampening=0.0,
        weight_decay=0.0,
        integral=0.0,
        derivative=10.0,
    )
    param.grad = torch.tensor([1.0])
    opt.step()
    assert torch.equal(
        opt.state[param]["grad_buffer"], torch.tensor([1.0])
    )
    # first derivative is 0: there is no earlier gradient
    assert torch.equal(param.detach(), torch.tensor([-1.0]))

    param.grad = torch.tensor([3.0])
    opt.step()
    # D = (1 - momentum) * (3 - 1) = 1, so the step is 3 + 10 * 1
    assert torch.allclose(param.detach(), torch.tensor([-14.0]))
