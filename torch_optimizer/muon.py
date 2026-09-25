import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params

__all__ = ("Muon",)


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor, steps: int
) -> torch.Tensor:
    r"""Newton-Schulz iteration for the orthogonal factor of a matrix.

    Uses the quintic iteration with coefficients (3.4445, -4.7750, 2.0315).
    The working dtype is float32 on CPU (bfloat16 is emulated there and
    ~200x slower at n=1024) and bfloat16 on CUDA/MPS. The result is cast
    back to the input dtype.
    """
    assert matrix.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    if matrix.device.type == "cpu":
        work = matrix.float()
    else:
        work = matrix.bfloat16()
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        work = work.mT
    # Ensure spectral norm is at most 1
    work = work / (work.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        mat_a = work @ work.mT
        mat_b = b * mat_a + c * mat_a @ mat_a
        work = a * work + mat_b @ work
    if transposed:
        work = work.mT
    return work.to(dtype=matrix.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    r"""Momentum plus Newton-Schulz orthogonalization (2D and 4D only)."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(Optimizer):
    r"""Implements Muon - MomentUm Orthogonalized by Newton-schulz.

    Each 2D parameter update is replaced with its nearest orthogonal
    matrix via a Newton-Schulz iteration, which is stable in bfloat16 on
    the GPU. Parameters with fewer than 2 dimensions (biases, gains)
    take a plain SGD-momentum step instead, since orthogonalization is
    undefined for them; the reference implementation raises an
    AssertionError there.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate, in units of spectral norm per update
            (default: 0.02)
        momentum: momentum factor (default: 0.95)
        weight_decay: AdamW-style weight decay (default: 0)
        nesterov: whether to use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations per step (default: 5)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Muon(model.parameters(), lr=0.02)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://kellerjordan.github.io/posts/muon/

    Note:
        Reference code: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params: Params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(
                "Invalid ns_steps value: {}".format(ns_steps)
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Perform a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns
                the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if p.grad.ndim >= 2:
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    ).reshape(p.shape)
                else:
                    # No orthogonal factor exists below 2D: plain
                    # SGD-momentum with the same buffer semantics.
                    buf = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - group["momentum"])
                    if group["nesterov"]:
                        update = p.grad.lerp_(buf, group["momentum"])
                    else:
                        update = buf
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

        return loss
