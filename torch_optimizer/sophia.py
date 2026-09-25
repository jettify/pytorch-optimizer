import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("Sophia",)


class Sophia(Optimizer):
    r"""Implements SophiaG algorithm.

    Sophia - A Scalable Stochastic Second-order Optimizer for Language Model
    Pre-training. It uses a diagonal Hessian estimate to scale the update of
    a momentum-normalized gradient, which allows larger step sizes than
    first-order methods on language models.

    The Hessian estimate must be refreshed with :meth:`update_hessian`
    (usually on a sampled batch every few steps) before calling
    :meth:`step`. Without any Hessian update the estimate stays zero and
    the update degrades to a signed momentum step.

    This port follows the non-capturable path of the reference
    implementation. CUDA-graph capturable execution is not supported.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-4)
        betas: coefficients used for computing running averages of gradient
            and Hessian diagonal (default: (0.965, 0.99))
        rho: Hutchinson trace clipping factor (default: 0.04)
        weight_decay: weight decay (L2 penalty) (default: 1e-1)
        maximize: maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Sophia(model.parameters(), lr=1e-4)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.update_hessian()
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2305.14342

    Note:
        Reference code: https://github.com/Liuhong99/Sophia
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: Betas2 = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(Sophia, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(Sophia, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Refresh the diagonal Hessian estimate from current gradients."""
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                state["hessian"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )

    @torch.no_grad()
    def step(
        self, closure: OptLossClosure = None, bs: int = 5120
    ) -> OptFloat:
        """Perform a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns
                the loss.
            bs: batch size used to scale the Hessian estimate
                (default: 5120).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, _ = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Sophia does not support sparse gradients"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                if torch.is_complex(p):
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    hess = torch.view_as_real(hess)
                    p_view = torch.view_as_real(p)
                else:
                    p_view = p

                if group["maximize"]:
                    grad = -grad

                state["step"] += 1

                # Perform stepweight decay
                p_view.mul_(1 - group["lr"] * group["weight_decay"])

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                ratio = (
                    exp_avg.abs() / (group["rho"] * bs * hess + 1e-15)
                ).clamp(None, 1)
                p_view.addcmul_(exp_avg.sign(), ratio, value=-group["lr"])

        return loss
