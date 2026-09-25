import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("Aida",)


class Aida(Optimizer):
    r"""Implements Aida algorithm.

    Aida exploits layerwise gradient statistics through mutual vector
    projections of the gradient and its running average, improving upon
    AdaBelief on vision and language tasks.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
            and its residual square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        k: number of mutual projection rounds per iteration (default: 2)
        xi: term used in the projections to avoid division by zero
            (default: 1e-20)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Aida(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2203.13273

    Note:
        Reference code: https://github.com/guoqiang-zhang-x/Aida-Optimizer
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        k: int = 2,
        xi: float = 1e-20,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not isinstance(k, int) or k < 1:
            raise ValueError(
                "Invalid projection rounds value: {}".format(k)
            )
        if not 0.0 < xi:
            raise ValueError("Invalid xi value: {}".format(xi))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            k=k,
            xi=xi,
        )
        super(Aida, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(Aida, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("k", 2)
            group.setdefault("xi", 1e-20)

    def reset(self) -> None:
        """Reset the optimizer state of all parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                )
                state["exp_avg_var"] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format
                )

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Perform a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns
                the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Aida does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                state = self.state[p]
                beta1, beta2 = group["betas"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                proj_g = grad.detach().clone()
                proj_m = exp_avg.detach().clone()
                for _ in range(group["k"]):
                    inner = torch.sum(torch.mul(proj_g, proj_m))
                    scalar_g = inner / (
                        torch.sum(torch.pow(proj_g, 2)) + group["xi"]
                    )
                    scalar_m = inner / (
                        torch.sum(torch.pow(proj_m, 2)) + group["xi"]
                    )
                    proj_g.mul_(scalar_g)
                    proj_m.mul_(scalar_m)

                grad_residual = proj_m - proj_g
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                # Out-of-place eps: the reference added eps into the state
                # in place, so eps accumulated into exp_avg_var every step.
                denom = (exp_avg_var + group["eps"]).sqrt() / (
                    bias_correction2**0.5
                )

                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
