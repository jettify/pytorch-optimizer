import math

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("AdaNorm",)


class AdaNorm(Optimizer):
    r"""Implements AdaNorm algorithm.

    AdaNorm corrects each gradient by the running average of its norm
    before the Adam update: when the historical norm exceeds the current
    one, the gradient is scaled up by their ratio. It also supports the
    AMSGrad variant, decoupled weight decay, and an RAdam-style rectified
    update.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant (default: False)
        weight_decouple: whether to use decoupled weight decay as in AdamW
            (default: False)
        fixed_decay: with decoupled weight decay, decay by ``decay`` instead
            of ``decay * lr`` (default: False)
        rectify: whether to use the RAdam-style rectified update
            (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AdaNorm(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2210.06364

    Note:
        Reference code: https://github.com/shivram1987/AdaNorm
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        rectify: bool = False,
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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            rectify=rectify,
        )
        super(AdaNorm, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(AdaNorm, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("weight_decouple", False)
            group.setdefault("fixed_decay", False)
            group.setdefault("rectify", False)

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
                state["exp_grad_norm"] = torch.tensor(0.0)
                if group["amsgrad"]:
                    state["max_exp_avg_var"] = torch.zeros_like(
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
                        "AdaNorm does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]
                state = self.state[p]
                beta1, beta2 = group["betas"]

                if len(state) == 0:
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_grad_norm"] = torch.tensor(0.0)
                    if amsgrad:
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format,
                        )
                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]
                exp_grad_norm = state["exp_grad_norm"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decouple"]:
                    if group["fixed_decay"]:
                        p.data.mul_(1.0 - group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                elif group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # Gradient norm correction
                grad_norm = torch.linalg.norm(grad)
                exp_grad_norm = 0.95 * exp_grad_norm + 0.05 * grad_norm
                if exp_grad_norm > grad_norm and grad_norm > 0:
                    grad_corr = grad * (exp_grad_norm / grad_norm)
                else:
                    grad_corr = grad
                state["exp_grad_norm"] = exp_grad_norm.clone().detach()

                exp_avg.mul_(beta1).add_(grad_corr, alpha=1 - beta1)
                exp_avg_var.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    torch.max(
                        max_exp_avg_var, exp_avg_var, out=max_exp_avg_var
                    )
                    denom = (
                        max_exp_avg_var + group["eps"]
                    ).sqrt() / math.sqrt(bias_correction2) + group["eps"]
                else:
                    denom = (
                        exp_avg_var + group["eps"]
                    ).sqrt() / math.sqrt(bias_correction2) + group["eps"]

                if not group["rectify"]:
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    state["rho_t"] = state["rho_inf"] - (
                        2
                        * state["step"]
                        * beta2 ** state["step"]
                        / (1.0 - beta2 ** state["step"])
                    )
                    if state["rho_t"] > 4:
                        rho_inf, rho_t = state["rho_inf"], state["rho_t"]
                        rect = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        step_size = (
                            math.sqrt(rect) * group["lr"] / bias_correction1
                        )
                        p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        p.data.add_(exp_avg, alpha=-group["lr"])

        return loss
