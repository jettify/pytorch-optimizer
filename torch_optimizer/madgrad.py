import math
from typing import Callable, Optional

import torch
import torch.optim

from .types import Params

__all__ = ('MADGRAD',)


class MADGRAD(torch.optim.Optimizer):
    r"""Implements MADGRAD algorithm.

    It has been proposed in `Adaptivity without Compromise: A Momentumized,
    Adaptive, Dual Averaged Gradient Method for Stochastic Optimization`__

    Arguments:
        params (iterable):
            Iterable of parameters to optimize
            or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation
            to improve numerical stability. (default: 1e-6).

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.MAGRAD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2101.11075

    Note:
        Reference code: https://github.com/facebookresearch/madgrad
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        defaults = dict(
            lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay, k=0
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['grad_sum_sq'] = torch.zeros_like(p.data).detach()
                state['s'] = torch.zeros_like(p.data).detach()
                if momentum != 0:
                    state['x0'] = torch.clone(p.data).detach()

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            k = group['k']
            lr = group['lr'] + eps
            decay = group['weight_decay']
            momentum = group['momentum']

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        'momentum != 0 is not compatible with '
                        'sparse gradients'
                    )

                grad_sum_sq = state['grad_sum_sq']
                s = state['s']

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            'weight_decay option is not '
                            'compatible with sparse gradients'
                        )

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    )
                    x0_masked_vals = p_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1
                    )

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)
                    )

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1
                    )
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state['x0']

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

            group['k'] = group['k'] + 1
        return loss
