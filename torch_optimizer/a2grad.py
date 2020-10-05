import copy
import math
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params

__all__ = ('A2GradUni',)


class A2GradUni(Optimizer):
    r"""Implements A2GradUni Optimizer Algorithm.

    It has been proposed in `Optimal Adaptive and Accelerated Stochastic
    Gradient Descent`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: not used for this optimizer (default: None)
        beta:  (default: 10)
        lips: Lipschitz constant (default: 10)


    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.A2GradUni(model.parameters(), lips=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1810.00553

    Note:
        Reference code: https://github.com/severilov/A2Grad_optimizer
    """

    def __init__(
        self,
        params: Params,
        lr: Optional[float] = None,
        beta: float = 10,
        lips: float = 10,
    ):

        defaults = dict(beta=beta, lips=lips, lr=lr)
        # lr is not supported for this optimizer, we need to make tests work
        # and schedulers not to fail
        if beta < 0.0:
            raise ValueError('Invalid beta value: {}'.format(beta))
        if lips < 0.0:
            raise ValueError('Invalid weight_decay value: {}'.format(lips))

        super().__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['alpha_k'] = 1
                    state['v_k'] = 0
                    state['avg_grad'] = copy.deepcopy(grad)
                    state['x_k'] = copy.deepcopy(p.data)

                gamma_k = 2 * group['lips'] / (state['step'] + 1)

                avg_grad = state['avg_grad']
                avg_grad.mul_(state['step'])
                avg_grad.add_(grad)
                avg_grad.div_(state['step'] + 1)

                delta_k = torch.add(grad, avg_grad, alpha=-1)

                state['v_k'] += torch.sum(delta_k * delta_k).item()

                h_k = math.sqrt(state['v_k'])
                alpha_k_1 = 2 / (state['step'] + 3)
                coef = 1 / (gamma_k + group['beta'] * h_k)
                x_k_1 = state['x_k']
                x_k_1.add_(grad, alpha=-coef)

                p.data.mul_(1 - alpha_k_1)
                p.data.add_(x_k_1, alpha=alpha_k_1)
                p.data.add_(
                    grad, alpha=-(1 - alpha_k_1) * state['alpha_k'] * coef
                )

                state['alpha_k'] = alpha_k_1
                state['step'] += 1

        return loss
