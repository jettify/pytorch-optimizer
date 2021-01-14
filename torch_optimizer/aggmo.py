from typing import List, Tuple, Type, TypeVar, Union

import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params

__all__ = ('AggMo',)


T = TypeVar('T', bound='AggMo')


class AggMo(Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent.

    It has been proposed in `Aggregated Momentum: Stability Through Passive
    Damping`__

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AggMo(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ https://arxiv.org/abs/1804.00325

    Note:
        Reference code: https://github.com/AtheMathmo/AggMo/blob/master/aggmo.py  # noqa
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Union[List[float], Tuple[float, ...]] = (0.0, 0.9, 0.99),
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))

        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                msg = 'Invalid beta parameter at index 1: {}'.format(betas[i])
                raise ValueError(msg)

        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)

    @classmethod
    def from_exp_form(
        cls: Type[T],
        params: Params,
        lr: float = 1e-3,
        a: float = 0.1,
        k: int = 3,
        weight_decay: float = 0,
    ) -> T:
        if lr <= 0.0:
            raise ValueError('Invalid parameter k: {}'.format(k))

        betas = [1 - a ** i for i in range(k)]  # type: List[float]
        return cls(params, lr, betas, weight_decay)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            total_mom = float(len(betas))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    for beta in betas:
                        param_state['momentum_buffer'][
                            beta
                        ] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                for beta in betas:
                    buf = param_state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(buf, alpha=group['lr'] / total_mom)
        return loss
