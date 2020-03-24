import torch
from torch.optim.optimizer import Optimizer
from .types import OptLossClosure, Params, OptFloat


__all__ = ('QHM',)


class QHM(Optimizer):

    GRAD = 'grad'
    DIRECT = 'direct'

    r"""Implements quasi-hyperbolic momentum (QHM)  optimization algorithm.

    It has been proposed in `Quasi-hyperbolic momentum and Adam for deep
    learning`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (:math:`\beta` from the paper)
        nu: immediate discount factor (:math:`\nu` from the paper)
        weight_decay: weight decay (L2 regularization coefficient, times two)
            (default: 0.0)
        weight_decay_type: method of applying the weight decay:
            ``"grad"`` for accumulation in the gradient
            (same as :class:`torch.optim.SGD`) or
            ``"direct"`` for direct application to the parameters
            (default: ``"grad"``)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.QHM(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()


    __ https://arxiv.org/abs/1810.06801

    Note:
        Reference code: https://github.com/facebookresearch/qhoptim
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nu: float = 0.7,
        weight_decay: float = 0.0,
        weight_decay_type: str = 'grad',
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if weight_decay_type not in (self.GRAD, self.DIRECT):
            _type = weight_decay_type
            msg = 'Invalid weight_decay_type value: {}'.format(_type)
            raise ValueError(msg)

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'nu': nu,
            'weight_decay': weight_decay,
            'weight_decay_type': weight_decay_type,
        }
        super(QHM, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, nu, momentum = group['lr'], group['nu'], group['momentum']
            weight_decay, weight_decay_type = (
                group['weight_decay'],
                group['weight_decay_type'],
            )

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0:
                    if weight_decay_type == self.GRAD:
                        d_p.add_(weight_decay, p.data)
                    else:
                        p.data.mul_(1.0 - lr * weight_decay)

                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = param_state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(1.0 - momentum, d_p)

                p.data.add_(-lr * nu, momentum_buffer)
                p.data.add_(-lr * (1.0 - nu), d_p)

        return loss
