import copy
from torch.optim.optimizer import Optimizer

from .types import OptLossClosure, Params, OptFloat


__all__ = ('AccSGD',)


class AccSGD(Optimizer):
    r"""Implements AccSGD algorithm.

    It has been proposed in `On the insufficiency of existing momentum
    schemes for Stochastic Optimization`__ and  `Accelerating Stochastic
    Gradient Descent For Least Squares Regression`__

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        kappa: ratio of long to short step (default: 1000)
        xi: statistical advantage parameter (default: 10)
        small_const: any value <=1 (default: 0.7)
        weight_decay: weight decay (L2 penalty) (default: 0)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AccSGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ https://arxiv.org/abs/1704.08227
     __ https://arxiv.org/abs/1803.05591

    Note:
        Reference code: https://github.com/rahulkidambi/AccSGD
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        kappa: float = 1000.0,
        xi: float = 10.0,
        small_const: float = 0.7,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            kappa=kappa,
            xi=xi,
            small_const=small_const,
            weight_decay=weight_decay,
        )
        super(AccSGD, self).__init__(params, defaults)

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
            large_lr = (group['lr'] * group['kappa']) / (group['small_const'])
            alpha = 1.0 - (
                (group['small_const'] * group['small_const'] * group['xi'])
                / group['kappa']
            )
            beta = 1.0 - alpha
            zeta = group['small_const'] / (group['small_const'] + beta)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = copy.deepcopy(p.data)
                buf = param_state['momentum_buffer']
                buf.mul_((1.0 / beta) - 1.0)
                buf.add_(-large_lr, d_p)
                buf.add_(p.data)
                buf.mul_(beta)

                p.data.add_(-group['lr'], d_p)
                p.data.mul_(zeta)
                p.data.add_(1.0 - zeta, buf)

        return loss
