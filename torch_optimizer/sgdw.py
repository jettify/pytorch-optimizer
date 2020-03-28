import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params, State


__all__ = ('SGDW',)


class SGDW(Optimizer):
    r"""Implements SGDW algorithm.

    It has been proposed in `Decoupled Weight Decay Regularization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        dampening: dampening for momentum (default: 0)
        nesterov: enables Nesterov momentum (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.SGDW(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1711.05101

    Note:
        Reference code: https://github.com/pytorch/pytorch/pull/22466
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if dampening < 0.0:
            raise ValueError('Invalid dampening value: {}'.format(dampening))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening'
            )
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state: State) -> None:
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if p.grad.is_sparse:
                    msg = (
                        'SGDW does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # Apply momentum
                p.data.add_(-group['lr'], d_p)

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-group['lr'], weight_decay)
        return loss
