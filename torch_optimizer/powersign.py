import math
import torch
from torch.optim import Optimizer


__all__ = ('PowerSign',)


class PowerSign(Optimizer):
    """Implements PowerSign algorithm.

    Described in `Neural Optimizer Search with Reinforcement Learning`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficients used for computing
            running averages of gradient (default: 0.9)
        alpha (float, optional): term powered to
            the internal_decay * sign(g) * sign(m) (default: math.e)
        sign_internal_decay(callable, optional): a function that returns
            an internal decay calculated based on the current training step and
            the total number of training steps.
            If None, the internal decay is assumed to be 1.

    .. _Neural Optimizer Search with Reinforcement Learning:
        https://arxiv.org/abs/1709.07417
    """

    # https://github.com/cydonia999/AddSign_PowerSign_in_PyTorch/blob/master/torch/optim/powersign.py

    def __init__(
        self, params, lr=1e-3, beta=0.9, alpha=math.e, sign_internal_decay=None
    ):
        if sign_internal_decay is not None and not callable(
            sign_internal_decay
        ):
            type_name = type(sign_internal_decay).__name__
            msg = '{} is not a callable'.format(type_name)
            raise TypeError(msg)

        if alpha <= 0:
            raise ValueError('alpha should be > 0.')
        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            sign_internal_decay=sign_internal_decay
            if sign_internal_decay is not None
            else lambda _: 1,
        )
        super(PowerSign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta = group['beta']
                alpha = group['alpha']

                state['step'] += 1
                internal_decay = group['sign_internal_decay'](
                    state['step'] - 1
                )

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)

                power_sign = grad.mul(
                    torch.pow(
                        alpha, internal_decay * grad.sign() * exp_avg.sign()
                    )
                )
                p.data.add_(-group['lr'], power_sign)

        return loss
