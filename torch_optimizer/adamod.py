import math
import torch
from torch.optim import Optimizer


__all__ = ('AdaMod',)


class AdaMod(Optimizer):
    """Implements AdaMod algorithm with Decoupled Weight Decay
    arxiv.org/abs/1711.05101)
    It has been proposed in
    `Adaptive and Momental Bounds for Adaptive Learning Rate Methods`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float, optional): smoothing coefficient for adaptive learning
            rates (default: 0.9999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    https://github.com/lancopku/AdaMod
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta3=0.999,
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f'Invalid beta3 parameter: {beta3}')
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, eps=eps, weight_decay=weight_decay
        )
        super(AdaMod, self).__init__(params, defaults)

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
                if grad.is_sparse:
                    msg = 'AdaMod does not support sparse gradients'
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of actual learning rates
                    state['exp_avg_lr'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_lr = (
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['exp_avg_lr'],
                )
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (
                    group['lr']
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # Applies momental bounds on actual learning rates
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom)
                exp_avg_lr.mul_(group['beta3']).add_(
                    1 - group['beta3'], step_size
                )
                step_size = torch.min(step_size, exp_avg_lr)
                step_size.mul_(exp_avg)

                p.data.add_(-step_size)

        return loss
