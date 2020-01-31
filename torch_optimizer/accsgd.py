import copy
from torch.optim import Optimizer


__all__ = ('AccSGD',)


class AccSGD(Optimizer):
    r"""Implements the algorithm proposed in
    https://arxiv.org/pdf/1704.08227.pdf, which is a provably accelerated
    method for stochastic optimization. This has been employed in
    https://openreview.net/forum?id=rJTutzbA- for training several deep
    learning models of practical interest. This code has been implemented by
    building on the construction of the SGD optimization module found in
    pytorch codebase.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        kappa (float, optional): ratio of long to short step (default: 1000)
        xi (float, optional): statistical advantage parameter (default: 10)
        small_const (float, optional): any value <=1 (default: 0.7)
    Example:
        >>> from AccSGD import *
        >>> optimizer = AccSGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        kappa=1000.0,
        xi=10.0,
        small_const=0.7,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = dict(
            lr=lr,
            kappa=kappa,
            xi=xi,
            small_const=small_const,
            weight_decay=weight_decay,
        )
        super(AccSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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
