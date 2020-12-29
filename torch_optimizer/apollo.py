import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params


class Apollo(Optimizer):
    r"""Implements Apollo Optimizer Algorithm.

    It has been proposed in `Apollo: An Adaptive Parameter-wise Diagonal
    Quasi-Newton Method for Nonconvex Stochastic Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        beta: coefficient used for computing
            running averages of gradient (default: 0.9)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-4)
        warmup: number of warmup steps (default: 0)
        init_lr: initial learning rate for warmup (default: 0.01)
        weight_decay: weight decay (L2 penalty) (default: 0)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Apollo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2009.13586

    Note:
        Reference code: https://github.com/XuezheMax/apollo
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        beta: float = 0.9,
        eps: float = 1e-4,
        warmup: int = 0,
        init_lr: float = 0.01,
        weight_decay: float = 0,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError('Invalid beta parameter: {}'.format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if not 0.0 <= warmup:
            raise ValueError('Invalid warmup updates: {}'.format(warmup))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError(
                'Invalid initial learning rate: {}'.format(init_lr)
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
        )
        super(Apollo, self).__init__(params, defaults)

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

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['approx_hessian'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Previous update direction
                    state['update'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # Calculate current lr
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state[
                        'step'
                    ] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Atom does not support sparse gradients.'
                    )

                # Perform step weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                beta = group['beta']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # Update the running average grad
                delta_grad = grad - exp_avg_grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(group['eps'])
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = (
                    delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha)
                    - B.mul(v_sq).sum()
                )

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                denom = B.abs().clamp_(min=1)
                d_p.copy_(exp_avg_grad.div(denom))

                p.data.add_(d_p, alpha=-curr_lr)

        return loss
