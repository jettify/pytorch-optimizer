import math

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ('AdaBelief',)


class AdaBelief(Optimizer):
    r"""Implements AdaBelief Optimizer Algorithm.
    It has been proposed in `AdaBelief Optimizer, adapting stepsizes by
    the belief in observed gradients`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 0.001)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple:  If set as True, then the optimizer uses decoupled
            weight decay as in AdamW (default: False)
        fixed_decay : This is used when
            weight_decouple is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in
            this case, the weight decay ratio decreases with learning
            rate (lr).  (default: False)
        rectify: (default: False) If set as True, then perform the rectified
            update similar to RAdam

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AdaBelief(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/2010.07468

    Note:
        Reference code: https://github.com/juntang-zhuang/Adabelief-Optimizer
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        rectify: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(AdaBelief, self).__init__(params, defaults)

        self._weight_decouple = weight_decouple
        self._rectify = rectify
        self._fixed_decay = fixed_decay

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of
                        # sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # perform weight decay, check if decoupled weight decay
                if self._weight_decouple:
                    if not self._fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running
                    # avg. till now
                    torch.max(
                        max_exp_avg_var, exp_avg_var, out=max_exp_avg_var
                    )

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])
                else:
                    denom = (
                        exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                if not self._rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state[
                        'step'
                    ] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])

                    if (
                        state['rho_t'] > 4
                    ):  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)

        return loss
