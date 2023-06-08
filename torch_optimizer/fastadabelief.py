import math
import torch
from torch.optim.optimizer import Optimizer

class FastAdaBelief(Optimizer):
    """
    Implements the FastAdaBelief optimization algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): L2 regularization coefficient (default: 0).
        amsgrad (bool, optional): Whether to use the AMSGrad variant of AdaBelief (default: False).
        weight_decouple (bool, optional): Whether to decouple the weight decay from the gradient-based optimization step (default: False).
        fixed_decay (bool, optional): Whether to use a fixed decay factor for weight decay (default: False).
        rectify (bool, optional): Whether to use rectification in the optimization step (default: False).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(FastAdaBelief, self).__init__(params, defaults)

        self._weight_decouple = weight_decouple
        self._rectify = rectify
        self._fixed_decay = fixed_decay

    def __setstate__(self, state):
        super(FastAdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value evaluated by the closure.
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
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self._weight_decouple:
                    if not self._fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self._rectify:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])

                    if state['rho_t'] > 4:
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t)
                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:
                        p.data.add_(-group['lr'], exp_avg)

        return loss

