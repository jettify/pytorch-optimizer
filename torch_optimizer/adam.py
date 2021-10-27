import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .types import Params

__all__ = ('Adam', 'AdamW')


def adam(
    params: Params,
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    adamd_bias_correction: bool
):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(
                max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
            )
            # Use the max. for normalizing running avg. of gradient
            denom = (
                max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)
            ).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if adamd_bias_correction:
            step_size = lr
        else:
            step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


def adamw(
    params: Params,
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    adamd_bias_correction: bool
):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(
                max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
            )
            # Use the max. for normalizing running avg. of gradient
            denom = (
                max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)
            ).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        if adamd_bias_correction:
            step_size = lr
        else:
            step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                            \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta)
                \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},
             \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\:
                \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                         \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}    \\
            &\hspace{5mm}g_t           \leftarrow
                \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                  \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}      \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} +
                (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} +
                (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow
                m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow
                v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad              \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow
                \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} -
                \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)       \\
            &\hspace{5mm}\textbf{else}                                  \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} -
                \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big) \\
            &\rule{110mm}{0.4pt}                   \\[-1.ex]
            &\bf{return} \:  \theta_t       \\[-1.ex]
            &\rule{110mm}{0.4pt}        \\[-1.ex]
       \end{aligned}
    For further details regarding the algorithm we refer to `Adam: A Method for
        Stochastic Optimization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        adamd_bias_correction (boolean, optional): When performing bias
            correction, only correct the denominator to avoid inflating step
            sizes early in training as suggested in `AdamD: Improved
            bias-correction in Adam`_ (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _AdamD: Improved bias-correction in Adam:
        https://arxiv.org/abs/2110.10828
    """

    def __init__(
        self,
        params: Params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adamd_bias_correction: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamd_bias_correction=adamd_bias_correction,
        )
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('adamd_bias_correction', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq.
                            # grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                adamd_bias_correction=group['adamd_bias_correction'],
            )
        return loss


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                      \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \:
                    f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}           \\
            &\hspace{13mm}      \lambda \text{(weight decay)}, \:  amsgrad  \\
            &\textbf{initialize} : m_0 \leftarrow 0
                \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \:
                    \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                      \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}  \\
            &\hspace{5mm}g_t           \leftarrow
                \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} -
                \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow
                \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow
                \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow
                m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow
                v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad             \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow
                \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1}
                - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)     \\
            &\hspace{5mm}\textbf{else}                                \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} -
                \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)   \\
            &\rule{110mm}{0.4pt}               \\[-1.ex]
            &\bf{return} \:  \theta_t        \\[-1.ex]
            &\rule{110mm}{0.4pt}             \\[-1.ex]
       \end{aligned}
    For further details regarding the algorithm we refer to `Decoupled Weight
        Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient
            (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        adamd_bias_correction (boolean, optional): When performing bias
            correction, only correct the denominator to avoid inflating step
            sizes early in training as suggested in `AdamD: Improved
            bias-correction in Adam`_ (default: False)
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _AdamD: Improved bias-correction in Adam:
        https://arxiv.org/abs/2110.10828
    """

    def __init__(
        self,
        params: Params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        adamd_bias_correction: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamd_bias_correction=adamd_bias_correction,
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'AdamW does not support sparse gradients'
                    )
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq.
                        #  grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                adamd_bias_correction=group['adamd_bias_correction'],
            )

        return loss
