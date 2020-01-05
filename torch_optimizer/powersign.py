import torch
from torch.optim import Optimizer


class PowerSign(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
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
        super(PowerSign, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PowerSign, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # print(weight_decay, momentum, dampening , nesterov)
            # They use pytorch functions most likely to speed up the operations
            # More research is defiently needed to understand how optimisers
            # are implemented in pytorch.

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(
                            p.data.size()
                        )
                        buf = torch.add(
                            buf.mul(momentum), (d_p.mul(1 - momentum))
                        )
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf.mul(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        # This is the gradient update rule that was found in
                        # the paper Neural Optimizer search with reinfrocmement
                        # learning
                        d_p = torch.mul(
                            torch.exp(torch.mul(d_p.sign(), buf.sign())), d_p
                        )
                        # print(d_p)
                        # d_p = buf

                p.data.add_(-group['lr'], d_p)
        return loss
