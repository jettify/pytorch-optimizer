from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer

from .types import OptLossClosure, OptFloat, State


__all__ = ('Lookahead',)


class Lookahead(Optimizer):
    r"""Implements Lookahead optimization algorithm.

    It has been proposed in `Lookahead Optimizer: k steps forward, 1
    step back`__

    Arguments:
        optimizer: base inner optimizer optimize
        k: number of lookahead steps (default: 5)
        alpha: linear interpolation factor. 1.0 recovers the inner optimizer.
            (default: 5)

    Example:
        >>> import torch_optimizer as optim
        >>> yogi = optim.Yogi(model.parameters(), lr=0.1)
        >>> optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1907.08610
    """

    def __init__(
        self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5
    ) -> None:
        if k < 0.0:
            raise ValueError('Invalid number of lookahead steps: {}'.format(k))
        if alpha < 0:
            raise ValueError(
                'Invalid linear interpolation factor: {}'.format(alpha)
            )

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group['counter'] = 0

    def _update(self, group) -> None:
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.clone(fast.data).detach()

            slow = param_state['slow_param']
            fast.data.mul_(self.alpha).add_(1.0 - self.alpha, slow)
            slow.data.copy_(fast)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self._update(group)
            group['counter'] = (group['counter'] + 1) % self.k
        return loss

    def state_dict(self) -> State:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict: State) -> None:
        r"""Loads the optimizer state.

        Arguments:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def zero_grad(self) -> None:
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self.optimizer.zero_grad()

    def __repr__(self) -> str:
        base_str = self.optimizer.__repr__()
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += 'k: {}\n'.format(self.k)
        format_string += 'alpha: {}\n'.format(self.alpha)
        format_string += base_str
        format_string += '\n'
        format_string += ')'
        return format_string
