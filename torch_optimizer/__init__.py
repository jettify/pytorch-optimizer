"""torch-optimizer -- collection of of optimization algorithms for PyTorch.

API and usage patterns are the same as `torch.optim`__

Example
-------

>>> import torch_optimizer as optim
# model = ...
>>> optimizer = optim.DiffGrad(model.parameters(), lr=0.001)
>>> optimizer.step()

See documentation for full list of supported optimizers.

__ https://pytorch.org/docs/stable/optim.html#module-torch.optim
"""
from typing import Dict, List, Type

from pytorch_ranger import Ranger, RangerQH, RangerVA
from torch.optim.optimizer import Optimizer

from .a2grad import A2GradExp, A2GradInc, A2GradUni
from .accsgd import AccSGD
from .adabelief import AdaBelief
from .adabound import AdaBound
from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamod import AdaMod
from .adamp import AdamP
from .aggmo import AggMo
from .apollo import Apollo
from .diffgrad import DiffGrad
from .lamb import Lamb
from .lars import LARS
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .novograd import NovoGrad
from .pid import PID
from .qhadam import QHAdam
from .qhm import QHM
from .radam import RAdam
from .sgdp import SGDP
from .sgdw import SGDW
from .shampoo import Shampoo
from .swats import SWATS
from .yogi import Yogi

__all__ = (
    'A2GradExp',
    'A2GradInc',
    'A2GradUni',
    'AccSGD',
    'AdaBelief',
    'AdaBound',
    'AdaMod',
    'Adafactor',
    'Adahessian',
    'AdamP',
    'AggMo',
    'Apollo',
    'DiffGrad',
    'LARS',
    'Lamb',
    'Lookahead',
    'MADGRAD',
    'NovoGrad',
    'PID',
    'QHAdam',
    'QHM',
    'RAdam',
    'Ranger',
    'RangerQH',
    'RangerVA',
    'SGDP',
    'SGDW',
    'SWATS',
    'Shampoo',
    'Yogi',
    # utils
    'get',
)
__version__ = '0.3.1a0'


_package_opts = [
    AdaBelief,
    AccSGD,
    AdaBound,
    AdaMod,
    AdamP,
    AggMo,
    DiffGrad,
    LARS,
    Lamb,
    Lookahead,
    MADGRAD,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    Ranger,
    RangerQH,
    RangerVA,
    SGDP,
    SGDW,
    SWATS,
    Shampoo,
    Yogi,
]  # type: List[Type[Optimizer]]


_NAME_OPTIM_MAP = {
    opt.__name__.lower(): opt for opt in _package_opts
}  # type: Dict[str, Type[Optimizer]]


def get(name: str) -> Type[Optimizer]:
    r"""Returns an optimizer class from its name. Case insensitive.

    Args:
        name: the optimizer name.
    """
    optimizer_class = _NAME_OPTIM_MAP.get(name.lower())
    if optimizer_class is None:
        raise ValueError('Optimizer {} not found'.format(name))
    return optimizer_class
