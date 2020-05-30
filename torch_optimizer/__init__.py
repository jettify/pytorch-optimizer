from typing import Type, List, Dict

from pytorch_ranger import Ranger, RangerQH, RangerVA
from torch.optim.optimizer import Optimizer

from .accsgd import AccSGD
from .adabound import AdaBound
from .adamod import AdaMod
from .diffgrad import DiffGrad
from .lamb import Lamb
from .lookahead import Lookahead
from .novograd import NovoGrad
from .pid import PID
from .qhadam import QHAdam
from .qhm import QHM
from .radam import RAdam
from .sgdw import SGDW
from .yogi import Yogi
from .shampoo import Shampoo


__all__ = (
    'AccSGD',
    'AdaBound',
    'AdaMod',
    'DiffGrad',
    'Lamb',
    'Lookahead',
    'NovoGrad',
    'PID',
    'QHAdam',
    'QHM',
    'RAdam',
    'Ranger',
    'RangerQH',
    'RangerVA',
    'SGDW',
    'Shampoo',
    'Yogi',
    # utils
    'get',
)
__version__ = '0.0.1a13'


_package_opts = [
    AccSGD,
    AdaBound,
    AdaMod,
    DiffGrad,
    Lamb,
    Lookahead,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    SGDW,
    Shampoo,
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
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
