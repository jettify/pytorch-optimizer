from typing import Optional, Type
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
from pytorch_ranger import Ranger, RangerQH, RangerVA
from torch import optim


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
    'SGDW',
    'Yogi',
    'Ranger',
    'RangerQH',
    'RangerVA'
)
__version__ = '0.0.1a11'


package_opts = [
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
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
]

builtin_opts = [
    optim.Adadelta,
    optim.Adagrad,
    optim.Adam,
    optim.AdamW,
    optim.SparseAdam,
    optim.Adamax,
    optim.ASGD,
    optim.SGD,
    optim.Rprop,
    optim.RMSprop,
    optim.LBFGS
]

NAME_OPTIM_MAP = {
    opt.__name__.lower(): opt for opt in package_opts + builtin_opts
}


def get(name: str,) -> Optional[Type[optim.Optimizer]]:
    r"""Returns an optimizer class from its name. Case insensitive.

    Args:
        name: the optimizer name.
    """
    if isinstance(name, str):
        cls = NAME_OPTIM_MAP.get(name.lower())
        if cls is None:
            raise ValueError('Could not interpret optimizer name: ' +
                             str(name))
        return cls
    raise ValueError('Could not interpret optimizer name: ' +
                     str(name))
