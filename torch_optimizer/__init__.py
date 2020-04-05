from typing import Optional, Type, List, Dict

from pytorch_ranger import Ranger, RangerQH, RangerVA
from torch.optim import Optimizer

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
    'RangerVA',
    # utils
    'get',
)
__version__ = '0.0.1a11'


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
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
]  # type: List[Optimizer]


_NAME_OPTIM_MAP = {
    opt.__name__.lower(): opt for opt in _package_opts
}  # type: Dict[str, Optimizer]


def get(name: str) -> Optional[Type[Optimizer]]:
    r"""Returns an optimizer class from its name. Case insensitive.

    Args:
        name: the optimizer name.
    """
    return _NAME_OPTIM_MAP.get(name.lower())
