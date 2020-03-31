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
