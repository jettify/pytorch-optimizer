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


def get(identifier):
    """ Returns an optimizer class from a string. Returns `identifier` if it
    is already callable.

    Args:
        identifier (Union[str, Callable, None]): the optimizer identifier.

    Returns:
        torch.optim.Optimizer or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Could not interpret optimizer identifier: ' +
                             str(identifier))
        return cls
    else:
        raise ValueError('Could not interpret optimizer identifier: ' +
                         str(identifier))

