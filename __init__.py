from .frodo import FRODO
from .EarendilsLight import EarendilsLight
from .sam import SAM
from .gandalf import GANDALF
from .merry_pippin import MERRY_PIPPIN
from .legolas import LEGOLAS
from .aragorn import ARAGORN, RunResult, XFoilCase, xfoil_parser
import sys
try:
    sys.path.append('/home/ninjaraiz/anaconda3/repos/pyLowOrder/')
except:
    sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')

__all__ = [
    "FRODO",
    "EarendilsLight",
    "SAM",
    "GANDALF",
    "MERRY_PIPPIN",
    "LEGOLAS",
    "ARAGORN",
    "RunResult",
    "XFoilCase",
    "xfoil_parser"

]