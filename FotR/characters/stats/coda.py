"""
stats/coda.py
=============
Stats class for the CODA CFD solver format.

"""

from typing import Literal, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
import h5py

import pyLOM as SMEAGOL

from ..sam import SAM
from .base import BaseStats

if TYPE_CHECKING:
    from ..frodo import FRODO

class CODAStats(BaseStats):
    """
    Parameters
    ----------
    db : FRODO
        Parent FRODO instance.
    """

    def __init__(self, db: 'FRODO'):
        super().__init__(db)