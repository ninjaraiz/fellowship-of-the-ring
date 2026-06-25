"""
stats/base.py
=============
Abstract base class that every FRODO Stats class must implement.

Responsibility
------------------
The Stats class is responsible for computing statistics from the data in the FRODO instance. It
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    # Imported only for type checkers; avoids circular import at runtime.
    from ..frodo import FRODO

class BaseStats(ABC):
    """
    Abstract base class for all FRODO Stats classes.
    
    Subclasses must implement ``compute_stats``.  All other methods are
    optional and can be added as the format requires.
    
    Attributes
    ----------
    db : FRODO
        Reference to the parent FRODO instance.  Gives access to
        ``db.data_dict``, ``db.metadata``, ``db.sim_metadata``, etc.
    """
    
    def __init__(self, db: 'FRODO'):
        """
        Parameters
        ----------
        db: FRODO
            The parent FRODO instance that owns this Stats object.
        """
        self.db = db
        
    # ── Abstract interface ────────────────────────────────────────────────────
