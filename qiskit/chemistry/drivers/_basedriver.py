# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for driver modules.
"""

from abc import ABC, abstractmethod
from enum import Enum


class UnitsType(Enum):
    """ Units Type Enum """
    ANGSTROM = 'Angstrom'
    BOHR = 'Bohr'


class HFMethodType(Enum):
    """ HFMethodType Enum """
    RHF = 'rhf'
    ROHF = 'rohf'
    UHF = 'uhf'


class BaseDriver(ABC):
    """
    Base class for Drivers.

    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        """ runs driver """
        pass
