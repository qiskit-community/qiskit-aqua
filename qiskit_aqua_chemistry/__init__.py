# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Main qiskit_aqua_chemistry public functionality."""

from .aqua_chemistry_error import AquaChemistryError
from .preferences import Preferences
from .qmolecule import QMolecule
from .aqua_chemistry import AquaChemistry
from .fermionic_operator import FermionicOperator
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config)

__version__ = '0.3.0'


def get_aqua_chemistry_logging():
    """
    Returns the current Aqua Chemistry logging level

    Returns:
        logging level
    """
    return get_logging_level()


def set_aqua_chemistry_logging(level):
    """
    Updates the Aqua Chemistry logging with the appropriate logging level

    Args:
        level (int): minimum severity of the messages that are displayed.
    """
    set_logging_config(build_logging_config(level))


__all__ = ['AquaChemistryError',
           'Preferences',
           'QMolecule',
           'AquaChemistry',
           'FermionicOperator',
           'get_aqua_chemistry_logging',
           'set_aqua_chemistry_logging']
