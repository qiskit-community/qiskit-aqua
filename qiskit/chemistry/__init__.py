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

"""Main public functionality."""

from .qiskit_chemistry_error import QiskitChemistryError
from .preferences import Preferences
from .qmolecule import QMolecule
from .qiskit_chemistry_problem import ChemistryProblem
from .qiskit_chemistry import (QiskitChemistry, run_experiment, run_driver_to_json)
from .fermionic_operator import FermionicOperator
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_qiskit_chemistry_logging,
                       set_qiskit_chemistry_logging)

__version__ = '0.4.3'

__all__ = ['QiskitChemistryError',
           'Preferences',
           'QMolecule',
           'ChemistryProblem',
           'QiskitChemistry',
           'run_experiment',
           'run_driver_to_json',
           'FermionicOperator',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_qiskit_chemistry_logging',
           'set_qiskit_chemistry_logging']
