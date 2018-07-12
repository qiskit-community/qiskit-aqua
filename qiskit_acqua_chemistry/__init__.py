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

"""Main qiskit_acqua_chemistry public functionality."""

from .acqua_chemistry_error import ACQUAChemistryError
from .qmolecule import QMolecule
from .acqua_chemistry import ACQUAChemistry
from .fermionic_operator import FermionicOperator

__version__ = '0.1.1'

__all__ = ['ACQUAChemistryError', 'QMolecule', 'ACQUAChemistry', 'FermionicOperator']
