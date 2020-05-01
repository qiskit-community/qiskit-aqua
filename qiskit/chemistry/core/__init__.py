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

""" Chemistry Core Packages """

from .chemistry_operator import ChemistryOperator
from .hamiltonian import Hamiltonian, TransformationType, QubitMappingType
from ._discover_chemoperator import (OPERATORS_ENTRY_POINT,
                                     refresh_operators,
                                     register_chemistry_operator,
                                     deregister_chemistry_operator,
                                     get_chemistry_operator_class,
                                     get_chem_operator_config,
                                     local_chemistry_operators)

__all__ = ['ChemistryOperator',
           'Hamiltonian',
           'TransformationType',
           'QubitMappingType',
           'OPERATORS_ENTRY_POINT',
           'refresh_operators',
           'register_chemistry_operator',
           'deregister_chemistry_operator',
           'get_chemistry_operator_class',
           'get_chem_operator_config',
           'local_chemistry_operators']
