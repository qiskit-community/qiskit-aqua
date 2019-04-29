# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM Corp. 2017 and later.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Main public functionality."""

from .version import __version__
from .qiskit_chemistry_error import QiskitChemistryError
from .qmolecule import QMolecule
from .qiskit_chemistry_problem import ChemistryProblem
from .qiskit_chemistry import (QiskitChemistry, run_experiment, run_driver_to_json)
from .fermionic_operator import FermionicOperator
from .mp2info import MP2Info
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_qiskit_chemistry_logging,
                       set_qiskit_chemistry_logging)

__all__ = ['__version__',
           'QiskitChemistryError',
           'QMolecule',
           'ChemistryProblem',
           'QiskitChemistry',
           'run_experiment',
           'run_driver_to_json',
           'FermionicOperator',
           'MP2Info',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_qiskit_chemistry_logging',
           'set_qiskit_chemistry_logging']
