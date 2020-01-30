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
==============================================================
Chemistry application stack for Aqua (:mod:`qiskit.chemistry`)
==============================================================
This is the chemistry domain logic....

.. currentmodule:: qiskit.chemistry

Chemistry Error
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QiskitChemistryError

Chemistry Classes
==================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   FermionicOperator
   QMolecule
   MP2Info

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
   components
   core
   drivers

"""

from .qiskit_chemistry_error import QiskitChemistryError
from .qmolecule import QMolecule
from .fermionic_operator import FermionicOperator
from .mp2info import MP2Info
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_qiskit_chemistry_logging,
                       set_qiskit_chemistry_logging)

__all__ = ['QiskitChemistryError',
           'QMolecule',
           'FermionicOperator',
           'MP2Info',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_qiskit_chemistry_logging',
           'set_qiskit_chemistry_logging']
