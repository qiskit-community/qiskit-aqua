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
Chemistry Core (:mod:`qiskit.chemistry.core`)
=============================================
The core was designed to be an extensible system that took a :class:`QMolecule`
and created output which was ready to be input directly to an Aqua algorithm
in the form of a qubit operator and list of auxiliary operators such as
dipole moments, spin, number of particles etc.

The one implementation here, :class:`Hamiltonian`, in essence wraps the
:class:`FermionicOperator` to provide easier, convenient access to common
capabilities such that the :class:`FermionicOperator` class need not be
used directly.

.. currentmodule:: qiskit.chemistry.core

Core Base Class
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ChemistryOperator

Core
====

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Hamiltonian
   TransformationType
   QubitMappingType

"""

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
