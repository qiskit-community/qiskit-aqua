# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Eigenvalues (:mod:`qiskit.aqua.components.eigs`)
================================================
These are components designed to find eigenvalues. They were initially designed for use by
:class:`~qiskit.aqua.algorithms.HHL` which remains their currently principal usage.

.. currentmodule:: qiskit.aqua.components.eigs

Eigenvalues Base Class
======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigenvalues

Eigenvalues
===========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EigsQPE

"""

from .eigs import Eigenvalues
from .eigs_qpe import EigsQPE

__all__ = ['EigsQPE', 'Eigenvalues']
