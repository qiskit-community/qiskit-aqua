# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Chemistry specific Aqua algorithms (:mod:`qiskit.chemistry.algorithms`)
=======================================================================
These are chemistry specific algorithms for Aqua. As they rely on chemistry
specific knowledge and/or function they are here in chemistry rather than in Aqua.

.. currentmodule:: qiskit.chemistry.algorithms

Chemistry Algorithms
====================
These are algorithms configured and/or functioning using chemistry specific knowledge. See also
the Aqua :mod:`~qiskit.aqua.algorithms` for other algorithms in these categories which may also
be used for chemistry problems such as :class:`~qiskit.aqua.algorithms.VQE`.

Eigensolvers
++++++++++++
Algorithms that can find the eigenvalues of an operator, i.e. excited states for chemistry.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QEomVQE
   QEomEE

Minimum Eigensolvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator, i.e. ground state for chemistry.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VQEAdapt

"""

from .eigen_solvers import QEomVQE, QEomEE
from .minimum_eigen_solvers import VQEAdapt, VQEAdaptResult

__all__ = [
    'QEomVQE',
    'QEomEE',
    'VQEAdapt',
    'VQEAdaptResult',
]
