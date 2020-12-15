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

**DEPRECATED** See the Excited States Solvers section below

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QEomVQE
   QEomEE

Excited State Solvers
+++++++++++++++++++++
Algorithms that can find the eigenvalues of an operator, i.e. excited states for chemistry.

The interface for such solvers,

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExcitedStatesSolver

the solvers themselves

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExcitedStatesEigensolver
   QEOM

and factories to provision Quantum and/or Classical algorithms upon which the above solvers may
depend

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EigensolverFactory
   NumPyEigensolverFactory

Ground State Solvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator, i.e. ground state for chemistry.

The interface for such solvers,

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GroundStateSolver

the solvers themselves

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AdaptVQE
   GroundStateEigensolver
   OrbitalOptimizationVQE

and factories to provision Quantum and/or Classical algorithms upon which the above solvers may
depend

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolverFactory
   NumPyMinimumEigensolverFactory
   VQEUCCSDFactory
   VQEUVCCSDFactory

Minimum Eigensolvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator, i.e. ground state for chemistry.

**DEPRECATED** See the Ground State Solvers section above

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VQEAdapt


Potential Energy Surface Samplers
+++++++++++++++++++++++++++++++++
Algorithms that can compute potential energy surfaces.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BOPESSampler

The samplers include extrapolators to facilitate convergence across a set of points and support
of various potentials. More detail may be found in the sub-module linked below

.. autosummary::
   :toctree:

   pes_samplers

"""

from .eigen_solvers import QEomVQE, QEomEE
from .minimum_eigen_solvers import VQEAdapt, VQEAdaptResult

from .excited_states_solvers import (ExcitedStatesEigensolver, ExcitedStatesSolver, QEOM,
                                     EigensolverFactory, NumPyEigensolverFactory)
from .ground_state_solvers import (AdaptVQE, GroundStateEigensolver, GroundStateSolver,
                                   OrbitalOptimizationVQE, MinimumEigensolverFactory,
                                   NumPyMinimumEigensolverFactory, VQEUCCSDFactory,
                                   VQEUVCCSDFactory)
from .pes_samplers import BOPESSampler

__all__ = [
    'QEomVQE',
    'QEomEE',
    'VQEAdapt',
    'VQEAdaptResult',
    'ExcitedStatesEigensolver',
    'ExcitedStatesSolver',
    'QEOM',
    'EigensolverFactory',
    'NumPyEigensolverFactory',
    'AdaptVQE',
    'GroundStateEigensolver',
    'GroundStateSolver',
    'OrbitalOptimizationVQE',
    'MinimumEigensolverFactory',
    'NumPyMinimumEigensolverFactory',
    'VQEUCCSDFactory',
    'VQEUVCCSDFactory',
    'BOPESSampler',
]
