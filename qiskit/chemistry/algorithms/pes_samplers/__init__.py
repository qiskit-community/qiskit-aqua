# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Potential energy surface samplers (:mod:`qiskit.chemistry.algorithms.pes_samplers`)
===================================================================================
Potential energy surface samplers.

.. currentmodule:: qiskit.chemistry.algorithms.pes_samplers

Algorithms that can compute potential energy surfaces.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BOPESSampler

When used with variational solvers, such as VQE, when computing a set of points there is support
for extrapolation from prior solution(s) to bootstrap the algorithm with a better starting point
to facilitate convergence. Extrapolators are:

 .. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Extrapolator
   DifferentialExtrapolator
   PCAExtrapolator
   PolynomialExtrapolator
   SieveExtrapolator
   WindowExtrapolator

There is also a set of support function for potentials:

 .. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    EnergySurface1DSpline
    HarmonicPotential
    MorsePotential
    EnergySurfaceBase
    PotentialBase
    VibronicStructureBase
"""

from .bopes_sampler import BOPESSampler
from .extrapolator import (Extrapolator, DifferentialExtrapolator, PCAExtrapolator,
                           PolynomialExtrapolator, SieveExtrapolator, WindowExtrapolator)
from .potentials import (EnergySurface1DSpline, HarmonicPotential, MorsePotential,
                         EnergySurfaceBase, PotentialBase, VibronicStructureBase)

__all__ = [
    'BOPESSampler',
    'Extrapolator',
    'DifferentialExtrapolator',
    'PCAExtrapolator',
    'PolynomialExtrapolator',
    'SieveExtrapolator',
    'WindowExtrapolator',
    'EnergySurface1DSpline',
    'HarmonicPotential',
    'MorsePotential',
    'EnergySurfaceBase',
    'PotentialBase',
    'VibronicStructureBase',
]
