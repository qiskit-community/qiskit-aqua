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
Chemistry Results (:mod:`qiskit.chemistry.results`)
===================================================
.. currentmodule:: qiskit.chemistry.results

Qiskit's chemistry results for ground and excited states, both Fermionic and Bosonic. Algorithms
may extend these to provide algorithm specific aspects in their result.

Results
=======

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EigenstateResult
   BOPESSamplerResult
   ElectronicStructureResult
   VibronicStructureResult

"""

from .bopes_sampler_result import BOPESSamplerResult
from .eigenstate_result import EigenstateResult
from .electronic_structure_result import DipoleTuple, ElectronicStructureResult
from .vibronic_structure_result import VibronicStructureResult

__all__ = [
    'BOPESSamplerResult',
    'DipoleTuple',
    'EigenstateResult',
    'ElectronicStructureResult',
    'VibronicStructureResult',
]
