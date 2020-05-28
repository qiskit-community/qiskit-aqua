# -*- coding: utf-8 -*-

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
Inverse Quantum Fourier Transforms (:mod:`qiskit.aqua.components.iqfts`)
========================================================================
In quantum computing, a Quantum Fourier Transform (QFT) is a linear transformation
on quantum bits, and is the quantum analogue of the discrete Fourier transform. Since there
is an efficient quantum circuit implementing the QFT, the circuit can be run in reverse to perform
the Inverse Quantum Fourier Transform (IQFT).

See :mod:`~qiskit.aqua.components.qfts` for more information about Quantum Fourier Transforms.

.. currentmodule:: qiskit.aqua.components.iqfts

Inverse Quantum Fourier Transform Base Class
============================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   IQFT

Inverse Quantum Fourier Transforms
==================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Standard
   Approximate

"""
# pylint: disable=cyclic-import

from .iqft import IQFT
from .standard import Standard
from .approximate import Approximate

__all__ = ['Standard', 'Approximate', 'IQFT']
