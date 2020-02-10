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

r"""
Quantum Fourier Transforms (:mod:`qiskit.aqua.components.qfts`)
===============================================================
In quantum computing, a Quantum Fourier Transform (QFT) is a linear transformation
on quantum bits, and is the quantum analogue of the discrete Fourier transform.
A QFT is a part of many quantum algorithms, such as
:class:`~qiskit.aqua.algorithms.Shor`'s algorithm for factoring and computing the discrete
logarithm, and the :class:`~qiskit.aqua.algorithms.QPE` algorithm for estimating the
eigenvalues of a unitary operator.

A QFT can be performed efficiently on a quantum computer, with a particular decomposition into a
product of simpler unitary matrices.
`It has been shown <http://csis.pace.edu/ctappert/cs837-19spring/QC-textbook.pdf>`__ how,
using a simple decomposition, the discrete Fourier transform on :math:`2^n` amplitudes can be
implemented as a quantum circuit consisting of only :math:`O(n^2)` Hadamard gates and
controlled phase shift gates, where :math:`n` is the number of qubits. This is in contrast to
the classical discrete Fourier transform, which takes :math:`O(n2^n)` gates, where in the classical
case :math:`n` is the number of bits.
`The best quantum Fourier transform algorithms currently known \
<https://pdfs.semanticscholar.org/deff/d6774d409478734db5f92011ff66bebd4a05.pdf>`__
require only :math:`O(n\log n)` gates to achieve an efficient approximation.

Most of the properties of the QFT follow from the fact that it is a unitary transformation.
This implies that, if :math:`F` is the matrix representing the QFT, then
:math:`FF^\dagger = F^{\dagger}F=I`, where :math:`F^\dagger` is the Hermitian adjoint of :math:`F`
where :math:`I` is the identity matrix. It follows that :math:`F^{-1} = F^\dagger`.

Since there is an efficient quantum circuit implementing the QFT, the circuit can be
run in reverse to perform the Inverse Quantum Fourier Transform (IQFT).
Thus, both transforms can be efficiently performed on a quantum computer.

.. currentmodule:: qiskit.aqua.components.qfts

Quantum Fourier Transform Base Class
====================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QFT

Quantum Fourier Transforms
==========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Standard
   Approximate

"""
# pylint: disable=cyclic-import

from .qft import QFT
from .standard import Standard
from .approximate import Approximate

__all__ = ['Standard', 'Approximate', 'QFT']
