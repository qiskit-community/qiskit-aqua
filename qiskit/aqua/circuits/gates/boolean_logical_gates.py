# -*- coding: utf-8 -*-

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

"""The Boolean Logical AND and OR Gates."""

import warnings
from qiskit.extensions.standard import logical_and, logical_or

warnings.warn('The qiskit.circuit.aqua.gates.boolean_logical_gates module is deprecated as of '
              '0.7.0 and will be removed no earlier than 3 months after the release. The logical '
              'gates have moved to the circuit library at qiskit.circuit.library.',
              DeprecationWarning, stacklevel=2)

__all__ = ['logical_and', 'logical_or']
