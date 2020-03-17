# -*- coding: utf-8 -*-

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

"""The deprecated UCCSD module.

Moved to qiskit/chemistry/components/ansatzes.
"""

import warnings
from qiskit.chemistry.components.ansatzes import UCCSD

warnings.warn('The module qiskit.chemistry.components.variational_forms has moved to '
              'qiskit.chemistry.components.ansatzes and is deprecated as of 0.7.0, and '
              'will be removed no earlier than 3 months after that release date. ',
              DeprecationWarning, stacklevel=2)

__all__ = ['UCCSD']
