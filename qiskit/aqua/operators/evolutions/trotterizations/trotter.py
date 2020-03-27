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

"""
Simple Trotter expansion.

"""

from .suzuki import Suzuki


class Trotter(Suzuki):
    """ Simple Trotter expansion """
    def __init__(self,
                 reps: int = 1) -> None:
        super().__init__(order=1, reps=1)
