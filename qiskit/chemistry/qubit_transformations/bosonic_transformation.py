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

"""TODO"""

from typing import Tuple, List

from qiskit.chemistry.drivers import BaseDriver
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from .qubit_operator_transformation import QubitOperatorTransformation


class BosonicTransformation(QubitOperatorTransformation):
    """TODO"""

    def __init__(self, h, basis):
        pass

    def transform(self, driver: BaseDriver
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """TODO"""
        raise NotImplementedError()
        # take code from bosonic operator

    def interpret(self):
        """TODO"""
        pass
