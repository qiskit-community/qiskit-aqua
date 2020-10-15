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

""" The excited states calculation interface """

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry import FermionicOperator, BosonicOperator
from qiskit.chemistry.results import ElectronicStructureResult, VibronicStructureResult


class ExcitedStatesSolver(ABC):
    """The excited states calculation interface"""

    @abstractmethod
    def solve(self, driver: BaseDriver,
              aux_operators: Optional[Union[List[FermionicOperator],
                                            List[BosonicOperator]]] = None
              ) -> Union[ElectronicStructureResult, VibronicStructureResult]:
        """Compute the excited states energies of the molecule that was supplied via the driver.
        Args:
            driver: a chemistry driver object which defines the chemical problem that is to be
                    solved by this calculation.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.
        Returns:
            an eigenstate result
        """
        raise NotImplementedError()
