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

from abc import ABC
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator)
from qiskit.chemistry.core import MolecularExcitedStatesResult


class ExcitedStatesCalculation(ABC):

    def __init__(self,
                 ground_state_calculation : GroundStateCalculation)-> None:
    
                 #transformation: TransformationType = TransformationType.FULL,
                 #qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 #two_qubit_reduction: bool = True,
                 #freeze_core: bool = False,
                 #orbital_reduction: Optional[List[int]] = None,
                 #z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:

        self._ground_state_calc = GroundStateCalculation


        #.__init__(transformation, qubit_mapping, two_qubit_reduction, freeze_core, orbital_reduction,
        #                 z2symmetry_reduction)

    def get_qubit_ops(self, driver):

        self._ground_state_calc._transform(driver)
        
    @abstractmethod
    def compute_excited_states(self,
                               driver: BaseDriver) -> MolecularExcitedStatesResult:
        raise NotImplementedError()
