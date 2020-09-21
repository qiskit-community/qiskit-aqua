from abc import ABC, abstractmethod
from qiskit.chemistry.drivers import BaseDriver
from typing import Tuple, List
from qiskit.aqua.operators.legacy import WeightedPauliOperator


class QubitOperatorTransformation(ABC):

    @abstractmethod
    def transform(driver: BaseDriver) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        raise NotImplementedError()

    #@abstractmethod
    #def interpret(value, aux_values, circuit, params=None): # -> GroundStateResult:  # might be fermionic / bosonic
    #    raise NotImplementedError()
