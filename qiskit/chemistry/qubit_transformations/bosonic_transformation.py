from qiskit.chemistry.drivers import BaseDriver
from typing import Tuple, List
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from .qubit_operator_transformation import QubitOperatorTransformation

class BosonicTransformation(QubitOperatorTransformation):

    #comment

    def __init__(self, h, basis):
        pass

    def transform(driver: BaseDriver) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        raise NotImplementedError()
        # take code from bosonic operator

    def interpret(...):
        pass
