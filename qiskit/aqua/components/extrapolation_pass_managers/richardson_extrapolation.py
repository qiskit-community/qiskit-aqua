from typing import List
import logging
import numpy as np
from qiskit.transpiler import PassManager

logger = logging.getLogger(__name__)

class RichardsonExtrapolator:

    def __init__(self, extrapolator, pass_managers: List[PassManager], order_parameters: List, extrapolated_value):
        self.extrapolator = extrapolator
        self.pass_managers = pass_managers
        self.order_parameters = order_parameters
        self.extrapolated_value = extrapolated_value

        if not all(map(lambda pm: isinstance(pm, PassManager), self.pass_managers)):
            raise TypeError('All elements of pass_managers must be instances of PassManager')

    def extrapolate(self, values: List[float]) -> float:
        return self.extrapolator(self.order_parameters, values, self.extrapolated_value)
