# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" The Quantum Approximate Optimization Algorithm. """

from typing import List, Callable, Optional
import logging
import numpy as np
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.utils.validation import validate_min
from .var_form import QAOAVarForm

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.

    See https://arxiv.org/abs/1411.4028
    """

    def __init__(self, operator: BaseOperator, optimizer: Optimizer, p: int = 1,
                 initial_state: Optional[InitialState] = None,
                 mixer: Optional[BaseOperator] = None, initial_point: Optional[np.ndarray] = None,
                 max_evals_grouped: int = 1, aux_operators: Optional[List[BaseOperator]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 auto_conversion: bool = True) -> None:
        """
        Args:
            operator: Qubit operator
            optimizer: The classical optimizer to use.
            p: the integer parameter p as specified in https://arxiv.org/abs/1411.4028
            initial_state: the initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with. Allows support of
                   optimizations in constrained subspaces
                   as per https://arxiv.org/abs/1709.03489
            optimizer: the classical optimization algorithm.
            initial_point: optimizer initial point.
            max_evals_grouped: max number of evaluations to be performed simultaneously.
            aux_operators: aux operators
            callback: a callback that can access the intermediate
                                 data during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion: an automatic conversion for operator and aux_operators
                into the type which is most suitable for the backend.

                - for *non-Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.MatrixOperator`
                - for *Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.WeightedPauliOperator`
                - for *qasm simulator or real backend:*
                  :class:`~qiskit.aqua.operators.TPBGroupedWeightedPauliOperator`
        """
        validate_min('p', p, 1)
        var_form = QAOAVarForm(operator.copy(), p, initial_state=initial_state,
                               mixer_operator=mixer)
        super().__init__(operator, var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)
