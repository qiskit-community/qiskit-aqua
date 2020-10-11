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

""" Hermitian Trotter Evolution """

from typing import Optional, Union
import numpy as np
import logging

from qiskit.quantum_info import Operator
from qiskit.aqua.operators.evolutions import PauliTrotterEvolution
from qiskit.aqua.operators import MatrixOp, OperatorBase, TrotterizationBase, TrotterizationFactory
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class HermitianTrotterEvolution(PauliTrotterEvolution):
    # pylint: disable=abstract-method
    r"""
    An Evolution algorithm which decompose a Hermitian operator to Pauli basis and applies Suzuki
    Trotter expansion using PauliTrotterEvolution.
    """

    def __init__(self, trotter_mode: Optional[Union[str, TrotterizationBase]] = 'trotter',
                 reps: Optional[int] = 1):
        """
        Args:
            trotter_mode: A string ('trotter', 'suzuki', or 'qdrift') to pass to the
                TrotterizationFactory, or a TrotterizationBase, indicating how to combine
                individual Pauli evolution circuits to equal the exponentiation of the Pauli sum.
            reps: How many Trotterization repetitions to make, to improve the approximation
                accuracy.
        """
        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            self._trotter = TrotterizationFactory.build(mode=trotter_mode, reps=reps)

        super().__init__(trotter_mode, reps)

    def convert(self, operator: MatrixOp) -> OperatorBase:
        r"""
        Check if operator is Hermitian, decompose to Pauli basis and apply PauliTrotterEvolution.

        Args:
            operator: The Operator to convert.

        Returns:
            The converted operator.
        """
        def check_if_hermitian(matrix: np.ndarray):
            if not np.allclose(matrix, np.conjugate(matrix.T)):
                raise AquaError("The operator to be converted by HermitianTrotterEvolution "
                                "must be hermitian!")

        if isinstance(operator, MatrixOp):
            check_if_hermitian(operator.primitive.data)
            operator = operator.to_pauli_op().exp_i()
        elif isinstance(operator, Operator):
            check_if_hermitian(operator.data)
            operator = MatrixOp(operator).to_pauli_op().exp_i()

        return super().convert(operator)
