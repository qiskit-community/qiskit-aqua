from typing import Optional, Union
import numpy as np

from qiskit.quantum_info import Operator

from qiskit.aqua.operators.evolutions import PauliTrotterEvolution

from qiskit.aqua.operators import MatrixOp, OperatorBase, TrotterizationBase, TrotterizationFactory

from qiskit.aqua import AquaError


class HermitianTrotterEvolution(PauliTrotterEvolution):
    def __init__(self, trotter_mode: Optional[Union[str, TrotterizationBase]] = 'trotter',
                 reps: Optional[int] = 1):

        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            self._trotter = TrotterizationFactory.build(mode=trotter_mode, reps=reps)

        super().__init__(trotter_mode, reps)

    def convert(self, operator: MatrixOp) -> OperatorBase:

        def check_if_hermitian(matrix: np.ndarray):
            if not np.allclose(matrix, np.conjugate(matrix.T)):
                raise AquaError("The input matrix for HermitianTrotterEvolution must be hermitian!")

        if isinstance(operator, MatrixOp):
            check_if_hermitian(operator.primitive.data)
            operator = operator.to_pauli_op().exp_i()
        elif isinstance(operator, Operator):
            check_if_hermitian(operator.data)
            operator = MatrixOp(operator).to_pauli_op().exp_i()

        return super().convert(operator)

