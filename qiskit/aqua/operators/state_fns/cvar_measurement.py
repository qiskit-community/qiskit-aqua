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

""" CVarStateFn Class """


from typing import Union, Optional, Callable
import numpy as np

from qiskit.aqua.aqua_error import AquaError
from qiskit.circuit import ParameterExpression, QuantumCircuit, Instruction
from qiskit.result import Result
from qiskit.quantum_info import Statevector

from ..operator_base import OperatorBase
from ..primitive_ops import MatrixOp, PauliOp
from ..list_ops import ListOp, SummedOp
from .state_fn import StateFn
from .operator_state_fn import OperatorStateFn


# pylint: disable=invalid-name

class CVaRMeasurement(OperatorStateFn):
    r"""
    A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """

    @staticmethod
    def __new__(cls,
                primitive: Union[str, dict, Result,
                                 list, np.ndarray, Statevector,
                                 QuantumCircuit, Instruction,
                                 OperatorBase] = None,
                alpha: float = 1.0,
                coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                ) -> 'StateFn':
        obj = object.__new__(cls)
        obj.__init__(primitive, alpha, coeff)
        return obj

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 alpha: float = 1.0,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
        Args:
            primitive: The ``OperatorBase`` which defines the behavior of the underlying State
                function.
            coeff: A coefficient by which to multiply the state function
            alpha: TODO

        Raises:
            ValueError: TODO remove that this raises an error
            ValueError: If alpha is not in [0, 1].
            AquaError: If the primitive is not diagonal.
        """
        if primitive is None:
            raise ValueError

        if not 0 <= alpha <= 1:
            raise ValueError('The parameter alpha must be in [0, 1].')
        self._alpha = alpha

        if not _check_is_diagonal(primitive):
            raise AquaError('Input operator to CVar must be diagonal, but is not:', str(primitive))

        super().__init__(primitive, coeff=coeff, is_measurement=True)

    @property
    def alpha(self) -> float:
        """TODO

        Returns:
            TODO
        """
        return self._alpha

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        """The adjoint of a CVaRMeasurement is not defined.

        Returns:
            Does not return anything, raises an error.

        Raises:
            AquaError: The adjoint of a CVaRMeasurement is not defined.
        """
        raise AquaError('Adjoint of a CVaR measurement not defined')

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              coeff=self.coeff * scalar,
                              alpha=self._alpha)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """Not defined."""
        raise NotImplementedError

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """Not defined."""
        raise NotImplementedError

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """Not defined."""
        raise NotImplementedError

    def to_circuit_op(self) -> OperatorBase:
        """Not defined."""
        raise NotImplementedError

    def __str__(self) -> str:
        return 'CVaRMeasurement({}) * {}'.format(str(self.primitive), self.coeff)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:

        from .dict_state_fn import DictStateFn
        from .vector_state_fn import VectorStateFn
        from .circuit_state_fn import CircuitStateFn

        if isinstance(front, CircuitStateFn):
            front = front.eval()

        # Standardize the inputs to a dict
        if isinstance(front, DictStateFn):
            data = front.primitive
        elif isinstance(front, VectorStateFn):
            vec = front.primitive.data
            # Determine how many bits are needed
            key_len = int(np.ceil(np.log2(len(vec))))
            data = {format(index, '0'+str(key_len)+'b'): val for index, val in enumerate(vec)}
        else:
            raise ValueError('Unsupported input to CVaRMeasurement.eval:', type(front))

        obs = self.primitive
        alpha = self._alpha

        outcomes = list(data.items())
        # add energy evaluation
        for i, outcome in enumerate(outcomes):
            key = outcome[0]
            outcomes[i] += (obs.eval(key).adjoint().eval(key),)

        # Sort each observation based on it's energy
        outcomes = sorted(outcomes, key=lambda x: x[2])

        # Here P are the probabilities of observing each state.
        # H are the expectation values of each state with the
        # provided Hamiltonian
        _, probabilities, energies = zip(*outcomes)

        # Square the dict values
        # (since CircuitSampler takes the root...)
        probabilities = [p_i * np.conj(p_i) for p_i in probabilities]

        # Determine j, the index of the measurement outcome
        # which will be only partially included in the CVar sum
        j = 0
        running_total = 0
        for i, p_i in enumerate(probabilities):
            running_total += p_i
            j = i
            if running_total > alpha:
                break

        h_j = energies[j]
        cvar = alpha * h_j

        if alpha == 0 or j == 0:
            return h_j

        energies = energies[:j]
        probabilities = probabilities[:j]

        # CVar = alpha*Hj + \sum_i P[i]*(H[i] - Hj)
        for h_i, p_i in zip(energies, probabilities):
            cvar += p_i * (h_i - h_j)

        return cvar/alpha

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = None
                 ) -> OperatorBase:
        r"""
        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in
        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.

        Args:
            convert_fn: The function to apply to the internal OperatorBase.
            coeff: A coefficient to multiply by after applying convert_fn.
                If it is None, self.coeff is used instead.

        Returns:
            The converted StateFn.
        """
        if coeff is None:
            coeff = self.coeff

        if isinstance(self.primitive, OperatorBase):
            return self.__class__(convert_fn(self.primitive), coeff=coeff, alpha=self._alpha)
        return self

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        raise NotImplementedError


def _check_is_diagonal(operator: OperatorBase) -> bool:
    """Check whether ``operator`` is diagonal.

    Args:
        operator: The operator to check for diagonality.

    Returns:
        True, if the operator is diagonal, False otherwise.

    Raises:
        AquaError: If the operator is not diagonal.
    """
    if isinstance(operator, PauliOp):
        # every X component must be False
        if not np.any(operator.primitive.x):
            return True
        return False

    if isinstance(operator, SummedOp) and operator.primitive_strings == {'Pauli'}:
        # cover the case of sums of diagonal paulis, but don't raise since there might be summands
        # cancelling the non-diagonal parts
        if np.all(not np.any(op.primitive.x) for op in operator.oplist):
            return True

    if isinstance(operator, ListOp):
        return np.all(operator.traverse(_check_is_diagonal))

    # cannot efficiently check if a operator is diagonal, converting to matrix
    operator = operator.to_matrix_op()

    if isinstance(operator, MatrixOp):
        matrix = operator.primitive.data
        if np.all(matrix == np.diag(np.diagonal(matrix))):
            return True
        return False
    else:
        raise AquaError('Could not convert to MatrixOp, something went wrong.', operator)
