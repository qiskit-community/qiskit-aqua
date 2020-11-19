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

""" SummedPauliOp Class """

import logging
from typing import Dict, List, Optional, Set, Union, cast

import numpy as np
from scipy.sparse import spmatrix

from qiskit.circuit import Instruction, ParameterExpression
from qiskit.quantum_info import Pauli, SparsePauliOp

from ... import AquaError
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from .primitive_op import PrimitiveOp

logger = logging.getLogger(__name__)


class SummedPauliOp(PrimitiveOp):
    """Class for Operators backend by Terra's ``SparsePauliOp`` class."""

    def __init__(
            self,
            primitive: SparsePauliOp,
            coeff: Union[int, float, complex, ParameterExpression] = 1.0,
    ) -> None:
        """
        Args:
            primitive: The SparsePauliOp which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive.

        Raises:
            TypeError: invalid parameters.
        """
        if not isinstance(primitive, SparsePauliOp):
            raise TypeError(
                "SummedPauliOp can only be instantiated with SparsePauliOp, "
                f"not {type(primitive)}"
            )

        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {"SparsePauliOp"}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits  # type: ignore

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over operators with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        if isinstance(other, SummedPauliOp):
            return SummedPauliOp(
                self.coeff * self.primitive + other.coeff * other.primitive, coeff=1  # type: ignore
            )

        from .pauli_op import PauliOp

        if isinstance(other, PauliOp):
            other = other.to_summed_pauli_op()
            return SummedPauliOp(
                self.coeff * self.primitive + other.coeff * other.primitive, coeff=1  # type: ignore
            )

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return SummedPauliOp(
            self.primitive.conjugate(), coeff=self.coeff.conjugate()  # type:ignore
        )

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, SummedPauliOp):
            return False

        if isinstance(self.coeff, ParameterExpression) or isinstance(
                other.coeff, ParameterExpression
        ):
            return (
                self.coeff == other.coeff
                and self.primitive.simplify()  # type:ignore
                == other.primitive.simplify()  # type:ignore
            )
        simple_self = (self.coeff * self.primitive).simplify()  # type:ignore
        simple_other = (other.coeff * other.primitive).simplify()  # type:ignore
        return len(simple_self) == len(simple_other) and simple_self == simple_other

    def _expand_dim(self, num_qubits: int) -> "SummedPauliOp":
        return SummedPauliOp(
            self.primitive.tensor(  # type:ignore
                SparsePauliOp(Pauli(label="I" * num_qubits))
            ),
            coeff=self.coeff,
        )

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, SummedPauliOp):
            return SummedPauliOp(
                self.primitive.tensor(other.primitive),  # type:ignore
                coeff=self.coeff * other.coeff,
            )

        return TensoredOp([self, other])

    def permute(self, permutation: List[int]) -> "SummedPauliOp":
        """Permutes the sequence of ``SummedPauliOp``.

        Args:
            permutation: A list defining where each Pauli should be permuted. The Pauli at index
                j of the primitive should be permuted to position permutation[j].

        Returns:
              A new SummedPauliOp representing the permuted operator. For operator (X ^ Y ^ Z) and
              indices=[1,2,4], it returns (X ^ I ^ Y ^ Z ^ I).

        Raises:
            AquaError: if indices do not define a new index for each qubit.
        """
        if len(permutation) != self.num_qubits:
            raise AquaError(
                "List of indices to permute must have the same size as Pauli Operator"
            )
        length = max(permutation) + 1
        spop = self.primitive.tensor(  # type:ignore
            SparsePauliOp(Pauli(label="I" * (length - self.num_qubits)))
        )
        permutation = [i for i in range(length) if i not in permutation] + permutation
        permutation = np.arange(length)[np.argsort(permutation)]
        permutation = np.hstack([permutation, permutation + length])  # type: ignore
        spop.table.array = spop.table.array[:, permutation]
        return SummedPauliOp(spop, self.coeff)

    def compose(
            self,
            other: OperatorBase,
            permutation: Optional[List[int]] = None,
            front: bool = False,
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(SummedPauliOp, new_self)

        if front:
            return other.compose(new_self)
        # If self is identity, just return other.
        if not np.any(new_self.primitive.table.array):  # type: ignore
            return other * new_self.coeff * sum(new_self.coeffs)  # type: ignore

        # Both SummedPauliOps
        if isinstance(other, SummedPauliOp):
            return SummedPauliOp(
                new_self.primitive * other.primitive,  # type:ignore
                coeff=new_self.coeff * other.coeff,
            )
        # TODO: implement compose with PauliOp

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from .circuit_op import CircuitOp

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            return new_self.to_circuit_op().compose(other)

        return super(SummedPauliOp, new_self).compose(other)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", True, self.num_qubits, massive)
        return self.primitive.to_matrix() * self.coeff  # type: ignore

    def __str__(self) -> str:
        prim_list = self.primitive.to_list()  # type:ignore

        def format_number(x):
            return x.real if np.isreal(x) else x

        coeff_string = "" if self.coeff == 1 else f" * {self.coeff}"

        return (
            "SummedPauliOp([\n"
            + "".join([f"{format_number(c)} * {p},\n" for p, c in prim_list])
            + "])"
            + coeff_string
        )

    def eval(
            self,
            front: Optional[
                Union[str, Dict[str, complex], np.ndarray, OperatorBase]
            ] = None,
    ) -> Union[OperatorBase, float, complex]:
        if front is None:
            return self.to_matrix_op()

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ..list_ops.list_op import ListOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.state_fn import StateFn
        from .circuit_op import CircuitOp
        from .pauli_op import PauliOp

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [
                    self.eval(front.coeff * front_elem)  # type: ignore
                    for front_elem in front.oplist
                ]
            )

        else:

            if self.num_qubits != front.num_qubits:
                raise ValueError(
                    "eval does not support operands with differing numbers of qubits, "
                    "{} and {}, respectively.".format(self.num_qubits, front.num_qubits)
                )

            if isinstance(front, DictStateFn):

                new_dict = {}  # type: Dict
                corrected_x_bits = self.primitive.table.X[::-1]  # type: ignore
                corrected_z_bits = self.primitive.table.Z[::-1]  # type: ignore
                coeffs = self.primitive.coeffs  # type:ignore

                for bstr, v in front.primitive.items():
                    bitstr = np.asarray(list(bstr)).astype(np.int).astype(np.bool)
                    new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                    new_str = ["".join(map(str, 1 * bs)) for bs in new_b_str]
                    z_factor = np.product(
                        1 - 2 * np.logical_and(bitstr, corrected_z_bits), axis=1
                    )
                    y_factor = np.product(
                        np.sqrt(
                            1
                            - 2 * np.logical_and(corrected_x_bits, corrected_z_bits)
                            + 0j
                        ),
                        axis=1,
                    )
                    for i, n_str in enumerate(new_str):
                        new_dict[n_str] = (
                            v * z_factor[i] * y_factor[i] * coeffs[i]
                        ) + new_dict.get(n_str, 0)
                    return DictStateFn(new_dict, coeff=self.coeff * front.coeff)

            elif isinstance(front, StateFn) and front.is_measurement:
                raise ValueError("Operator composed with a measurement is undefined.")

            # Composable types with PauliOp
            elif isinstance(front, (SummedPauliOp, PauliOp, CircuitOp, CircuitStateFn)):
                return self.compose(front)

            # Covers VectorStateFn and OperatorStateFn
            elif isinstance(front, OperatorBase):
                return self.to_matrix_op().eval(front.to_matrix_op())  # type: ignore
        return None

    def exp_i(self) -> OperatorBase:
        """ Return a ``CircuitOp`` equivalent to e^-iH for this operator H. """
        # TODO: optimize for some special cases
        from ..evolutions.evolved_op import EvolvedOp

        return EvolvedOp(self)

    def to_instruction(self) -> Instruction:
        return self.to_matrix_op().to_circuit().to_instruction()  # type: ignore

    def to_pauli_op(self, massive: bool = False) -> OperatorBase:
        from .pauli_op import PauliOp

        def to_real(x):
            return x.real if np.isreal(x) else x

        def to_native(x):
            return x.item() if isinstance(x, np.generic) else x

        if len(self.primitive) == 1:
            return PauliOp(
                Pauli(x=self.primitive.table.X[0], z=self.primitive.table.Z[0]),  # type: ignore
                to_native(to_real(self.primitive.coeffs[0])) * self.coeff,  # type: ignore
            )
        return SummedOp(
            [
                PauliOp(
                    Pauli(x=s.table.X[0], z=s.table.Z[0]),
                    to_native(to_real(s.coeffs[0])),
                )
                for s in self.primitive
            ],
            coeff=self.coeff,
        )

    def __getitem__(self, offset: Union[int, slice]) -> "SummedPauliOp":
        """Allows array-indexing style access to the ``SummedPauliOp``.

        Args:
            offset: The index of ``SummedPauliOp``.

        Returns:
            The ``SummedPauliOp`` at index ``offset``,
        """
        return SummedPauliOp(self.primitive[offset], self.coeff)

    def __len__(self) -> int:
        """Length of ``SparsePauliOp``.

        Returns:
            An int equal to the length of SparsePauliOp.
        """
        return len(self.primitive)


    def reduce(
            self, atol: Optional[float] = None, rtol: Optional[float] = None
    ) -> "SummedPauliOp":
        """Simplify the primitive ``SparsePauliOp``.

        Args:
            atol: Absolute tolerance for checking if coefficients are zero (Default: 1e-8).
            rtol: Relative tolerance for checking if coefficients are zero (Default: 1e-5).

        Returns:
            The simplified ``SummedPauliOp``.
        """
        return SummedPauliOp(self.primitive.simplify(atol=atol, rtol=rtol), self.coeff)  # type: ignore

    def to_spmatrix(self) -> spmatrix:
        """Returns SciPy sparse matrix representation of the ``SummedPauliOp``.

        Returns:
            CSR sparse matrix representation of the ``SummedPauliOp``.

        Raises:
            ValueError: invalid parameters.
        """
        return self.primitive.to_matrix(sparse=True) * self.coeff  # type: ignore

    @property
    def oplist(self) -> List[OperatorBase]:
        """The list of operators.
        This function is for compatibility with ``SummedOp``.

        Return:
            The list of operators.
        """
        return self.to_pauli_op().oplist  # type: ignore

