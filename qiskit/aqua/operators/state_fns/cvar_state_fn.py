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


from typing import Union, Set, Dict, Optional, cast, Callable, Tuple 
import numpy as np

from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from .operator_state_fn import OperatorStateFn
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp


# pylint: disable=invalid-name

class CVarStateFn(OperatorStateFn):
    r"""
    A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 alpha: float = 1,
                 is_measurement: bool = True) -> None:
        """
        Args:
            primitive: The ``OperatorBase`` which defines the behavior of the underlying State
                function.
            coeff: A coefficient by which to multiply the state function
            is_measurement: Whether the StateFn is a measurement operator
        """
        if primitive is None:
            raise ValueError
        if not is_measurement:
            raise ValueError("CostFnMeasurement is only defined as a measurement")

        self._alpha = alpha

        super().__init__(primitive, coeff=coeff, is_measurement=True)

    @property
    def alpha(self):
        return self._alpha

    def primitive_strings(self) -> Set[str]:
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        if hasattr(self.primitive, 'num_qubits'):
            return self.primitive.num_qubits
        else:
            return None

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        return ValueError("Adjoint of a cost function not defined")

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:

        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              coeff=self.coeff * scalar,
                              is_measurement=self.is_measurement,
                              alpha = self._alpha)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        if self.alpha == 1:
            return OperatorStateFn(self.primitive.to_matrix_op(massive=massive) * self.coeff,
                                   is_measurement=self.is_measurement)
        else: 
            raise NotImplementedError

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        r"""
        Not Implemented
        """
        raise NotImplementedError

    def to_circuit_op(self) -> OperatorBase:
        r""" Return ``StateFnCircuit`` corresponding to this StateFn. Ignore for now because this is
        undefined. TODO maybe call to_pauli_op and diagonalize here, but that could be very
        inefficient, e.g. splitting one Stabilizer measurement into hundreds of 1 qubit Paulis."""
        raise NotImplementedError

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('CVarMeasurement', prim_str)
        else:
            return "{}({}) * {}".format(
                'CVarMeasurement',
                prim_str,
                self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:

        from .state_fn import StateFn
        from .dict_state_fn import DictStateFn
        from .vector_state_fn import VectorStateFn
        from .circuit_state_fn import CircuitStateFn
        from ..list_ops.list_op import ListOp
        from ..primitive_ops.circuit_op import CircuitOp

        if isinstance(front, CircuitStateFn):
            front = front.eval()

        #Standardize the inputs to a dict  
        if isinstance(front, DictStateFn):
            data = front.primitive
        elif isinstance(front, VectorStateFn):
            vec = front.primitive.data
            #Determine how many bits are needed 
            key_len = int(np.ceil(np.log2(len(vec))))
            data = {format(index, '0'+str(key_len)+'b'): val for index,val in enumerate(vec)}
        else:
            raise ValueError("Unexpected Input to CVarStateFn: ", type(front))

        assert isinstance(data, Dict)

        obs = self.primitive

        #Handle probability gradients
        outcomes = list(data.items())

        #Sort based on energy evaluation
        for i,outcome in enumerate(outcomes):
            key = outcomes[i][0]
            outcomes[i] += (obs.eval(key).adjoint().eval(key),)

        outcomes = sorted(outcomes, key=lambda x: x[2])

        #outcomes = outcomes
        # states: bitstrings, bi
        # P: sampling probabilities
        # H: <bi|H|bi>
        states, P, H = zip(*outcomes)

        #Square the dict values so they represent sampling probabilities
        # (since CircuitSampler takes the root...)
        P = [np.real(p*np.conj(p)) for p in P]

        # Determine the index of the measurement outcome for which some shots will
        # be discarded. 
        alpha = self.alpha
        running_total = 0
        j=0 
        for i,pi in enumerate(P):
            if isinstance(outcome[1], Tuple):
                p = pi[0]
            else:
                p = pi
            running_total += p
            j = i
            if running_total > alpha:
                break

        #handle the case j<1
        Hj = H[j]
        H = H[:j]
        P = P[:j]

        if alpha == 0:
            return Hj

        CVar = alpha * Hj

        #CVar = alpha*Hj + \sum_i P[i]*(H[i] - Hj)
        for i in range(len(H)):
            if isinstance(P[i], Tuple):
                CVar += P[i][0]*(H[i]-Hj)
            else:
                CVar += P[i]*(H[i]-Hj)

        return CVar/alpha

    def compose(self, other: OperatorBase) -> OperatorBase:
        r"""
        Composition (Linear algebra-style: A@B(x) = A(B(x))) is not well defined for states
        in the binary function model, but is well defined for measurements.

        Args:
            other: The Operator to compose with self.

        Returns:
            An Operator equivalent to the function composition of self and other.

        Raises:
            ValueError: If self is not a measurement, it cannot be composed from the right.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        if not self.is_measurement:
            raise ValueError(
                'Composition with a Statefunction in the first operand is not defined.')

        new_self = self
        if self.num_qubits is not None:
            new_self, other = self._check_zero_for_composition_and_expand(other)
            # TODO maybe include some reduction here in the subclasses - vector and Op, op and Op, etc.
            # pylint: disable=import-outside-toplevel
            from qiskit.aqua.operators import CircuitOp

            if self.primitive == {'0' * self.num_qubits: 1.0} and isinstance(other, CircuitOp):
                # Returning CircuitStateFn
                raise NotImplementedError("understand what practical scenarios cause this to happen.")
                #return StateFn(other.primitive, is_measurement=self.is_measurement,
                #               coeff=self.coeff * other.coeff)

        from qiskit.aqua.operators import ComposedOp
        from copy import deepcopy as dc
        def tuple_grad_combo_fn(self, params, method='param_shift'):
            
            assert self.coeff == 1.0, "Unexpected coefficient on specialized ListOp"
            
            #Assume for now params is a single parameter
            param = params
            
            if len(self.oplist) == 1:
                op = ListOp([dc(self.oplist[0])], 
                           combo_fn=lambda x: x[0],
                           grad_combo_fn=tuple_grad_combo_fn)
                d_op = ListOp([dc(self.oplist[0]).autograd(param, method)], 
                           combo_fn=lambda x: x[0],
                           grad_combo_fn=tuple_grad_combo_fn)
                
                tuple_op = ListOp([dc(op), dc(d_op)],
                                   combo_fn=lambda x: (x[0],x[1]),
                                   grad_combo_fn=self.grad_combo_fn)
                return tuple_op   
            
            elif len(self.oplist) == 2:
                tuple_ops = []
                try:
                    d_op_0 = dc(self.oplist[0].autograd(param, method, replace_autograd=True))
                    d_op_1 = dc(self.oplist[1].autograd(param, method, replace_autograd=True))
                except:
                    d_op_0 = dc(self.oplist[0].autograd(param, method))
                    d_op_1 = dc(self.oplist[1].autograd(param, method))

                final = ListOp([dc(d_op_0), 
                               dc(d_op_1)],
                               combo_fn=lambda x: x[0]+x[1],
                               grad_combo_fn=self.grad_combo_fn)
                
                return final
                
            else:
                raise ValueError("Unexpected number of operators ({n}) stored in oplist")

                other = ListOp([other], 
                               combo_fn=lambda x: x[0],
                               grad_combo_fn=tuple_grad_combo_fn)

        return ComposedOp([new_self, other])
    
#Todo: implement this logic...
#d/dx CVar = alpha*Hj + \sum_i  d/dx P[i] * (H[i] - Hj) + P[i] * d/dx (H[i] - Hj)



    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        raise NotImplementedError
