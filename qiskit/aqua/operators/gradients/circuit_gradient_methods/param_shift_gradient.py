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

"""The module to compute the state gradient with the parameter shift rule."""
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import List, Union, Optional, Tuple

import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua.operators import (One, OperatorBase, StateFn, Zero, CircuitStateFn,
                                   CircuitOp)
from qiskit.aqua.operators import SummedOp, ListOp, ComposedOp, DictStateFn, VectorStateFn
from qiskit.aqua.operators.gradients.circuit_gradient_methods.circuit_gradient_method \
    import CircuitGradientMethod
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from ..derivatives_base import DerivativeBase


class ParamShiftGradient(CircuitGradientMethod):
    """Compute the gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω with the parameter shift method."""

    def __init__(self,
                 analytic: bool = True,
                 epsilon: float = 1e-6):
        r"""
        Args:
            analytic: If True use the parameter shift rule to compute analytic gradients,
                      else use a finite difference approach
            epsilon: The offset size to use when computing finite difference gradients.
                     Ignored if analytic == True

        
        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        self._analytic = analytic
        self._epsilon = epsilon

    @property
    def analytic(self):
        return self._analytic

    @property
    def epsilon(self):
        return self._epsilon

    # pylint: disable=arguments-differ
    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter],
                                       Tuple[Parameter, Parameter],
                                       List[Tuple[Parameter, Parameter]]]] = None) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to our quantum state we are taking the
                      gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
                    If a Parameter, ParameterVector or List[Parameter] is given, then
                    the 1st oder derivative of the operator is calculated.
                    If a Tuple[Parameter, Parameter] or List[Tuple[Parameter, Parameter]]
                    is given, then the 2nd oder derivative of the operator is calculated.

        Returns:
            An operator corresponding to the gradient resp. Hessian. The order is in accordance with
            the order of the given parameters.

        """
        if isinstance(params, Parameter) or isinstance(params, ParameterVector):
            return self.parameter_shift(operator, params)
        elif isinstance(params, tuple):
            return self.parameter_shift(self.parameter_shift(operator, params[0]), params[1])
        elif isinstance(params, Iterable):
            if isinstance(params[0], Parameter):
                return self.parameter_shift(operator, params)
            elif isinstance(params[0], tuple):
                return ListOp(
                    [self.parameter_shift(self.parameter_shift(operator, pair[0]), pair[1])
                     for pair in params])
            else:
                raise AquaError('The linear combination gradient does only support the computation '
                                'of 1st gradients and 2nd order gradients.')
        else:
            raise AquaError('The linear combination gradient does only support the computation '
                            'of 1st gradients and 2nd order gradients.')

    def parameter_shift(self,
                        operator: OperatorBase,
                        params: Union[Parameter, ParameterVector, List]) -> OperatorBase:
        r"""
        Args:
            operator: the operator containing circuits we are taking the derivative of
            params: The parameters (ω) we are taking the derivative with respect to. If
                    a ParameterVector is provided, each parameter will be shifted.
        Returns:
            param_shifted_op: A ListOp of SummedOps corresponding to
                [2*(V(ω_i + π/2) - V(ω_i - π/2)) for w_i in params]
            or for analytic = False ListOp of SummedOps corresponding to
                [(V(ω_i + 1e-8) - V(ω_i - 1e-8))/2.e-8
            for w_i in params]

        Raises:
            ValueError: TODO
            TypeError: TODO
            AquaError: TODO
        """
        # pylint: disable=too-many-return-statements
        if isinstance(params, (ParameterVector, list)):
            param_grads = [self.parameter_shift(operator, param) for param in params]
            absent_params = [params[i] for i, grad_ops in enumerate(param_grads) if
                             grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params)
            return ListOp(absent_params)

        # by this point, it's only one parameter
        param = params

        if not isinstance(param, Parameter):
            raise ValueError
        if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
            return_op = operator.traverse(partial(self.parameter_shift, params=param))

            # Remove any branch of the tree where the relevant parameter does not occur
            trimmed_oplist = [op for op in return_op.oplist if op is not None]
            # If all branches are None, remove the parent too
            if len(trimmed_oplist) == 0:
                return None
            # Rebuild the operator with the trimmed down oplist
            properties = {'coeff': return_op._coeff, 'abelian': return_op._abelian}
            if return_op.__class__ == ListOp:
                properties['combo_fn'] = return_op.combo_fn
            return return_op.__class__(oplist=trimmed_oplist, **properties)

        else:
            circs = self.get_unique_circuits(operator)

            if len(circs) > 1:
                # Understand how this happens
                raise TypeError(
                    'Please define an operator with a single circuit representing '
                    'the quantum state.')
            if len(circs) == 0:
                return operator
            circ = circs[0]

            circ = ParamShiftGradient._unroll_to_supported_operations(circ)
            operator = ParamShiftGradient._replace_operator_circuit(operator, circ)

            if param not in circ._parameter_table:
                return ~Zero @ One

            shifted_ops = []
            summed_shifted_op = None
            for m, param_occurence in enumerate(circ._parameter_table[param]):
                param_index = param_occurence[1]
                pshift_op = deepcopy(operator)
                mshift_op = deepcopy(operator)

                # We need the circuit objects of the newly instantiated operators
                pshift_circ = self.get_unique_circuits(pshift_op)[0]
                mshift_circ = self.get_unique_circuits(mshift_op)[0]

                pshift_gate = pshift_circ._parameter_table[param][m][0]
                mshift_gate = mshift_circ._parameter_table[param][m][0]

                p_param = pshift_gate.params[param_index]
                m_param = mshift_gate.params[param_index]

                # Assumes the gate is a standard qiskit gate

                if self._analytic:
                    shift_constant = 0.5
                    pshift_gate.params[param_index] = (p_param + (np.pi / (4 * shift_constant)))
                    mshift_gate.params[param_index] = (m_param - (np.pi / (4 * shift_constant)))
                else:
                    shift_constant = 1. / (2 * self._epsilon)
                    pshift_gate.params[param_index] = (p_param + self._epsilon)
                    mshift_gate.params[param_index] = (m_param - self._epsilon)

                if not isinstance(operator, ComposedOp):
                    shifted_op = ListOp(
                        [pshift_op, mshift_op],
                        combo_fn=partial(self._prob_combo_fn, shift_constant=shift_constant))
                else:
                    shifted_op = shift_constant * (pshift_op - mshift_op)

                if isinstance(p_param, ParameterExpression) and not isinstance(p_param,
                                                                               Parameter):
                    expr_grad = DerivativeBase.parameter_expression_grad(p_param, param)
                    shifted_op *= expr_grad
                if not summed_shifted_op:
                    summed_shifted_op = shifted_op
                else:
                    summed_shifted_op += shifted_op

            shifted_ops.append(summed_shifted_op)

            if not SummedOp(shifted_ops).reduce():
                return ~StateFn(Zero) @ One
            else:
                return SummedOp(shifted_ops).reduce()

    @staticmethod
    def _prob_combo_fn(x, shift_constant):
        # In the probability gradient case, the amplitudes still need to be converted
        # into sampling probabilities.
        def get_primitives(item):
            if isinstance(item, DictStateFn):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data
            return item

        if not isinstance(x, Iterable):
            x = get_primitives(x)
        else:
            items = []
            for item in x:
                items.append(get_primitives(item))
        if isinstance(items[0], dict):
            prob_dict = {}
            for i, item in enumerate(items):
                for key, prob_counts in item.items():
                    if not key in prob_dict.keys():
                        prob_dict[key] = shift_constant * ((-1) ** i) * prob_counts
                    else:
                        prob_dict[key] += shift_constant * ((-1) ** i) * prob_counts
            return prob_dict
        elif isinstance(items[0], Iterable):
            try:
                return shift_constant * np.subtract(np.multiply(items[0], np.conj(items[0])),
                                                    np.multiply(items[1], np.conj(items[1])))
            except Exception:
                pass

        print('type error', x
              )
        raise TypeError(
            'Probability gradients can only be evaluated from VectorStateFs or DictStateFns.')

    @staticmethod
    def _unroll_to_supported_operations(circuit):
        supported = {'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'p', 'u', 'cx', 'cy', 'cz'}
        unique_ops = set(circuit.count_ops().keys())
        if not unique_ops.issubset(supported):
            circuit = transpile(circuit, basis_gates=list(supported), optimization_level=0)
        return circuit

    @staticmethod
    def _replace_operator_circuit(operator: OperatorBase,
                                  circuit: QuantumCircuit) -> OperatorBase:
        """Replace a circuit element in an operator with a single element given as circuit.

        Args:
            operator: Operator for which the circuit representing the quantum state shall be
                replaced.
            circuit: Circuit which shall replace the circuit in the given operator.

        Returns:
            Operator with replaced circuit quantum state function

        """
        if isinstance(operator, CircuitStateFn):
            return CircuitStateFn(circuit, coeff=operator.coeff)
        elif isinstance(operator, CircuitOp):
            return CircuitOp(circuit, coeff=operator.coeff)
        elif isinstance(operator, ComposedOp) or isinstance(operator, ListOp):
            return operator.traverse(
                partial(ParamShiftGradient._replace_operator_circuit, circuit=circuit))
        else:
            return operator

    @classmethod
    def get_unique_circuits(cls, operator: OperatorBase) -> List[QuantumCircuit]:
        """Traverse the operator and return all unique circuits

        Args:
            operator: An operator that potentially includes QuantumCircuits

        Returns:
            A list of all unique quantum circuits that appear in the operator

        """
        if isinstance(operator, CircuitStateFn):
            return [operator.primitive]

        def get_circuit(op):
            return op.primitive if isinstance(op, (CircuitStateFn, CircuitOp)) else None

        unrolled_op = cls.unroll_operator(operator)
        circuits = []
        for ops in unrolled_op:
            if not isinstance(ops, list):
                ops = [ops]
            for op in ops:
                if isinstance(op, (CircuitStateFn, CircuitOp, QuantumCircuit)):
                    c = get_circuit(op)
                    if c and c not in circuits:
                        circuits.append(c)
        return circuits

    @classmethod
    def unroll_operator(cls, operator: OperatorBase) -> Union[OperatorBase, List[OperatorBase]]:
        """TODO

        Args:
            operator: TODO

        Returns:
            TODO

        """
        if isinstance(operator, ListOp):
            return [cls.unroll_operator(op) for op in operator]
        if hasattr(operator, 'primitive') and isinstance(operator.primitive, ListOp):
            return [operator.__class__(op) for op in operator.primitive]
        return operator
