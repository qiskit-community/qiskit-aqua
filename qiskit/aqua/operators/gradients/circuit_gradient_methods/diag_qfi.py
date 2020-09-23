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

"""The module for Quantum the Fisher Information."""
from typing import List, Union, Optional

import numpy as np
from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, ComposedOp, OperatorStateFn
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.gradients.circuit_gradient_methods import CircuitGradientMethod
from qiskit.aqua.operators.operator_globals import Zero
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import (Parameter, ParameterVector, ParameterExpression)

from .block_diag_qfi import BlockDiagQFI


class DiagQFI(CircuitGradientMethod):
    r"""Compute the Quantum Fisher Information (QFI) given a pure, parametrized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def convert(self,
                operator: CircuitStateFn,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
                ) -> ListOp(List[OperatorBase]):
        r"""
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl

        Raises:
            ValueError: If the value for ``approx`` is not supported.
        """
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)

        return self._qfi_states(cleaned_op, params)

    def _qfi_states(self,
                    operator: Union[CircuitOp, CircuitStateFn],
                    params: Union[Parameter, ParameterVector, List] = None
                    ) -> OperatorBase:
        """
        TODO
        Args:
            self:
            operator:
            params:

        Returns:

        """

        if not isinstance(operator, (CircuitOp, CircuitStateFn)):
            raise NotImplementedError

        circuit = operator.primitive

        # Parition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = BlockDiagQFI._partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.
        generators = BlockDiagQFI._get_generators(params, circuit)

        diag = []
        for param in params:
            if len(circuit._parameter_table[param]) > 1:
                raise NotImplementedError("The QFI Approximations do not yet support multiple "
                                          "gates parameterized by a single parameter. For such "
                                          "circuits set approx = None")

            gate = circuit._parameter_table[param][0][0]

            assert len(gate.params) == 1, "Circuit was not properly decomposed"

            param_value = gate.params[0]
            generator = generators[param]
            meas_op = ~StateFn(generator)

            # get appropriate psi_i
            psi = [(psi) for psi in psis if param in psi.primitive.parameters][0]

            op = meas_op @ psi @ Zero
            if isinstance(param_value, ParameterExpression) and not isinstance(param_value,
                                                                               Parameter):
                expr_grad = self._parameter_expression_grad(param_value, param)
                op *= expr_grad
            rotated_op = PauliExpectation().convert(op)
            diag.append(rotated_op)

        grad_op = ListOp(diag, combo_fn=lambda x: np.diag(np.real([1 - y ** 2 for y in x])))
        return grad_op

    @classmethod
    def _factor_coeffs_out_of_composed_op(cls, operator: OperatorBase) -> OperatorBase:
        """TODO

        Args:
            operator: TODO

        Returns:
            TODO

        Raises:
            ValueError: TODO
        """
        if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
            return operator.traverse(cls._factor_coeffs_out_of_composed_op)
        if isinstance(operator, ComposedOp):
            total_coeff = 1.0
            take_norm_of_coeffs = False
            for op in operator.oplist:

                if take_norm_of_coeffs:
                    total_coeff *= (op._coeff * np.conj(op._coeff))
                else:
                    total_coeff *= op._coeff
                if hasattr(op, 'primitive'):
                    prim = op.primitive
                    if isinstance(prim, ListOp):
                        raise ValueError("This operator was not properly decomposed. "
                                         "By this point, all operator measurements should "
                                         "contain single operators, otherwise the coefficient "
                                         "gradients will not be handled properly.")
                    if hasattr(prim, 'coeff'):
                        if take_norm_of_coeffs:
                            total_coeff *= (prim._coeff * np.conj(prim._coeff))
                        else:
                            total_coeff *= prim._coeff

                if isinstance(op, OperatorStateFn) and op._is_measurement:
                    take_norm_of_coeffs = True
            return total_coeff * cls._erase_operator_coeffs(operator)

        else:
            return operator
