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

import copy
from functools import cmp_to_key
from typing import List, Union, Optional

import numpy as np
from scipy.linalg import block_diag
from qiskit.aqua import AquaError
from qiskit.aqua.operators import ListOp, CircuitOp
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.operator_globals import I, Z, Y, X, Zero
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.converters import dag_to_circuit, circuit_to_dag

from .circuit_qfi import CircuitQFI
from ..derivative_base import DerivativeBase


class OverlapBlockDiag(CircuitQFI):
    r"""Compute the block-diagonal of the Quantum Fisher Information (QFI) given a pure,
    parametrized quantum state. The blocks are given by all parameterized gates in quantum circuit
    layer.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def convert(self,
                operator: Union[CircuitOp, CircuitStateFn],
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression]]] = None
                ) -> ListOp:

        r"""
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl

        Raises:
            NotImplementedError: If ``operator`` is neither ``CircuitOp`` nor ``CircuitStateFn``.

        """
        if not isinstance(operator, (CircuitOp, CircuitStateFn)):
            raise NotImplementedError('operator must be a CircuitOp or CircuitStateFn')
        return self._block_diag_approx(operator=operator, params=params)

    def _block_diag_approx(self,
                           operator: Union[CircuitOp, CircuitStateFn],
                           params: Optional[Union[ParameterExpression, ParameterVector,
                                                  List[ParameterExpression]]] = None
                           ) -> ListOp:
        r"""
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            `ListOp[ListOp]` where the operator at position k,l corresponds to QFI_kl

        Raises:
            NotImplementedError: If a circuit is found such that one parameter controls multiple
                gates, or one gate contains multiple parameters.
            AquaError: If there are more than one parameter.

        """

        circuit = operator.primitive
        # Parition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = self._partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        block_params = [list(layer.parameters) for layer in layers]
        # Remove any parameters found which are not in params
        block_params = [[param for param in block if param in params] for block in block_params]

        # Determine the permutation needed to ensure that the final
        # operator is consistent with the ordering of the input parameters
        perm = [params.index(param) for block in block_params for param in block]

        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # Get generators
        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.

        generators = self._get_generators(params, circuit)

        blocks = []

        # Psi_i = layer_i @ layer_i-1 @ ... @ layer_0 @ Zero
        for k, psi_i in enumerate(psis):
            params = block_params[k]
            block = np.zeros((len(params), len(params))).tolist()

            # calculate all single-operator terms <psi_i|generator_i|psi_i>
            single_terms = np.zeros(len(params)).tolist()
            for i, p_i in enumerate(params):
                generator = generators[p_i]
                psi_gen_i = ~StateFn(generator) @ psi_i @ Zero
                psi_gen_i = PauliExpectation().convert(psi_gen_i)
                single_terms[i] = psi_gen_i

            def get_parameter_expression(circuit, param):
                if len(circuit._parameter_table[param]) > 1:
                    raise NotImplementedError("OverlapDiag does not yet support multiple "
                                              "gates parameterized by a single parameter. For such "
                                              "circuits use LinCombFull")
                gate = circuit._parameter_table[param][0][0]
                if len(gate.params) > 1:
                    raise AquaError("OverlapDiag cannot yet support gates with more than one "
                                    "parameter.")

                param_value = gate.params[0]
                return param_value

            # Calculate all double-operator terms <psi_i|generator_j @ generator_i|psi_i>
            # and build composite operators for each matrix entry
            for i, p_i in enumerate(params):
                generator_i = generators[p_i]
                param_expr_i = get_parameter_expression(circuit, p_i)

                for j, p_j in enumerate(params):
                    if i == j:
                        block[i][i] = ListOp([single_terms[i]], combo_fn=lambda x: 1 - x[0] ** 2)
                        if isinstance(param_expr_i, ParameterExpression) and not isinstance(
                                param_expr_i, Parameter):
                            expr_grad_i = DerivativeBase.parameter_expression_grad(
                                param_expr_i, p_i)
                            block[i][j] *= (expr_grad_i) * (expr_grad_i)
                        continue

                    generator_j = generators[p_j]
                    generator = ~generator_j @ generator_i
                    param_expr_j = get_parameter_expression(circuit, p_j)

                    psi_gen_ij = ~StateFn(generator) @ psi_i @ Zero
                    psi_gen_ij = PauliExpectation().convert(psi_gen_ij)
                    cross_term = ListOp([single_terms[i], single_terms[j]], combo_fn=np.prod)
                    block[i][j] = psi_gen_ij - cross_term

                    if isinstance(param_expr_i, ParameterExpression) and not isinstance(
                            param_expr_i, Parameter):
                        expr_grad_i = DerivativeBase.parameter_expression_grad(param_expr_i, p_i)
                        block[i][j] *= expr_grad_i
                    if isinstance(param_expr_j, ParameterExpression) and not isinstance(
                            param_expr_j, Parameter):
                        expr_grad_j = DerivativeBase.parameter_expression_grad(param_expr_j, p_j)
                        block[i][j] *= expr_grad_j

            wrapped_block = ListOp([ListOp(row) for row in block])
            blocks.append(wrapped_block)

        block_diagonal_qfi = ListOp(oplist=blocks,
                                    combo_fn=lambda x: np.real(block_diag(*x))[:, perm][perm, :])
        return block_diagonal_qfi

    @staticmethod
    def _partition_circuit(circuit):
        dag = circuit_to_dag(circuit)
        dag_layers = ([i['graph'] for i in dag.serial_layers()])
        num_qubits = circuit.num_qubits
        layers = list(
            zip(dag_layers, [{x: False for x in range(0, num_qubits)} for layer in dag_layers]))

        # initialize the ledger
        # The ledger tracks which qubits in each layer are available to have
        # gates from subsequent layers shifted backward.
        # The idea being that all parameterized gates should have
        # no descendants within their layer
        for i, (layer, ledger) in enumerate(layers):
            op_node = layer.op_nodes()[0]
            is_param = op_node.op.is_parameterized()
            qargs = op_node.qargs
            indices = [qarg.index for qarg in qargs]
            if is_param:
                for index in indices:
                    ledger[index] = True

        def apply_node_op(node, dag, back=True):
            op = copy.copy(node.op)
            qargs = copy.copy(node.qargs)
            cargs = copy.copy(node.cargs)
            condition = copy.copy(node.condition)
            if back:
                dag.apply_operation_back(op, qargs, cargs, condition)
            else:
                dag.apply_operation_front(op, qargs, cargs, condition)

        converged = False

        for _ in range(dag.depth() + 1):
            if converged:
                break

            converged = True

            for i, (layer, ledger) in enumerate(layers):
                if i == len(layers) - 1:
                    continue

                (next_layer, next_ledger) = layers[i + 1]
                for next_node in next_layer.op_nodes():
                    is_param = next_node.op.is_parameterized()
                    qargs = next_node.qargs
                    indices = [qarg.index for qarg in qargs]

                    # If the next_node can be moved back a layer without
                    # without becoming the descendant of a parameterized gate,
                    # then do it.
                    if not any([ledger[x] for x in indices]):

                        apply_node_op(next_node, layer)
                        next_layer.remove_op_node(next_node)

                        if is_param:
                            for index in indices:
                                ledger[index] = True
                                next_ledger[index] = False

                        converged = False

                # clean up empty layers left behind.
                if len(next_layer.op_nodes()) == 0:
                    layers.pop(i + 1)

        partitioned_circs = [dag_to_circuit(layer[0]) for layer in layers]
        return partitioned_circs

    @staticmethod
    def _sort_params(params):
        def compare_params(param1, param2):
            name1 = param1.name
            name2 = param2.name
            value1 = name1[name1.find("[") + 1:name1.find("]")]
            value2 = name2[name2.find("[") + 1:name2.find("]")]
            return int(value1) - int(value2)

        return sorted(params, key=cmp_to_key(compare_params), reverse=False)

    @staticmethod
    def _get_generators(params, circuit):
        dag = circuit_to_dag(circuit)
        layers = list(dag.serial_layers())

        generators = {}
        num_qubits = dag.num_qubits()

        for layer in layers:
            instr = layer['graph'].op_nodes()[0].op
            if len(instr.params) == 0:
                continue
            assert len(instr.params) == 1, "Circuit was not properly decomposed"
            param_value = instr.params[0]
            for param in params:
                if param in param_value.parameters:

                    if isinstance(instr, RYGate):
                        generator = Y
                    elif isinstance(instr, RZGate):
                        generator = Z
                    elif isinstance(instr, RXGate):
                        generator = X
                    else:
                        raise NotImplementedError

                    # Get all qubit indices in this layer where the param parameterizes
                    # an operation.
                    indices = [[q.index for q in qreg] for qreg in layer['partition']]
                    indices = [item for sublist in indices for item in sublist]

                    if len(indices) > 1:
                        raise NotImplementedError
                    index = indices[0]
                    generator = (I ^ (index)) ^ generator ^ (I ^ (num_qubits - index - 1))
                    generators[param] = generator

        return generators
