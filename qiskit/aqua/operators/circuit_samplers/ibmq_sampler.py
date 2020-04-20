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

""" Expectation Algorithm Base """

from typing import Dict, List, Optional
import logging

from qiskit.providers import BaseBackend
from qiskit.circuit import ParameterExpression
from qiskit.aqua import QuantumInstance
from ..operator_base import OperatorBase
from ..combo_operators import ListOp
from ..state_functions import StateFn, CircuitStateFn, DictStateFn
from .circuit_sampler_base import CircuitSamplerBase

logger = logging.getLogger(__name__)


class IBMQSampler(CircuitSamplerBase):
    """ A sampler for remote IBMQ backends.

    """

    def __init__(self,
                 backend: Optional[BaseBackend] = None,
                 quantum_instance: Optional[QuantumInstance] = None,
                 kwargs: Optional[Dict] = None) -> None:
        """
        Args:
            backend:
            quantum_instance:
            kwargs:
        """
        kwargs = {} if kwargs is None else kwargs
        self._qi = quantum_instance or QuantumInstance(backend=backend, **kwargs)

    def convert(self,
                operator: OperatorBase,
                params: dict = None):
        """ Accept the Operator and return the converted Operator """

        operator_dicts_replaced = operator.to_circuit_op()
        reduced_op = operator_dicts_replaced.reduce()
        op_circuits = {}

        # pylint: disable=inconsistent-return-statements
        def extract_circuitstatefns(operator):
            if isinstance(operator, CircuitStateFn):
                op_circuits[str(operator)] = operator
            elif isinstance(operator, ListOp):
                for op in operator.oplist:
                    extract_circuitstatefns(op)
            else:
                return operator

        extract_circuitstatefns(reduced_op)
        sampled_statefn_dicts = self.sample_circuits(list(op_circuits.values()))

        def replace_circuits_with_dicts(operator):
            if isinstance(operator, CircuitStateFn):
                return sampled_statefn_dicts[str(operator)]
            elif isinstance(operator, ListOp):
                return operator.traverse(replace_circuits_with_dicts)
            else:
                return operator

        return replace_circuits_with_dicts(reduced_op)

    def sample_circuits(self,
                        circuit_sfns: Optional[List[CircuitStateFn]] = None,
                        param_bindings: Optional[List[Dict[
                            ParameterExpression, List[float]]]] = None) -> Dict[int, DictStateFn]:
        """
        Args:
            circuit_sfns: The list of circuits or CircuitStateFns to sample
            param_bindings: a list of parameter dictionaries to bind to each circuit.
        Returns:
            Dict: dictionary of sampled state functions
        """
        circuits = [op_c.to_circuit(meas=True) for op_c in circuit_sfns]

        results = self._qi.execute(circuits)
        sampled_statefn_dicts = {}
        for (op_c, circuit) in zip(circuit_sfns, circuits):
            # Taking square root because we're replacing a
            # statevector representation of probabilities.
            sqrt_counts = {b: (v * op_c.coeff / self._qi._run_config.shots) ** .5
                           for (b, v) in results.get_counts(circuit).items()}
            sampled_statefn_dicts[str(op_c)] = StateFn(sqrt_counts)
        return sampled_statefn_dicts
