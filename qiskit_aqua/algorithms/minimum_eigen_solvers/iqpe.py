# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Iterative Quantum Phase Estimation Algorithm.

See https://arxiv.org/abs/quant-ph/0610214
"""

from typing import Optional, List, Dict, Union, Any
import logging
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import (WeightedPauliOperator, suzuki_expansion_slice_pauli_list,
                                   evolution_instruction)
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.operators import LegacyBaseOperator, OperatorBase
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from .qpe import QPEResult

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class IQPE(QuantumAlgorithm, MinimumEigensolver):
    """The Iterative Quantum Phase Estimation algorithm.

    IQPE, as its name suggests, iteratively computes the phase so as to require fewer qubits.
    It takes has the same set of parameters as :class:`QPE`, except for the number of
    ancillary qubits *num_ancillae*, being replaced by *num_iterations* and that
    an Inverse Quantum Fourier Transform (IQFT) is not used for IQPE.

    **Reference:**

    [1]: Dobsicek et al. (2006), Arbitrary accuracy iterative phase estimation algorithm as a two
       qubit benchmark, `arxiv/quant-ph/0610214 <https://arxiv.org/abs/quant-ph/0610214>`_
    """

    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 state_in: Optional[InitialState] = None,
                 num_time_slices: int = 1,
                 num_iterations: int = 1,
                 expansion_mode: str = 'suzuki',
                 expansion_order: int = 2,
                 shallow_circuit_concat: bool = False,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """

        Args:
            operator: The hamiltonian Operator
            state_in: An InitialState component representing an initial quantum state.
            num_time_slices: The number of time slices, has a minimum value of 1.
            num_iterations: The number of iterations, has a minimum value of 1.
            expansion_mode: The expansion mode ('trotter'|'suzuki')
            expansion_order: The suzuki expansion order, has a min. value of 1.
            shallow_circuit_concat: Set True to use shallow (cheap) mode for circuit concatenation
                of evolution slices. By default this is False.
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('num_time_slices', num_time_slices, 1)
        validate_min('num_iterations', num_iterations, 1)
        validate_in_set('expansion_mode', expansion_mode, {'trotter', 'suzuki'})
        validate_min('expansion_order', expansion_order, 1)
        super().__init__(quantum_instance)
        self._state_in = state_in
        self._num_time_slices = num_time_slices
        self._num_iterations = num_iterations
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._state_register = None
        self._ancillary_register = None
        self._ancilla_phase_coef = None
        self._in_operator = operator
        self._operator = None  # type: Optional[WeightedPauliOperator]
        self._ret = {}  # type: Dict[str, Any]
        self._pauli_list = None  # type: Optional[List[List[Union[complex, Pauli]]]]
        self._phase_estimation_circuit = None
        self._slice_pauli_list = None  # type: Optional[List[List[Union[complex, Pauli]]]]
        self._setup(operator)

    def _setup(self, operator: Optional[Union[OperatorBase, LegacyBaseOperator]]) -> None:
        self._operator = None
        self._ret = {}
        self._pauli_list = None
        self._phase_estimation_circuit = None
        self._slice_pauli_list = None
        if operator:
            # Convert to Legacy Operator if Operator flow passed in
            if isinstance(operator, OperatorBase):
                operator = operator.to_legacy_op()
            self._operator = op_converter.to_weighted_pauli_operator(operator.copy())
            self._ret['translation'] = sum([abs(p[0]) for p in self._operator.reorder_paulis()])
            self._ret['stretch'] = 0.5 / self._ret['translation']

            # translate the operator
            self._operator.simplify()
            translation_op = WeightedPauliOperator([
                [
                    self._ret['translation'],
                    Pauli(
                        np.zeros(self._operator.num_qubits),
                        np.zeros(self._operator.num_qubits)
                    )
                ]
            ])
            translation_op.simplify()
            self._operator += translation_op
            self._pauli_list = self._operator.reorder_paulis()

            # stretch the operator
            for p in self._pauli_list:
                p[0] = p[0] * self._ret['stretch']

            if len(self._pauli_list) == 1:
                slice_pauli_list = self._pauli_list
            else:
                if self._expansion_mode == 'trotter':
                    slice_pauli_list = self._pauli_list
                else:
                    slice_pauli_list = suzuki_expansion_slice_pauli_list(self._pauli_list,
                                                                         1, self._expansion_order)
            self._slice_pauli_list = slice_pauli_list

    @property
    def operator(self) -> Optional[Union[OperatorBase, LegacyBaseOperator]]:
        """ Returns operator """
        return self._in_operator

    @operator.setter
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """ set operator """
        self._in_operator = operator
        self._setup(operator)

    @property
    def aux_operators(self) -> Optional[List[Union[OperatorBase, LegacyBaseOperator]]]:
        """ Returns aux operators """
        raise TypeError('aux_operators not supported.')

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[List[Union[OperatorBase, LegacyBaseOperator]]]
                      ) -> None:
        """ Set aux operators """
        raise TypeError('aux_operators not supported.')

    def construct_circuit(self,
                          k: Optional[int] = None,
                          omega: float = 0,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the kth iteration Quantum Phase Estimation circuit.

        For details of parameters, please see Fig. 2 in https://arxiv.org/pdf/quant-ph/0610214.pdf.

        Args:
            k: the iteration idx.
            omega: the feedback angle.
            measurement: Boolean flag to indicate if measurement should
                    be included in the circuit.

        Returns:
            QuantumCircuit: the quantum circuit per iteration
        """
        if self._operator is None or self._state_in is None:
            return None

        k = self._num_iterations if k is None else k
        a = QuantumRegister(1, name='a')
        q = QuantumRegister(self._operator.num_qubits, name='q')
        self._ancillary_register = a
        self._state_register = q
        qc = QuantumCircuit(q)
        qc += self._state_in.construct_circuit('circuit', q)
        # hadamard on a[0]
        qc.add_register(a)
        qc.u2(0, np.pi, a[0])
        # controlled-U
        qc_evolutions_inst = evolution_instruction(self._slice_pauli_list, -2 * np.pi,
                                                   self._num_time_slices,
                                                   controlled=True, power=2 ** (k - 1),
                                                   shallow_slicing=self._shallow_circuit_concat)
        if self._shallow_circuit_concat:
            qc_evolutions = QuantumCircuit(q, a)
            qc_evolutions.append(qc_evolutions_inst, list(q) + [a[0]])
            qc.data += qc_evolutions.data
        else:
            qc.append(qc_evolutions_inst, list(q) + [a[0]])
        # global phase due to identity pauli
        qc.u1(2 * np.pi * self._ancilla_phase_coef * (2 ** (k - 1)), a[0])
        # rz on a[0]
        qc.u1(omega, a[0])
        # hadamard on a[0]
        qc.u2(0, np.pi, a[0])
        if measurement:
            c = ClassicalRegister(1, name='c')
            qc.add_register(c)
            # qc.barrier(self._ancillary_register)
            qc.measure(self._ancillary_register, c)
        return qc

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Union[OperatorBase, LegacyBaseOperator]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def _estimate_phase_iteratively(self):
        """
        Iteratively construct the different order of controlled evolution
        circuit to carry out phase estimation.
        """
        self._ret['top_measurement_label'] = ''

        omega_coef = 0
        # k runs from the number of iterations back to 1
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2
            if self._quantum_instance.is_statevector:
                qc = self.construct_circuit(k, -2 * np.pi * omega_coef, measurement=False)
                result = self._quantum_instance.execute(qc)
                complete_state_vec = result.get_statevector(qc)
                ancilla_density_mat = get_subsystem_density_matrix(
                    complete_state_vec,
                    range(self._operator.num_qubits)
                )
                ancilla_density_mat_diag = np.diag(ancilla_density_mat)
                max_amplitude = max(ancilla_density_mat_diag.min(),
                                    ancilla_density_mat_diag.max(), key=abs)
                x = np.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            else:
                qc = self.construct_circuit(k, -2 * np.pi * omega_coef, measurement=True)
                measurements = self._quantum_instance.execute(qc).get_counts(qc)

                if '0' not in measurements:
                    if '1' in measurements:
                        x = 1
                    else:
                        raise RuntimeError('Unexpected measurement {}.'.format(measurements))
                else:
                    if '1' not in measurements:
                        x = 0
                    else:
                        x = 1 if measurements['1'] > measurements['0'] else 0
            self._ret['top_measurement_label'] = \
                '{}{}'.format(x, self._ret['top_measurement_label'])
            omega_coef = omega_coef + x / 2
            logger.info('Reverse iteration %s of %s with measured bit %s',
                        k, self._num_iterations, x)
        return omega_coef

    def _compute_energy(self):
        # check for identify paulis to get its coef for applying global phase shift on ancilla later
        num_identities = 0
        self._pauli_list = self._operator.reorder_paulis()
        for p in self._pauli_list:
            if np.all(np.logical_not(p[1].z)) and np.all(np.logical_not(p[1].x)):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

        self._ret['phase'] = self._estimate_phase_iteratively()
        self._ret['top_measurement_decimal'] = sum([t[0] * t[1] for t in zip(
            [1 / 2 ** p for p in range(1, self._num_iterations + 1)],
            [int(n) for n in self._ret['top_measurement_label']]
        )])
        self._ret['energy'] = self._ret['phase'] / self._ret['stretch'] - self._ret['translation']

    def _run(self) -> 'IQPEResult':
        self._compute_energy()

        result = IQPEResult()
        if 'translation' in self._ret:
            result.translation = self._ret['translation']
        if 'stretch' in self._ret:
            result.stretch = self._ret['stretch']
        if 'top_measurement_label' in self._ret:
            result.top_measurement_label = self._ret['top_measurement_label']
        if 'top_measurement_decimal' in self._ret:
            result.top_measurement_decimal = self._ret['top_measurement_decimal']
        if 'energy' in self._ret:
            result.eigenvalue = complex(self._ret['energy'])
        if 'phase' in self._ret:
            result.phase = self._ret['phase']

        return result


class IQPEResult(QPEResult):
    """ IQPE Result."""

    @property
    def phase(self) -> float:
        """ Returns phase """
        return self.get('phase')

    @phase.setter
    def phase(self, value: float) -> None:
        """ Sets phase """
        self.data['phase'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'IQPEResult':
        """ create new object from a dictionary """
        return IQPEResult(a_dict)
