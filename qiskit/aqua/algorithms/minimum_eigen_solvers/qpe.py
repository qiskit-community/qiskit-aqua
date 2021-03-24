# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Quantum Phase Estimation Algorithm."""

import logging
from typing import Optional, List, Dict, Union, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import op_converter, OperatorBase
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.circuits import PhaseEstimationCircuit
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import LegacyBaseOperator
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class QPE(QuantumAlgorithm, MinimumEigensolver):
    """The Quantum Phase Estimation algorithm.

    QPE (also sometimes abbreviated as PEA, for Phase Estimation Algorithm), has two quantum
    registers, **control** and **target**, where the control consists of several qubits initially
    put in uniform superposition, and the target a set of qubits prepared in an eigenstate
    (often a guess of the eigenstate) of the unitary operator of a quantum system.
    QPE then evolves the target under the control using dynamics on the unitary operator.
    The information of the corresponding eigenvalue is then 'kicked-back' into the phases of the
    control register, which can then be deconvoluted by an Inverse Quantum Fourier Transform (IQFT),
    and measured for read-out in binary decimal format. QPE also requires a reasonably good
    estimate of the eigen wave function to start the process. For example, when estimating
    molecular ground energies in chemistry, the Hartree-Fock method could be used to provide such
    trial eigen wave functions.
    """

    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 state_in: Optional[Union[InitialState, QuantumCircuit]] = None,
                 iqft: Optional[QuantumCircuit] = None,
                 num_time_slices: int = 1,
                 num_ancillae: int = 1,
                 expansion_mode: str = 'trotter',
                 expansion_order: int = 1,
                 shallow_circuit_concat: bool = False,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """

        Args:
            operator: The Hamiltonian Operator
            state_in: An optional InitialState component representing an initial quantum state.
                ``None`` may be supplied.
            iqft: A Inverse Quantum Fourier Transform component
            num_time_slices: The number of time slices, has a minimum value of 1.
            num_ancillae: The number of ancillary qubits to use for the measurement,
                has a min. value of 1.
            expansion_mode: The expansion mode ('trotter'|'suzuki')
            expansion_order: The suzuki expansion order, has a min. value of 1.
            shallow_circuit_concat: Set True to use shallow (cheap) mode for circuit concatenation
                of evolution slices. By default this is False.
                See :meth:`qiskit.aqua.operators.common.evolution_instruction` for more information.
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('num_time_slices', num_time_slices, 1)
        validate_min('num_ancillae', num_ancillae, 1)
        validate_in_set('expansion_mode', expansion_mode, {'trotter', 'suzuki'})
        validate_min('expansion_order', expansion_order, 1)
        super().__init__(quantum_instance)  # type: ignore

        self._state_in = state_in
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._binary_fractions = [1 / 2 ** p for p in range(1, self._num_ancillae + 1)]
        self._in_operator = operator
        self._operator = None  # type: Optional[WeightedPauliOperator]
        self._ret = {}  # type: Dict[str, Any]
        self._pauli_list = None  # type: Optional[List[List[Union[complex, Pauli]]]]
        self._phase_estimation_circuit = None
        self._setup(operator)

    def _setup(self, operator: Optional[Union[OperatorBase, LegacyBaseOperator]]) -> None:
        self._operator = None
        self._ret = {}
        self._pauli_list = None
        self._phase_estimation_circuit = None
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
                        (np.zeros(self._operator.num_qubits),
                         np.zeros(self._operator.num_qubits))
                    )
                ]
            ])
            translation_op.simplify()
            self._operator += translation_op
            self._pauli_list = self._operator.reorder_paulis()

            # stretch the operator
            for p in self._pauli_list:
                p[0] = p[0] * self._ret['stretch']

            self._phase_estimation_circuit = PhaseEstimationCircuit(
                operator=self._operator, state_in=self._state_in, iqft=self._iqft,
                num_time_slices=self._num_time_slices, num_ancillae=self._num_ancillae,
                expansion_mode=self._expansion_mode, expansion_order=self._expansion_order,
                shallow_circuit_concat=self._shallow_circuit_concat, pauli_list=self._pauli_list
            )

    @property
    def operator(self) -> Optional[LegacyBaseOperator]:
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

    def construct_circuit(self, measurement: bool = False) -> QuantumCircuit:
        """
        Construct circuit.

        Args:
            measurement: Boolean flag to indicate if measurement
                should be included in the circuit.

        Returns:
            QuantumCircuit: quantum circuit.
        """
        if self._phase_estimation_circuit:
            return self._phase_estimation_circuit.construct_circuit(measurement=measurement)

        return None

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Union[OperatorBase, LegacyBaseOperator]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def _compute_energy(self):
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            ancilla_density_mat = get_subsystem_density_matrix(
                complete_state_vec,
                range(self._num_ancillae, self._num_ancillae + self._operator.num_qubits)
            )
            ancilla_density_mat_diag = np.diag(ancilla_density_mat)
            max_amplitude = \
                max(ancilla_density_mat_diag.min(), ancilla_density_mat_diag.max(), key=abs)
            max_amplitude_idx = np.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            top_measurement_label = np.binary_repr(max_amplitude_idx, self._num_ancillae)[::-1]
        else:
            qc = self.construct_circuit(measurement=True)
            result = self._quantum_instance.execute(qc)
            ancilla_counts = result.get_counts(qc)
            top_measurement_label = \
                sorted([(ancilla_counts[k], k) for k in ancilla_counts])[::-1][0][-1][::-1]

        top_measurement_decimal = sum(
            [t[0] * t[1] for t in zip(self._binary_fractions,
                                      [int(n) for n in top_measurement_label])]
        )

        self._ret['top_measurement_label'] = top_measurement_label
        self._ret['top_measurement_decimal'] = top_measurement_decimal
        self._ret['eigvals'] = \
            [top_measurement_decimal / self._ret['stretch'] - self._ret['translation']]
        self._ret['energy'] = self._ret['eigvals'][0]

    def _run(self) -> 'QPEResult':
        self._compute_energy()

        result = QPEResult()
        if 'translation' in self._ret:
            result.translation = self._ret['translation']
        if 'stretch' in self._ret:
            result.stretch = self._ret['stretch']
        if 'top_measurement_label' in self._ret:
            result.top_measurement_label = self._ret['top_measurement_label']
        if 'top_measurement_decimal' in self._ret:
            result.top_measurement_decimal = self._ret['top_measurement_decimal']
        if 'eigvals' in self._ret:
            result.eigenvalue = self._ret['eigvals'][0]

        return result


class QPEResult(MinimumEigensolverResult):
    """ QPE Result."""

    @property
    def translation(self) -> float:
        """ Returns translation """
        return self.get('translation')

    @translation.setter
    def translation(self, value: float) -> None:
        """ Sets translation """
        self.data['translation'] = value

    @property
    def stretch(self) -> float:
        """ Returns stretch """
        return self.get('stretch')

    @stretch.setter
    def stretch(self, value: float) -> None:
        """ Sets stretch """
        self.data['stretch'] = value

    @property
    def top_measurement_label(self) -> str:
        """ Returns top measurement label """
        return self.get('top_measurement_label')

    @top_measurement_label.setter
    def top_measurement_label(self, value: str) -> None:
        """ Sets top measurement label """
        self.data['top_measurement_label'] = value

    @property
    def top_measurement_decimal(self) -> float:
        """ Returns top measurement decimal """
        return self.get('top_measurement_decimal')

    @top_measurement_decimal.setter
    def top_measurement_decimal(self, value: float) -> None:
        """ Sets top measurement decimal """
        self.data['top_measurement_decimal'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'QPEResult':
        """ create new object from a dictionary """
        return QPEResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'aux_operator_eigenvalues':
            raise KeyError('aux_operator_eigenvalues not supported.')

        return super().__getitem__(key)
