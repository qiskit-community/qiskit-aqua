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


"""The Iterative Quantum Phase Estimation Algorithm."""


from typing import Optional, Union
import numpy
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import AlgorithmResult


class IPhaseEstimation(QuantumAlgorithm):
    """Run the Iterative quantum phase estimation (QPE) algorithm.

    Given a unitary circuit and a circuit preparing an eigenstate, return the phase of the
    eigenvalue as a number in :math:`[0,1)` using the iterative phase estimation algorithm.

    [1]: Dobsicek et al. (2006), Arbitrary accuracy iterative phase estimation algorithm as a two
       qubit benchmark, `arxiv/quant-ph/0610214 <https://arxiv.org/abs/quant-ph/0610214>`_
    """

    def __init__(self,
                 num_iterations: int,
                 unitary: Optional[QuantumCircuit] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:

        """Args:
            num_iterations: The number of iterations (rounds) of the phase estimation to run.
            unitary: The circuit representing the unitary operator whose eigenvalue (via phase)
                     will be measured.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                                 measured.  If this parameter is omitted, no preparation circuit
                                 will be run and input state will be the all-zero state in the
                                 computational basis.
            quantum_instance: The quantum instance on which the circuit will be run.
        """

        self._num_iterations = num_iterations
        self._unitary = unitary
        self._state_preparation = state_preparation

        super().__init__(quantum_instance)

    def construct_circuit(self,
                          k: int,
                          omega: float = 0,
                          measurement: bool = False) -> QuantumCircuit:
        """Construct the kth iteration Quantum Phase Estimation circuit.

        For details of parameters, see Fig. 2 in https://arxiv.org/pdf/quant-ph/0610214.pdf.

        Args:
            k: the iteration idx.
            omega: the feedback angle.
            measurement: Boolean flag to indicate if measurement should
                    be included in the circuit.

        Returns:
            QuantumCircuit: the quantum circuit per iteration
        """
        if self._unitary is None is None:
            return None

        k = self._num_iterations if k is None else k
        # The auxiliary (phase measurement) qubit
        phase_register = QuantumRegister(1, name='a')
        eigenstate_register = QuantumRegister(self._unitary.num_qubits, name='q')
        qc = QuantumCircuit(eigenstate_register)
        qc.add_register(phase_register)
        if isinstance(self._state_preparation, QuantumCircuit):
            qc.append(self._state_preparation, eigenstate_register)
        elif self._state_preparation is not None:
            qc += self._state_preparation.construct_circuit('circuit', eigenstate_register)
        # hadamard on phase_register[0]
        qc.h(phase_register[0])
        # controlled-U
        # TODO: We may want to allow flexibility in how the power is computed
        # For example, it may be desirable to compute the power via Trotterization, if
        # we are doing Trotterization anyway.
        unitary_power = self._unitary.power(2 ** (k - 1)).control()
        qc.append(unitary_power, list(range(1, self._unitary.num_qubits + 1)) + [0])
        qc.p(omega, phase_register[0])
        # hadamard on phase_register[0]
        qc.h(phase_register[0])
        if measurement:
            c = ClassicalRegister(1, name='c')
            qc.add_register(c)
            qc.measure(phase_register, c)
        return qc

    def _estimate_phase_iteratively(self):
        """
        Main loop of iterative phase estimation.
        """
        omega_coef = 0
        # k runs from the number of iterations back to 1
        for k in range(self._num_iterations, 0, -1):
            omega_coef /= 2
            if self._quantum_instance.is_statevector:
                qc = self.construct_circuit(k, -2 * numpy.pi * omega_coef, measurement=False)
                result = self._quantum_instance.execute(qc)
                complete_state_vec = result.get_statevector(qc)
                ancilla_density_mat = get_subsystem_density_matrix(
                    complete_state_vec,
                    range(self._unitary.num_qubits)
                )
                ancilla_density_mat_diag = numpy.diag(ancilla_density_mat)
                max_amplitude = max(ancilla_density_mat_diag.min(),
                                    ancilla_density_mat_diag.max(), key=abs)
                x = numpy.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            else:
                qc = self.construct_circuit(k, -2 * numpy.pi * omega_coef, measurement=True)
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
            omega_coef = omega_coef + x / 2
        return omega_coef

    def estimate(self,
                 num_iterations: Optional[int] = None,
                 unitary: Optional[QuantumCircuit] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None
                 ) -> 'IPhaseEstimationResult':
        """
        Estimate the phase. The parameters for `estimate` are the same as those in the constructor
        for `IPhaseEstimation`. Here, any of them may be omitted, in which case the previous value
        will be used. Thus, this method is used to repeat an experiment while changing only some
        of the inputs.
        """

        if num_iterations is not None:
            self._num_iterations = num_iterations
        if unitary is not None:
            self._unitary = unitary
        if state_preparation is not None:
            self._state_preparation = state_preparation
        if quantum_instance is not None:
            self._quantum_instance = quantum_instance
        phase = self._estimate_phase_iteratively()

        return IPhaseEstimationResult(self._num_iterations, phase)

    def _run(self):
        pass


class IPhaseEstimationResult(AlgorithmResult):
    """Phase Estimation Result."""

    def __init__(self,
                 num_iterations: int,
                 phase: float) -> None:
        """
        Args:
            num_iterations: number of iterations used in the phase estimation.
            phase: the estimated phase.
        """

        super().__init__({'phase': phase, 'num_iterations': num_iterations})

    @property
    def phase(self) -> float:
        r"""Return the estimated phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. It is assumed that the input vector is an
        eigenvector of the unitary so that the peak of the probability density occurs at the bit
        string that most closely approximates the true phase.
        """
        return self['phase']

    @property
    def num_iterations(self) -> int:
        r"""Return the number of iterations used in the estimation algorithm."""
        return self['num_iterations']
