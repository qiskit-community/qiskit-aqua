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

"""Phase estimation for the spectrum of a Hamiltonian"""

from typing import Optional, Union
import numpy
from qiskit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import EvolutionBase, OperatorBase
from qiskit.providers import BaseBackend

# TODO: Remove temporary code when possible.
# This temporary code is spread out a bit in order to satisfy
# the linter. When the global phase changes in Terra have settled down,
# this can be removed.
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer


from .phase_estimator import PhaseEstimator
from . import phase_estimation_scale
from .hamiltonian_pe_result import HamiltonianPEResult
from .phase_estimation_scale import PhaseEstimationScale

# TODO: Remove temporary code when possible.
_DECOMPOSER1Q = OneQubitEulerDecomposer('U3')


# TODO: Remove temporary code when possible.
class TempPauliEvolve():
    """Evolve a Hamiltonian (for working around a bug in terra.)

    This works for 1Q operators. This is only a stop gap while
    waiting for a bug fix in Terra. This class does not do Trotterization, etc.
    It just computes the exact gate.
    Hopefully, this can be removed soon.
    """

    def __init__(self) -> None:
        pass

    def convert(self, evolved_operator: OperatorBase) -> QuantumCircuit:
        """Return a circuit for EvolvedOp

        Hmm, looks like the sign in the exponent may be wrong.
        """
        from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose
        matrix = evolved_operator.to_matrix()
        if evolved_operator.num_qubits == 1:
            return self._matrix_to_circuit_1q(matrix)
        elif evolved_operator.num_qubits == 2:
            return two_qubit_cnot_decompose(matrix)
        else:
            raise ValueError("wrong num qubits")

    def _matrix_to_circuit_1q(self, matrix: numpy.ndarray) -> QuantumCircuit:
        theta, phi, lam, global_phase = _DECOMPOSER1Q.angles_and_phase(matrix)
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(1)
        qc._append(U3Gate(theta, phi, lam), [q[0]], [])
        qc.global_phase = global_phase
        return qc


class HamiltonianPE(PhaseEstimator):
    r"""Run the Quantum Phase Estimation algorithm to find the eigenvalues of a Hermitian operator.

    This class is nearly the same as :class:`~qiskit.aqua.algorithms.PhaseEstimator`, differing only
    in that the input in that class is a unitary operator, whereas here the input is a Hermitian
    operator from which a unitary will be obtained by scaling and exponentiating. The scaling is
    performed in order to prevent the phases from wrapping around :math:`2\pi`. This class uses and
    works together with :class:`~qiskit.aqua.algorithms.PhaseEstimationScale` to manage scaling the
    Hamiltonian and the phases that are obtained by the QPE algorithm. This includes setting, or
    computing, a bound on the eigenvalues of the operator, using this bound to obtain a scale
    factor, scaling the operator, and shifting and scaling the measured phases to recover the
    eigenvalues.

    Note that, although we speak of "evolving" the state according the the Hamiltonian, in the
    present algorithm, we are not actually considering time evolution. Rather, the role of time is
    played by the scaling factor, which is chosen to best extract the eigenvalues of the
    Hamiltonian.
    """
    def __init__(self,
                 num_evaluation_qubits: int,
                 hamiltonian: OperatorBase,
                 evolution: Optional[Union[EvolutionBase, TempPauliEvolve]] = None,
                 state_preparation: Optional[QuantumCircuit] = None,
                 bound: Optional[float] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """
        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The
                                   phase will be estimated as a binary string with this many
                                   bits.
            hamiltonian: a Hamiltonian or Hermitian operator
            evolution: An evolution object that generates a unitary from `hamiltonian`.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                                 measured.  If this parameter is omitted, no preparation circuit
                                 will be run and input state will be the all-zero state in the
                                 computational basis.
            bound: An upper bound on the absolute value of the eigenvalues of
                `hamiltonian`. If omitted, and `hamiltonian` is a Pauli sum, then a bound will be
                computed.
            quantum_instance: The quantum instance on which the circuit will be run.
        """

        self._hamiltonian = hamiltonian
        self._evolution = evolution
        self._bound = bound

        self._set_scale()
        unitary = self._get_unitary()

        super().__init__(num_evaluation_qubits,
                         unitary=unitary,
                         pe_circuit=None,
                         num_unitary_qubits=None,
                         state_preparation=state_preparation,
                         quantum_instance=quantum_instance)

    def _set_scale(self) -> None:
        if self._bound is None:
            pe_scale = phase_estimation_scale.from_pauli_sum(self._hamiltonian)
            self._pe_scale = pe_scale
        else:
            self._pe_scale = PhaseEstimationScale(self._bound)

    def _get_unitary(self) -> QuantumCircuit:
        """Evolve the Hamiltonian to obtain a unitary.

        Apply the scaling to the Hamiltonian that has been computed from an eigenvalue bound
        and compute the unitary by applying the evolution object.
        """
        # scale so that phase does not wrap.
        scaled_hamiltonian = self._pe_scale.scale * self._hamiltonian
        unitary = self._evolution.convert(scaled_hamiltonian.exp_i())
        if not isinstance(unitary, QuantumCircuit):
            unitary_circuit = unitary.to_circuit()
        else:
            unitary_circuit = unitary

        # Decomposing twice allows some 1Q Hamiltonians to give correct results
        # when using MatrixEvolution(), that otherwise would give incorrect results.
        # It does not break any others that we tested.
        return unitary_circuit.decompose().decompose()

    def _run(self) -> HamiltonianPEResult:
        """Run the circuit and return and return `HamiltonianPEResult`.
        """

        circuit_result = self._quantum_instance.execute(self.construct_circuit())
        phases = self._compute_phases(circuit_result)
        return HamiltonianPEResult(
            self._num_evaluation_qubits, phases=phases,
            circuit_result=circuit_result, phase_estimation_scale=self._pe_scale)
