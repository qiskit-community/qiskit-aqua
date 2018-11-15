from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit_aqua import AlgorithmError

#TODO: should come from qiskit_aqua
from utils.controlledcircuit import get_controlled_circuit


class CircuitFactory(ABC):

    """ Base class for CircuitFactories """

    def __init__(self, num_target_qubits):
        self._num_target_qubits = num_target_qubits
        pass

    @property
    def num_target_qubits(self):
        """ Returns the number of target qubits """
        return self._num_target_qubits

    @abstractmethod
    def required_ancillas(self):
        """ Returns the number of required ancillas for an uncontrolled circuit application """
        raise NotImplementedError()

    @abstractmethod
    def required_ancillas_controlled(self):
        """ Returns the number of required ancillas for a controlled circuit application """
        raise NotImplementedError()

    def get_num_qubits(self):
        return self._num_target_qubits + self.required_ancillas()

    def get_num_qubits_controlled(self):
        return self._num_target_qubits + self.required_ancillas_controlled()

    @abstractmethod
    def build(self, qc, q, q_ancillas=None, params=None):
        """ Adds corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """
        raise NotImplementedError()

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        """ Adds inverse of corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """

        qreg_names = list(qc.get_qregs().keys())
        qc_ = QuantumCircuit(qc.get_qregs()[qreg_names[0]], name=qreg_names[0])
        for name in qreg_names[1:]:
            qc_.add(qc.get_qregs()[name])

        self.build(qc_, q, q_ancillas, params)
        try:
            qc_.data = [gate.inverse() for gate in reversed(qc_.data)]
        except AlgorithmError:
            print('Irreversible circuit! Does not support inverse method.')
        qc.extend(qc_)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        """ Adds corresponding controlled sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_control : control qubit
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """

        qreg_names = list(qc.get_qregs().keys())
        uncontrolled_circuit = QuantumCircuit(qc.get_qregs()[qreg_names[0]], name=qreg_names[0])
        for name in qreg_names[1:]:
            uncontrolled_circuit.add(qc.get_qregs()[name])

        self.build(uncontrolled_circuit, q, q_ancillas, params)
        controlled_circuit = get_controlled_circuit(uncontrolled_circuit, q_control)
        qc.extend(controlled_circuit)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        """ Adds controlled inverse of corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_control : control qubit
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """

        qreg_names = list(qc.get_qregs().keys())
        qc_ = QuantumCircuit(qc.get_qregs()[qreg_names[0]], name=qreg_names[0])
        for name in qreg_names[1:]:
            qc_.add(qc.get_qregs()[name])

        self.build_controlled(qc_, q, q_control, q_ancillas, params)
        try:
            qc_.data = [gate.inverse() for gate in reversed(qc_.data)]
        except AlgorithmError:
            print('Irreversible circuit! Does not support inverse method.')
        qc.extend(qc_)

    def build_power(self, qc, q, power, q_ancillas=None, params=None):
        """ Adds power of corresponding circuit - can be overwritten in case a more efficient implementation is possible """
        for _ in range(power):
            self.build(qc, q, q_ancillas, params)

    def build_inverse_power(self, qc, q, power, q_ancillas=None, params=None):
        """ Adds power of corresponding circuit - can be overwritten in case a more efficient implementation is possible """
        for _ in range(power):
            self.build_inverse(qc, q, q_ancillas, params)

    def build_controlled_power(self, qc, q, q_control, power, q_ancillas=None, params=None):
        """ Adds power of corresponding circuit - can be overwritten in case a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled(qc, q, q_control, q_ancillas, params)

    def build_controlled_inverse_power(self, qc, q, q_control, power, q_ancillas=None, params=None):
        """ Adds power of corresponding circuit - can be overwritten in case a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled_inverse(qc, q, q_control, q_ancillas, params)
