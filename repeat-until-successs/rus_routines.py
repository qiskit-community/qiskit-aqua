import numpy as np
import qiskit.extensions.simulator
from abc import ABC, abstractmethod
import ncx, ncu1

class RUS_Subroutine(ABC):

    def __init__(self, vals, qubits, output, input_qubits=[], mode=""):
        self._vals = vals
        self._qubits = qubits
        self._input_qubits = input_qubits
        self._output = output
        self._slot = None
        self._mode = mode
        super().__init__()
    
    @abstractmethod
    def _construct_circuit(self, qc):
        pass

    @abstractmethod
    def _is_success(self, d):
        pass

    def _measure_and_reset(self, qc, c, slot):
        for qubit in self._qubits + self._input_qubits:
            qc.measure(qubit, c)
        qc.snapshot(slot)
        self._slot = str(slot)
        for qubit in self._qubits + self._input_qubits:
            qc.reset(qubit)

    def get_output_bit(self):
        return self._output

    def get_slot(self):
        return self._slot

    def get_success(self, x):
        if not isinstance(x, list):
            x = [x]
        return np.array(list(map(self._is_success, x)))


class RUS_GB(RUS_Subroutine):

    def _construct_circuit(self, qc):
        for val, qubit in zip(self._vals, self._qubits):
            qc.rx(2*val, qubit)
        gp = 1j if self._mode == "neg_rot" else -1j
        qc.ncx(self._qubits + self._input_qubits, self._output, global_phase=gp)
        for val, qubit in zip(self._vals, self._qubits):
            qc.rx(-2*val, qubit)

    def _is_success(self, d):
        key = '0'*len(self._qubits+self._input_qubits)
        if len(d) == 1:
            if key in d:
                return True
        else:
            if key in d and abs(d[key]-1) < 1e-10:
                return True
        return False

class RUS_PAR(RUS_Subroutine):

    def _construct_circuit(self, qc):
        for val, qubit in zip(self._vals, self._qubits):
            qc.rx(2*val, qubit)
        qc.ncx(self._qubits + self._input_qubits, self._output,
                global_phase=1j**(len(self._qubits+self._input_qubits)-1))
        qubits = self._qubits + self._input_qubits
        for i in range(len(qubits)-1):
            qc.cx(qubits[i+1], qubits[i])
        qc.h(qubits[-1])
        qc.cz(qubits[-1], self._output)

    def _is_success(self, d):
        key1 = '0'*(len(self._qubits+self._input_qubits)-1)+'1'
        key2 = '0'*len(self._qubits+self._input_qubits)
        if len(d) == 1:
            if key1 in d or key2 in d:
                return True
        else:
            if key1 in d and abs(d[key1]-1) < 1e-10:
                return True
            elif key2 in d and abs(d[key2]-1) < 1e-10:
                return True
        return False

