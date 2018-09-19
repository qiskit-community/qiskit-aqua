from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import plot_circuit
import qiskit.extensions.simulator
from qiskit.extensions.simulator.snapshot import Snapshot
import numpy as np
from rus_routines import RUS_Subroutine, RUS_GB, RUS_PAR

class RUSCircuit():

    def __init__(self, qc=None):
        self._qc = qc if qc else QuantumCircuit()
        self._gcr = ClassicalRegister(1)
        self._null = self._gcr[0]
        self._qc.add(self._gcr)
        self._initialized_free_qubits = []
        self._qregs = []
        self._cregs = []
        self.data = []
    
    def get_circuit(self):
        return self._qc

    def plot(self):
        origdata = self._qc.data
        self._qc.data = list(filter(lambda x: not isinstance(x, Snapshot),
            origdata))
        plot_circuit(self._qc)
        self._qc.data = origdata

    def _get_qubits(self, n):
        if len(self._initialized_free_qubits) == n:
            ret = self._initialized_free_qubits
            self._initialized_free_qubits = []
        elif len(self._initialized_free_qubits) > n:
            ret = self._initialized_free_qubits[:n]
            self._initialized_free_qubits = self._initialized_free_qubits[n:]
        else:
            ret = self._initialized_free_qubits
            self._initialized_free_qubits = []
            q = QuantumRegister(n-len(ret))
            self._qregs.append(q)
            self._qc.add(q) 
            ret += [qi for qi in q]
        return ret

    def add_qreg(self, reg):
        self._qc.add(reg)
        self._qregs.append(reg)

    def add(self, RUS_Class, inputs, input_qubits=[], output=None, mode=""):
        nqubits = len(inputs)
        if isinstance(output, RUS_Subroutine):
            output = output.get_output_bit()
        if output == None:
            nqubits += 1
        _input_qubits = []
        for iq in input_qubits:
            if isinstance(iq, tuple):
                if iq[0] not in self._qregs:
                    self._qc.add(iq[0])
                    self._qregs.append(iq[0])
                ret = iq
            if isinstance(iq, RUS_Subroutine):
                ret = iq._output
            _input_qubits += [ret]
        qubits = self._get_qubits(nqubits)
        if output == None:
            output = qubits.pop()
        subroutine = RUS_Class(inputs, qubits, output,
                input_qubits=_input_qubits, mode=mode)
        self.data.append(subroutine)
        subroutine._construct_circuit(self._qc)
        subroutine._measure_and_reset(self._qc, self._null, len(self.data))
        self._initialized_free_qubits += subroutine._qubits + subroutine._input_qubits
        return subroutine

    def _key_index_for_qubit(self, qubit):
        regd = 0
        i = 0
        qreg = self._qregs[i]
        while qreg != qubit[0]:
            regd += len(qreg)+1
            i += 1
            qreg = self._qregs[i]
        return regd + qubit[1]
    
    def _filter_interests(self, subroutine, qs):
        idx_interests = list(map(self._key_index_for_qubit,
            subroutine._qubits + subroutine._input_qubits))
        ret = []
        for qstate in qs:
            d = {}
            for key, val in qstate.items():
                nkey = "".join(np.array(list(reversed(key)))[idx_interests])
                prob = val[0]**2+val[1]**2
                if nkey in d:
                    d[nkey] += prob
                else:
                    d[nkey] = prob
            ret += [d]
        return ret

    def _filter_result(self, qs, usable, qubit):
        idx = self._key_index_for_qubit(qubit)
        res = []
        for qstate, use in zip(qs, usable):
            state = np.zeros(2, dtype=complex)
            if use: 
                for key, val in qstate.items():
                    state[int(list(reversed(key))[idx])] += val[0]+1j*val[1]
                res.append(state)
        difs = [res[0]]
        counts = [0]
        for r in res[1:]:
            fit = False
            for i, vgl in enumerate(difs):
                if (r-vgl).conj().dot(r-vgl) < 1e-10:
                    counts[i] += 1
                    fit = True
                    break
            if not fit:
                difs.append(r)
                counts.append(0)
        ret = sorted(zip(difs, counts), key=lambda x: x[1], reverse=True)
        return ret[0][0] if ret[0][0].conj().dot(ret[0][0]) != 0 else ret[1][0]

    def _calculate_angle(self, state):
        a, b = state
        if a == 0:
            a = 1e-16
        return np.arctan(-b.imag/a).real
            
    def eval(self, shots=100):
        self._qc.snapshot("-1")
        res = execute(self._qc, "local_qasm_simulator", {"data":
            ["hide_statevector", "quantum_state_ket"]}, shots=shots).result()
        usable = np.ones(shots, dtype=bool)
        for sub in self.data:
            qs = res.get_snapshot(sub.get_slot()).get("quantum_state_ket")
            filtered = self._filter_interests(sub, qs)
            usable = np.logical_and(usable, sub.get_success(filtered))
        final_snap = res.get_snapshot("-1").get("quantum_state_ket")
        ret_dir = {}
        ret_dir["final_state"] = self._filter_result(final_snap, usable,
                self.data[-1].get_output_bit())
        ret_dir["prob"] = sum(usable)/len(usable)
        ret_dir["angle"] = self._calculate_angle(ret_dir["final_state"])
        return ret_dir

def square(alpha, x, output, rc, order=3):
    factor = alpha
    if factor < 0:
        factor = -factor
        mode = "neg_rot"
    else:
        mode = ""
    base = rc.add(RUS_GB, [x, np.arcsin(np.sqrt(factor))], mode=mode)
    if order >= 2:
        factor = alpha**2-alpha/3
        if factor < 0:
            factor = -factor
            mode = ""
        else:
            mode = "neg_rot"
        rc.add(RUS_GB, [x, x, np.arcsin(np.sqrt(factor))],
                output=base, mode=mode)
    if order >= 3:
        factor = 2*alpha**3/3-8*alpha/45
        if factor < 0:
            factor = -factor
            mode = ""
        else:
            mode = "neg_rot"
        rc.add(RUS_GB, [x, x, x, np.arcsin(np.sqrt(factor))],
                output=base, mode=mode)

def tothe4(alpha, x, output, rc, order=3):
    factor = alpha
    if factor < 0:
        factor = -factor
        mode = "neg_rot"
    else:
        mode = ""
    base = rc.add(RUS_GB, [x, x, np.arcsin(np.sqrt(factor))], mode=mode)
    if order >= 2:
        factor = -2*alpha/3
        if factor < 0:
            factor = -factor
            mode = ""
        else:
            mode = "neg_rot"
        rc.add(RUS_GB, [x, x, x, np.arcsin(np.sqrt(factor))],
                output=base, mode=mode)


def eval_bunch(x, SUBR, input_qubits=[]):
    angles, probs = [], []
    for i in x:
        out = QuantumRegister(1)
        rc = RUSCircuit()
        rc.add_qreg(out)

        #square(0.5, i, out[0], rc)
        tothe4(-0.3, i, out[0], rc)

        #r1 = rc.add(RUS_PAR, [i, np.arcsin(0.608)], output=out[0])

        #rc._qc.rx(2*np.arcsin(1), out)
        
        #r2 = rc.add(RUS_GB, [i, np.arcsin(np.sqrt(2.66/3))], output=out[0])
        #r2 = rc.add(RUS_GB, [i, np.arcsin(np.sqrt(2.66/3))], output=out[0])
        #r2 = rc.add(RUS_GB, [i, np.arcsin(np.sqrt(2.66/3))], output=out[0])


        # r1 = rc.add(RUS_GB, [i, np.arcsin(np.sqrt(1/6))],
        #         input_qubits=input_qubits, mode="neg_rot", output=out[0])
        # # r2 = rc.add(SUBR, [i, np.arcsin(np.sqrt(1/6))],
        # #         input_qubits=input_qubits, mode="neg_rot", output=r1)
        # rc._qc.rx(np.pi/2, r2._output)
        # # q = QuantumRegister(1)
        # # rc.add_qreg(q)
        # # rc._qc.rx(2*np.arctan(1), q)
        # r3 = rc.add(RUS_PAR, [i, i], input_qubits=[r2])
        # # r3 = rc.add(RUS_PAR, [i, i, np.arctan(1)])
        # if i==0: rc.plot()
        res = rc.eval()
        angles.append(res.get("angle"))
        probs.append(res.get("prob"))
    return angles, probs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 20)
    angles, probs = eval_bunch(x, RUS_GB)
    plt.plot(x, angles)
    plt.plot(x, probs)
    # plt.plot(x, -2/6*x**2+np.pi/4)
    plt.plot(x, -0.3*x**4)#-0.3*x**4)
    plt.show()

    #gb2 = rc.add(RUS_GB, [b, np.arcsin(np.sqrt(1/6))], output=gb1,
    #        mode="neg_rot")
    #rc._qc.rx(np.pi/2, gb2._output)
    #par = rc.add(RUS_PAR, [a, b], input_qubits=[gb2])
    # rc.plot()

