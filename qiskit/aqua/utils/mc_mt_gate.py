import logging
from math import pi, ceil

from qiskit import QuantumCircuit, QuantumRegister


def ccx_v_chain_compute(qc, control_qubits, target_qubit, ancillary_qubits,
                        anci_idx):
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
    for idx in range(2, len(control_qubits)):
        assert anci_idx + 1 < len(
            ancillary_qubits), "Insufficient number of ancillary qubits."
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx],
               ancillary_qubits[anci_idx + 1])
        anci_idx += 1
    return anci_idx


def ccx_v_chain_uncompute(qc, control_qubits, target_qubit, ancillary_qubits,
                          anci_idx):
    for idx in (range(2, len(control_qubits)))[::-1]:
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1],
               ancillary_qubits[anci_idx])
        anci_idx -= 1
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])


def mc_gate(self,
            control_qubits,
            ancillary_qubits,
            single_control_gate_fun,
            target_gate_qubit,
            mode="basic"):
    if mode == 'basic':
        # last ancillary qubit is the control of the gate
        ancn = len(ancillary_qubits)
        anci_idx = ccx_v_chain_compute(
            self, control_qubits,
            [ancillary_qubits[i]
             for i in range(0, ancn - 1)], ancillary_qubits[ancn - 1], 0)
        single_control_gate_fun(self, ancillary_qubits[ancn - 1],
                                target_gate_qubit)
        ccx_v_chain_uncompute(
            self, control_qubits,
            [ancillary_qubits[i] for i in range(0, ancn - 1)],
            ancillary_qubits[ancn - 1], anci_idx)
    else:
        raise ValueError(
            'Unrecognized mode for building MC_gate circuit: {}.'.format(mode))


def mc_mt_gate(self,
               control_qubits,
               target_control_qubit,
               ancillary_qubits,
               c_gate_fun,
               target_gate_qubits,
               mode="basic"):
    if mode == 'basic':
        anci_idx = ccx_v_chain_compute(
            self, control_qubits, target_control_qubit, ancillary_qubits, 0)
        for target_gate_qubit in target_gate_qubits:
            c_gate_fun(self, target_control_qubit, target_gate_qubit)
        ccx_v_chain_uncompute(self, control_qubits, target_control_qubit,
                              ancillary_qubits, anci_idx)
    else:
        raise ValueError('No other mode supported yet: {}.'.format(mode))


QuantumCircuit.mc_gate = mc_gate
QuantumCircuit.mc_mt_gate = mc_mt_gate
