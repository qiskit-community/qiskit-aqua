# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
import logging
from itertools import combinations, chain

import numpy as np
from parameterized import parameterized
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua import get_aer_backend
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity
from test.common import QiskitAquaTestCase

import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class TestMCGate(QiskitAquaTestCase):
    @parameterized.expand([(3, QuantumCircuit.cz), (4, QuantumCircuit.cz),
                           (5, QuantumCircuit.cz)])
    def test_mc_gate(self, num_controls, single_control_gate_function):
        logger.debug("test_mc_gate -> Num controls = {0}".format(num_controls))
        logger.debug("test_mc_gate -> Gate function is {0}".format(
            single_control_gate_function.__name__))
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(1, name='o')

        allsubsets = list(
            chain(*[
                combinations(range(num_controls), ni)
                for ni in range(num_controls + 1)
            ]))
        for subset in allsubsets:
            logger.debug("test_mc_gate -> Subset is {0}".format(subset))
            for mode in ['basic']:
                logger.debug("test_mc_gate -> Mode is {0}".format(mode))
                qc = QuantumCircuit(o, c)
                # TODO just for cz, generalize
                qc.x(o)

                if mode == 'basic':
                    num_ancillae = num_controls - 1
                    logger.debug("test_mc_gate -> Num ancillae is {0} ".format(
                        num_ancillae))
                    a = QuantumRegister(num_ancillae, name='a')
                    qc.add_register(a)
                for idx in subset:
                    qc.x(c[idx])
                qc.mc_gate([c[i] for i in range(num_controls)],
                           [a[i] for i in range(num_ancillae)],
                           single_control_gate_function,
                           o[0],
                           mode=mode)
                for idx in subset:
                    qc.x(c[idx])

                vec = np.asarray(
                    q_execute(qc, get_aer_backend('statevector_simulator')).
                    result().get_statevector(qc, decimals=16))
                vec_o = [0, -1] if len(subset) == num_controls else [0, 1]
                tmp = np.array(vec_o + [0] *
                               (2**(num_controls + num_ancillae + 1) - 2))
                f = state_fidelity(vec, tmp)

                logger.debug("test_mc_gate -> drawing")
                file_path = "imgs/multi_ctrl_{0}_{1}_{2}_{3}".format(
                    single_control_gate_function.__name__, num_controls,
                    subset, mode)
                draw_circuit(qc, file_path)
                self.assertAlmostEqual(f, 1)
                # try:
                #     self.assertAlmostEqual(f, 1)
                # except AssertionError:
                #     file_path = "imgs/multi_ctrl_{0}_{1}_{2}_{3}".format(
                #         single_control_gate_function.__name__, num_controls,
                #         subset, mode)
                #     draw_circuit(qc, file_path)
                #     raise


class TestMCMTGate(QiskitAquaTestCase):
    @parameterized.expand([
        # [1],
        # [2],
        (3, 2, QuantumCircuit.cz),
        # (3, 3, QuantumCircuit.cz),
        # (3, 4, QuantumCircuit.cz),
        # [5],
        # [6],
        # [7],
        # [12],
    ])
    @unittest.skip('No reason')
    def test_mc_mt_gate(self, num_controls, num_targets,
                        single_control_gate_function):
        logger.debug(
            "test_mc_mt_gate -> Num controls = {0}".format(num_controls))
        logger.debug(
            "test_mc_mt_gate -> Num targets = {0}".format(num_targets))
        logger.debug("test_mc_mt_gate -> Gate function is {0}".format(
            single_control_gate_function.__name__))
        c = QuantumRegister(num_controls, name='c')
        o = QuantumRegister(num_targets, name='o')

        # all possible combinations of controls set to 1
        allsubsets = list(
            chain(*[
                combinations(range(num_controls), ni)
                for ni in range(num_controls + 1)
            ]))
        for subset in allsubsets:
            logger.debug("test_mc_mt_gate -> Subset is {0}".format(subset))
            for mode in ['basic']:
                logger.debug("test_mc_mt_gate -> Mode is {0}".format(mode))
                qc = QuantumCircuit(o, c)
                # TODO just for cz, generalize
                qc.x(o)

                if mode == 'basic':
                    if num_controls <= 2:
                        num_ancillae = 0
                    else:
                        num_ancillae = num_controls - 1
                    logger.debug("Num ancillae is {0} ".format(num_ancillae))
                if num_ancillae > 0:
                    a = QuantumRegister(num_ancillae, name='a')
                    qc.add_register(a)
                for idx in subset:
                    qc.x(c[idx])
                qc.mc_mt_gate([c[i] for i in range(num_controls)],
                              a[num_ancillae - 1],
                              [a[i] for i in range(num_ancillae)],
                              single_control_gate_function,
                              o[0],
                              mode=mode)
                for idx in subset:
                    qc.x(c[idx])

                logger.debug("test_mc_mt_gate -> drawing")
                file_path = "imgs/mctrl_mtgt_{0}_{1}_{2}_{3}".format(
                    single_control_gate_function.__name__, num_controls, mode)
                draw_circuit(qc, file_path)
                assert (True)


def draw_circuit(qc, file_path):
    style_mpl = {
        'cregbundle': True,
        'compress': True,
        'usepiformat': True,
        'subfontsize': 12,
        'fold': 100,
        'showindex': True,
        "displaycolor": {
            "id": "#ffca64",
            "u0": "#f69458",
            "u1": "#f69458",
            "u2": "#f69458",
            "u3": "#f69458",
            "x": "#a6ce38",
            "y": "#a6ce38",
            "z": "#a6ce38",
            "h": "#00bff2",
            "s": "#00bff2",
            "sdg": "#00bff2",
            "t": "#ff6666",
            "tdg": "#ff6666",
            "rx": "#ffca64",
            "ry": "#ffca64",
            "rz": "#ffca64",
            "reset": "#d7ddda",
            "target": "#00bff2",
            "meas": "#f070aa"
        }
    }
    circuit_drawer(qc, filename=file_path, style=style_mpl, output='mpl')


if __name__ == '__main__':
    unittest.main()
