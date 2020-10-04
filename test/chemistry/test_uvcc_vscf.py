# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UVCC and VSCF Aqua extensions """

from test.chemistry import QiskitChemistryTestCase

from qiskit import BasicAer
from qiskit.chemistry import BosonicOperator

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.components.initial_states import VSCF
from qiskit.chemistry.components.variational_forms import UVCC


class TestUVCCVSCF(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 8
        self.reference_energy = 592.5346633819712

    def test_uvcc_vscf(self):
        """ uvcc vscf test """

        co2_2modes_2modals_2body = [[[[[0, 0, 0]], 320.8467332810141],
                                     [[[0, 1, 1]], 1760.878530705873],
                                     [[[1, 0, 0]], 342.8218290247543],
                                     [[[1, 1, 1]], 1032.396323618631]],
                                    [[[[0, 0, 0], [1, 0, 0]], -57.34003649795117],
                                     [[[0, 0, 1], [1, 0, 0]], -56.33205925807966],
                                     [[[0, 1, 0], [1, 0, 0]], -56.33205925807966],
                                     [[[0, 1, 1], [1, 0, 0]], -60.13032761856809],
                                     [[[0, 0, 0], [1, 0, 1]], -65.09576309934431],
                                     [[[0, 0, 1], [1, 0, 1]], -62.2363839133389],
                                     [[[0, 1, 0], [1, 0, 1]], -62.2363839133389],
                                     [[[0, 1, 1], [1, 0, 1]], -121.5533969109279],
                                     [[[0, 0, 0], [1, 1, 0]], -65.09576309934431],
                                     [[[0, 0, 1], [1, 1, 0]], -62.2363839133389],
                                     [[[0, 1, 0], [1, 1, 0]], -62.2363839133389],
                                     [[[0, 1, 1], [1, 1, 0]], -121.5533969109279],
                                     [[[0, 0, 0], [1, 1, 1]], -170.744837386338],
                                     [[[0, 0, 1], [1, 1, 1]], -167.7433236025723],
                                     [[[0, 1, 0], [1, 1, 1]], -167.7433236025723],
                                     [[[0, 1, 1], [1, 1, 1]], -179.0536532281924]]]

        basis = [2, 2]

        bosonic_op = BosonicOperator(co2_2modes_2modals_2body, basis)
        qubit_op = bosonic_op.mapping('direct', threshold=1e-5)

        init_state = VSCF(basis)

        num_qubits = sum(basis)
        uvcc_varform = UVCC(num_qubits, basis, [0, 1], initial_state=init_state)

        q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_transpiler=90, seed_simulator=12)
        optimizer = COBYLA(maxiter=1000)

        algo = VQE(qubit_op, uvcc_varform, optimizer)
        vqe_result = algo.run(q_instance)

        energy = vqe_result['optimal_value']

        self.assertAlmostEqual(energy, self.reference_energy, places=4)
