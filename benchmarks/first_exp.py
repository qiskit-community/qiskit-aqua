from qiskit import Aer
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover
import logging
import sys

import qiskit.aqua.utils.qprofile_utils

logging.basicConfig(stream=sys.stdout, level=logging.MEMPROFILE)

sat_cnf = """
c Example DIMACS 3-sat
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""

backend = Aer.get_backend('qasm_simulator')
oracle = LogicalExpressionOracle(sat_cnf)
algorithm = Grover(oracle)
result = algorithm.run(backend)
print(result["result"])
