from qiskit import Aer
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover
import logging

import sys

import qiskit.aqua.utils.qprofile_utils
# addLoggingLevel('PROFILE', logging.DEBUG - 5)
# logging.getLogger(__name__).setLevel("PROFILE")
# logging.getLogger(__name__).trace('that worked')
# logging.PROFILE = 5

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
