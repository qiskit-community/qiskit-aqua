from qiskit_acqua import Operator, run_algorithm, get_algorithm_instance
from qiskit_acqua.input import get_input_instance
from qiskit_acqua.ising import graphpartition
import numpy as np


# w = maxcut.parse_gset_format('sample.maxcut')
# qubitOp, offset = maxcut.get_maxcut_qubitops(w)
# algo_input = get_input_instance('EnergyInput')
# algo_input.qubit_op = qubitOp

algo_input = get_input_instance('EnergyInput')
if True:
    np.random.seed(100)
    w = graphpartition.random_graph(4, edge_prob=0.8, weight_range=10)
    qubitOp, offset = graphpartition.get_graphpartition_qubitops(w)
    algo_input.qubit_op = qubitOp
print(w)


to_be_tested_algos = ['ExactEigensolver', 'CPLEX', 'VQE']
operational_algos = []
for algo in to_be_tested_algos:
    try:
        get_algorithm_instance(algo)
        operational_algos.append(algo)
    except:
        print("{} is unavailable, please check your setting.".format(algo))
print(operational_algos)


if 'ExactEigensolver' not in operational_algos:
    print("ExactEigensolver is not in operational algorithms.")
else:
    algorithm_cfg = {
        'name': 'ExactEigensolver',
    }

    params = {
        'problem': {'name': 'ising'},
        'algorithm': algorithm_cfg
    }
    result = run_algorithm(params,algo_input)
    # print('objective function:', maxcut.maxcut_obj(result, offset))
    x = graphpartition.sample_most_likely(len(w), result['eigvecs'][0])

    print('solution:', graphpartition.get_graph_solution(x))
    print('solution objective:', graphpartition.objective_value(x, w))




# if 'VQE' not in operational_algos:
#     print("VQE is not in operational algorithms.")
# else:
#     algorithm_cfg = {
#         'name': 'VQE',
#         'operator_mode': 'matrix'
#     }
#
#     optimizer_cfg = {
#         'name': 'L_BFGS_B',
#         'maxfun': 6000
#     }
#
#     var_form_cfg = {
#         'name': 'RYRZ',
#         'depth': 3,
#         'entanglement': 'linear'
#     }
#
#     params = {
#         'problem': {'name': 'ising'},
#         'algorithm': algorithm_cfg,
#         'optimizer': optimizer_cfg,
#         'variational_form': var_form_cfg,
#         'backend': {'name': 'local_statevector_simulator'}
#     }
#
#     result = run_algorithm(params,algo_input)
#
#     x = maxcut.sample_most_likely(len(w), result['eigvecs'][0])
#     print('energy:', result['energy'])
#     print('time:', result['eval_time'])
#     print('maxcut objective:', result['energy'] + offset)
#     print('solution:', maxcut.get_graph_solution(x))
#     print('solution objective:', maxcut.maxcut_value(x, w))
