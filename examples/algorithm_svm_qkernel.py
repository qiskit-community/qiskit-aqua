import os
import sys
algo_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,algo_directory)
from algorithm_svm_datasets import *
from qiskit_acqua.svm.data_preprocess import *
from qiskit_acqua.input import get_input_instance
from qiskit_acqua import run_algorithm


num_of_qubits=2
sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=20, test_size=10, n=num_of_qubits, gap=0.3, PLOT_DATA=False)
total_array, label_to_labelclass = get_points(test_input, class_labels)


params = {
    'problem': {'name': 'svm_classification'},
    'backend': {'name': 'local_qasm_simulator', 'shots':1000},
    'algorithm': {
        'name': 'SVM_QKernel',
        'num_of_qubits': num_of_qubits
    }
}

algo_input = get_input_instance('SVMInput')
algo_input.training_dataset  = training_input
algo_input.test_dataset = test_input
algo_input.datapoints = total_array
result = run_algorithm(params,algo_input)
print(result)

