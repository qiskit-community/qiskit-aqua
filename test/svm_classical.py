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

from qiskit_aqua.utils.multiclass.data_preprocess import *
from qiskit_aqua.svm_qkernel.data_preprocess import *
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import run_algorithm
from qiskit_aqua.svm_qkernel.data_preprocess import *
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import run_algorithm

# binary:
# sample_Total, training_input, test_input, class_labels = ad_hoc_data(
#     training_size=20, test_size=10, n=2, gap=0.3, PLOT_DATA=False) # n=2 is the dimension of each data point
# total_array, label_to_labelclass = get_points(test_input, class_labels)

# multiclass:
sample_Total, training_input, test_input, class_labels = \
Wine(training_size=40, test_size=10, n=2, # 2 is the dimension of each data point
            PLOT_DATA=False)
total_array, label_to_labelclass = get_points(test_input, class_labels)


params = {
    'problem': {'name': 'svm_classification'},
    'algorithm': {
        'name': 'SVM_Classical',
        'print_info': True
    }
}

algo_input = get_input_instance('SVMInput')
algo_input.training_dataset = training_input
algo_input.test_dataset = test_input
algo_input.datapoints = total_array
result = run_algorithm(params, algo_input)
print(result)
