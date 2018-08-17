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

from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua.svm_qkernel.svm_qkernel_binary import SVM_QKernel_Binary
from qiskit_aqua.svm_qkernel.svm_qkernel_multiclass import SVM_QKernel_Multiclass


class SVM_QKernel(QuantumAlgorithm):
    """
    The qkernel interface.
    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """
    SVM_QKERNEL_CONFIGURATION = {
        'name': 'SVM_QKernel',
        'description': 'SVM_QKernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_QKernel_schema',
            'type': 'object',
            'properties': {
                'multiclass_alg': {
                    'type': 'string',
                    'default': 'all_pairs'
                },
                'print_info': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'problems': ['svm_classification']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_QKERNEL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        SVMQK_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        is_multiclass = (len(algo_input.training_dataset.keys()) > 2)
        if is_multiclass:
            self.instance = SVM_QKernel_Multiclass()
        else:
            self.instance = SVM_QKernel_Binary()
        self.instance.init_args(algo_input.training_dataset, algo_input.test_dataset, algo_input.datapoints,
                                SVMQK_params.get('print_info'), SVMQK_params.get('multiclass_alg'), self._backend,
                                self._execute_config['shots'], self._random_seed)

    def run(self):
        self.instance.run()
        return self.instance.ret
