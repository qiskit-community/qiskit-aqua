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

import logging

from qiskit_aqua.algorithms.many_sample.qsvm.svm_qkernel_binary import SVM_QKernel_Binary
from qiskit_aqua.algorithms.many_sample.qsvm.svm_qkernel_multiclass import SVM_QKernel_Multiclass
from qiskit_aqua import (QuantumAlgorithm, get_multiclass_extension_instance)

logger = logging.getLogger(__name__)


class SVM_QKernel(QuantumAlgorithm):
    """
    The qkernel interface.
    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """
    SVM_QKERNEL_CONFIGURATION = {
        'name': 'QSVM.Kernel',
        'description': 'SVM_QKernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_QKernel_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'depends': ['multiclass_extension'],
        'problems': ['svm_classification'],
        'defaults': {
            'multiclass_extension': {
                'name': 'AllPairs',
                'estimator': 'RBF_SVC_Estimator'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_QKERNEL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        SVMQK_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)

        is_multiclass = (len(algo_input.training_dataset.keys()) > 2)
        if is_multiclass:
            multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION)
            multiclass_extension = get_multiclass_extension_instance(multiclass_extension_params['name'])
            multiclass_extension_params['params'] = [self._backend, self._execute_config['shots'], self._random_seed] # we need to set this explicitly for quantum version
            multiclass_extension.init_params(multiclass_extension_params)
            # checking the options:
            estimator = multiclass_extension_params.get('estimator', None)
            if estimator == None:
                logger.debug("You did not provide the estimator, which is however required!")
            if estimator not in ["QKernalSVM_Estimator"]:
                logger.debug("You should use one of the qkernel estimators")
            logger.debug("We will apply the multiclass classifcation:" + multiclass_extension_params['name'])

            self.instance = SVM_QKernel_Multiclass(multiclass_extension)
        else:
            logger.debug("We will apply the binary classifcation and ignore all options related to the multiclass")
            self.instance = SVM_QKernel_Binary()
        self.instance.init_args(algo_input.training_dataset, algo_input.test_dataset, algo_input.datapoints, SVMQK_params.get('multiclass_alg'), self._backend, self._execute_config['shots'], self._random_seed)

    def run(self):
        self.instance.run()
        return self.instance.ret
