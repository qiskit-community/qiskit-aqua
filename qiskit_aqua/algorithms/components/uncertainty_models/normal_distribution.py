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

from scipy.stats.distributions import norm
from .univariate_uncertainty_model import UnivariateUncertaintyModel


class NormalDistribution(UnivariateUncertaintyModel):

    CONFIGURATION = {
        'name': 'NORMAL_DISTRIBUTION',
        'description': 'Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'NormalDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'mu': {
                    'type': 'integer',
                    'default': 0,
                    'minimum': 0
                },
                'sigma': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
                'low': {
                    'type': 'integer',
                    'default': -1,
                },
                'high': {
                    'type': 'integer',
                    'default': 1,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, mu=0, sigma=1, low=-1, high=1):
        self.validate(locals())
        probabilities, _ = UnivariateUncertaintyModel.\
            pdf_to_probabilities(lambda x: norm.pdf(x, mu, sigma), low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high)
