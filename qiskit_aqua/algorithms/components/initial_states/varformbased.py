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


class VarFormBased:
    """An initial state derived from a variational form.
       Intended to be used programmatically only.
    """

    def __init__(self):
        self._var_form = None
        self._var_form_params = None

    def init_args(self, var_form, params):
        if not var_form.num_parameters == len(params):
            raise RuntimeError('Incompatible parameters provided.')
        self._var_form = var_form
        self._var_form_params = params

    def construct_circuit(self, mode, register=None):
        if mode == 'vector':
            raise RuntimeError('Initial state based on variational form does not support vector mode.')
        elif mode == 'circuit':
            return self._var_form.construct_circuit(self._var_form_params)
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
