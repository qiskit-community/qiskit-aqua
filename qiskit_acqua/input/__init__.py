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

from .algorithminput import AlgorithmInput
from ._discover_input import (register_input,
                              deregister_input,
                              get_input_class,
                              get_input_instance,
                              get_input_configuration,
                              local_inputs)

__all__ = ['AlgorithmInput',
          'register_input',
          'deregister_input',
          'get_input_class',
          'get_input_instance',
          'get_input_configuration',
          'local_inputs']
