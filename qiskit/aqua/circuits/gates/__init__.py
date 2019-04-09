# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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

from .multi_control_u1_gate import mcu1
from .multi_control_u3_gate import mcu3
from .multi_control_toffoli_gate import mct
from .multi_control_multi_target_gate import mcmt
from .boolean_logical_gates import logical_and, logical_or
from .controlled_hadamard_gate import ch
from .controlled_ry_gates import cry, mcry

__all__ = [
    'mcu1',
    'mcu3',
    'mct',
    'mcmt',
    'logical_and',
    'logical_or',
    'ch',
    'cry',
    'mcry',
]
