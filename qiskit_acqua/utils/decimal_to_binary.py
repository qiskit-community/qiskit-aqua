# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

from numpy import binary_repr


def decimal_to_binary(decimal_val, max_num_digits=20, fractional_part_only=False):
    decimal_val_fractional_part = abs(decimal_val - int(decimal_val))
    current_binary_position_val = 1 / 2
    binary_fractional_part_digits = []
    while decimal_val_fractional_part > 0 and len(binary_fractional_part_digits) < max_num_digits:
        if decimal_val_fractional_part >= current_binary_position_val:
            binary_fractional_part_digits.append('1')
            decimal_val_fractional_part -= current_binary_position_val
        else:
            binary_fractional_part_digits.append('0')
        current_binary_position_val /= 2

    binary_repr_fractional_part = ''.join(binary_fractional_part_digits)

    if fractional_part_only:
        return binary_repr_fractional_part
    else:
        return binary_repr(int(decimal_val)) + '.' + binary_repr_fractional_part
