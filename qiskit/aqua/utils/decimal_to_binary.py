# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" decimal to binary """
from numpy import binary_repr


def decimal_to_binary(decimal_val, max_num_digits=20, fractional_part_only=False):
    """ decimal to binary """
    decimal_val_fractional_part = abs(decimal_val - int(decimal_val))
    current_binary_position_val = 1 / 2
    binary_fractional_part_digits = []
    while decimal_val_fractional_part >= 0 and len(binary_fractional_part_digits) < max_num_digits:
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
