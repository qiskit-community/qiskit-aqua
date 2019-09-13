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

"""Utilities for dict and json conversion."""

import numpy


def convert_dict_to_json(in_item):
    """
    Combs recursively through a list/dictionary and finds any non-json \
    compatible elements and converts them. E.g. complex ndarray's are \
    converted to lists of strings. Assume that all such elements are \
    stored in dictionaries!

    Args:
        in_item (dict or list): the input dict/list
    Returns:
        dict or list: Result in_item possibly modified
    """

    if in_item is None:
        return in_item

    nonjson_key_list = []
    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

        if isinstance(in_item[curkey], (list, dict)):
            # go recursively through nested list/dictionaries
            convert_dict_to_json(in_item[curkey])
        elif isinstance(in_item[curkey], numpy.ndarray):
            # ndarray's are not json compatible. Save the key.
            key_list.append(curkey)

        if isinstance(in_item, dict) and not isinstance(curkey, str):
            nonjson_key_list.append(curkey)

    # convert ndarray's to lists
    # split complex arrays into two lists because complex values are not
    # json compatible
    for curkey in key_list:
        if in_item[curkey].dtype == 'complex':
            in_item[curkey + '_ndarray_imag'] = numpy.imag(in_item[curkey]).tolist()
            in_item[curkey + '_ndarray_real'] = numpy.real(in_item[curkey]).tolist()
            in_item.pop(curkey)
        else:
            in_item[curkey] = in_item[curkey].tolist()

    # if dictionary key is not string, convert it to string
    for curkey in nonjson_key_list:
        in_item[str(curkey)] = in_item[curkey]
        del in_item[curkey]

    return in_item


def convert_json_to_dict(in_item):
    """
    Combs recursively through a list/dictionary that was loaded from json \
    and finds any lists that were converted from ndarray and converts them back \

    Args:
        in_item (dict or list): the input dict/list

    Returns:
        dict or list: Result in_item possibly modified
    """
    if in_item is None:
        return in_item

    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

            # flat these lists so that we can recombine back into a complex
            # number
            if '_ndarray_real' in curkey:
                key_list.append(curkey)
                continue

        if isinstance(in_item[curkey], (list, dict)):
            convert_json_to_dict(in_item[curkey])

    for curkey in key_list:
        curkey_root = curkey[0:-13]
        in_item[curkey_root] = numpy.array(in_item[curkey])
        in_item.pop(curkey)
        if curkey_root + '_ndarray_imag' in in_item:
            in_item[curkey_root] = in_item[curkey_root] + 1j * numpy.array(
                in_item[curkey_root + '_ndarray_imag'])
            in_item.pop(curkey_root + '_ndarray_imag')

    return in_item
