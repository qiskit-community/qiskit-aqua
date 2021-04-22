# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains the Deprecation msg methods."""

import warnings
from typing import Optional


class AquaObjects:
    """ Aqua objects """
    def __init__(self):
        self._packages = set()
        self._classes = set()
        self._variables = set()

    def add_package(self, package: str) -> None:
        """ add a package """
        self._packages.add(package)

    def package_exists(self, package: str) -> bool:
        """ tests if package was added """
        return package in self._packages

    def add_class(self, fullname: str) -> None:
        """ add a class """
        self._classes.add(fullname)

    def class_exists(self, fullname: str) -> bool:
        """ tests if class was added """
        return fullname in self._classes

    def add_variable(self, fullname: str) -> None:
        """ add a variable """
        self._variables.add(fullname)

    def variable_exists(self, fullname: str) -> bool:
        """ tests if variable was added """
        return fullname in self._variables


_AQUA_OBJECTS = AquaObjects()


def warn_package(aqua_package: str,
                 package: Optional[str] = None,
                 library: Optional[str] = None,
                 stacklevel: int = 2) -> None:
    """ emit package deprecation warning """
    if _AQUA_OBJECTS.package_exists(aqua_package):
        return

    _AQUA_OBJECTS.add_package(aqua_package)
    msg = f'The package qiskit.{aqua_package} is deprecated.'
    if package:
        msg += f' It was moved/refactored to {package}'
    if library:
        msg += f' (pip install {library}).'

    msg += ' For more information see ' \
           '<https://github.com/Qiskit/qiskit-aqua/blob/main/README.md#migration-guide>'

    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def warn_class(fullname: str, new_fullname: str, library: str,
               stacklevel: int = 2) -> None:
    """ emit class deprecation warning """
    if _AQUA_OBJECTS.class_exists(fullname):
        return

    _AQUA_OBJECTS.add_class(fullname)
    msg = f'The class qiskit.{fullname} is deprecated. It was moved/refactored to ' \
          f'{new_fullname} (pip install {library}). For more information see ' \
          f'<https://github.com/Qiskit/qiskit-aqua/blob/main/README.md#migration-guide>'

    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def warn_variable(fullname: str, new_fullname: str, library: str,
                  stacklevel: int = 2) -> None:
    """ emit class deprecation warning """
    if _AQUA_OBJECTS.variable_exists(fullname):
        return

    _AQUA_OBJECTS.add_variable(fullname)
    msg = f'The variable qiskit.{fullname} is deprecated. It was moved/refactored to ' \
          f'{new_fullname} (pip install {library}). For more information see ' \
          f'<https://github.com/Qiskit/qiskit-aqua/blob/main/README.md#migration-guide>'

    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
