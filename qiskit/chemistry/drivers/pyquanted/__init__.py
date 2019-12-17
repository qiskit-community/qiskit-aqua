# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
PyQuante Installation
=====================
`PyQuante <https://github.com/rpmuller/pyquante2/>`__ is an open-source library for computational
chemistry. Qiskit Chemistry specifically requires PyQuante V2, also known as PyQuante2.
In order for Qiskit Chemistry to interface PyQuante and execute
PyQuante to extract the electronic structure information PyQuante2 must be installed and
discoverable on the system where Qiskit Chemistry is also installed.

Installing PyQuante2 according to the
`installation instructions <https://github.com/rpmuller/pyquante2/blob/master/README.md>`__ while
in the Python virtual environment where Qiskit Chemistry has also been installed will automatically
make PyQuante2 dynamically discovered by Qiskit Chemistry at run time. If you are not using
conda then alternatively you can git clone or download/unzip a zip of the repository and run pip
install off the setup.py that is there.

.. note::
    Like all the other drivers currently interfaced by Qiskit Chemistry,
    PyQuante2 provides enough intermediate data for Qiskit Chemistry to compute a molecule's ground
    state molecular energy.  However, unlike the other drivers, the data computed by PyQuante is
    not sufficient for Qiskit Chemistry to compute a molecule's dipole moment.  Therefore, PyQuante
    is currently the only driver interfaced by Qiskit Chemistry that does not allow for the
    computation of a molecule's dipole moment.

"""

from .pyquantedriver import PyQuanteDriver, BasisType

__all__ = ['PyQuanteDriver',
           'BasisType']
