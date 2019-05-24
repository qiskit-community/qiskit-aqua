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

import setuptools
import inspect
import sys
import os

long_description = """<a href="https://qiskit.org/aqua" rel=nofollow>Qiskit Chemistry</a>
 is a set of quantum computing algorithms,
 tools and APIs for experimenting with real-world chemistry applications on near-term quantum devices."""

requirements = [
    "qiskit-aqua>=0.5.2",
    "numpy>=1.13",
    "h5py",
    "psutil>=5",
    "jsonschema>=2.6,<2.7",
    "networkx>=2.2",
    "pyscf; sys_platform != 'win32'",
    "setuptools>=40.1.0"
]

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
          "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit", "chemistry", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name='qiskit-chemistry',
    version=VERSION,
    description='Qiskit Chemistry: Experiment with chemistry applications on a quantum machine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/qiskit-chemistry',
    author='Qiskit Chemistry Development Team',
    author_email='qiskit@us.ibm.com',
    license='Apache-2.0',
    classifiers=(
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"
    ),
    keywords='qiskit sdk quantum chemistry',
    packages=setuptools.find_namespace_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    entry_points={
        'qiskit.aqua.pluggables': [
            'HartreeFock = qiskit.chemistry.aqua_extensions.components.initial_states:HartreeFock',
            'UCCSD = qiskit.chemistry.aqua_extensions.components.variational_forms:UCCSD',
        ],
    },
)
