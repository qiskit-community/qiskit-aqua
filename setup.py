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

long_description = """<a href="https://qiskit.org/aqua" rel=nofollow>Qiskit Aqua</a> is an extensible,
 modular, open-source library of quantum computing algorithms.
 Researchers can experiment with Aqua algorithms, on near-term quantum devices and simulators,
 and can also get involved by contributing new algorithms and algorithm-supporting objects,
 such as optimizers and variational forms. Qiskit Aqua is used by Qiskit Aqua Chemistry,
 Qiskit Aqua Artificial Intelligence, and Qiskit Aqua Optimization to experiment with real-world applications to quantum computing."""

requirements = [
    "qiskit-terra>=0.8.0,<0.9",
    "qiskit-ignis>=0.1.0,<0.2",
    "scipy>=0.19,!=0.19.1",
    "sympy>=1.3",
    "numpy>=1.13",
    "psutil>=5",
    "jsonschema>=2.6,<2.7",
    "scikit-learn>=0.20.0",
    "cvxopt",
    "dlx",
    "docplex",
    "fastdtw",
    "quandl",
    "setuptools>=40.1.0",
]

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
          "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit", "aqua", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name='qiskit-aqua',
    version=VERSION,
    description='Qiskit Aqua: An extensible library of quantum computing algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/qiskit-aqua',
    author='Qiskit Aqua Development Team',
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
    keywords='qiskit sdk quantum aqua',
    packages=setuptools.find_namespace_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    extras_require={
        'torch': ["torch; sys_platform != 'win32'"],
        'eda': ["pyeda; sys_platform != 'win32'"],
    }
)
