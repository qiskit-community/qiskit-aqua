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

import setuptools

long_description = """<a href="https://qiskit.org/aqua" rel=nofollow>Qiskit Chemistry</a>
 is a set of quantum computing algorithms,
 tools and APIs for experimenting with real-world chemistry applications on near-term quantum devices."""

requirements = [
    "qiskit-aqua>=0.4.2",
    "numpy>=1.13,<1.16",
    "h5py",
    "psutil>=5",
    "jsonschema>=2.6,<2.7",
    "setuptools>=40.5.0"
]


setuptools.setup(
    name='qiskit-chemistry',
    version="0.4.3",  # this should match __init__.__version__
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
