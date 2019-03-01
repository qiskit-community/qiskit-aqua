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

long_description="""<a href="https://qiskit.org/aqua" rel=nofollow>Qiskit Aqua</a> is an extensible,
 modular, open-source library of quantum computing algorithms.
 Researchers can experiment with Aqua algorithms, on near-term quantum devices and simulators,
 and can also get involved by contributing new algorithms and algorithm-supporting objects,
 such as optimizers and variational forms. Qiskit Aqua is used by Qiskit Aqua Chemistry,
 Qiskit Aqua Artificial Intelligence, and Qiskit Aqua Optimization to experiment with real-world applications to quantum computing."""

requirements = [
    "qiskit-terra>=0.8.0,<0.9",
    "scipy>=0.19,!=0.19.1",
    "sympy>=1.3",
    "numpy>=1.13,<1.16",
    "psutil>=5",
    "jsonschema>=2.6,<2.7",
    "scikit-learn>=0.20.0",
    "cvxopt",
    "setuptools>=40.5.0",
    "pyobjc-core; sys_platform == 'darwin'",
    "pyobjc-framework-Cocoa; sys_platform == 'darwin'",
    "dlx",
    "pyeda"
]

setuptools.setup(
    name='qiskit-aqua',
    version="0.4.2",  # this should match __init__.__version__
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
    entry_points={
        'console_scripts': [
                'qiskit_aqua_cmd=qiskit_aqua_cmd.command_line:main'
        ],
        'gui_scripts': [
                'qiskit_aqua_ui=qiskit_aqua_ui.run.command_line:main',
                'qiskit_aqua_browser=qiskit_aqua_ui.browser.command_line:main'
        ]
    }
)
