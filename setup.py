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

long_description="""<a href="https://qiskit.org/aqua/chemistry" rel=nofollow>Qiskit Aqua Chemistry</a> 
 is a set of quantum computing algorithms, 
 tools and APIs for experimenting with real-world chemistry applications on near-term quantum devices."""
    

requirements = [
    "qiskit-aqua>=0.3.0",
    "qiskit>=0.6.1,<0.7",
    "numpy>=1.13",
    "h5py",
    "psutil>=5",
    "jsonschema>=2.6,<2.7",
    "pyobjc-core; sys_platform == 'darwin'",
    "pyobjc-framework-Cocoa; sys_platform == 'darwin'"
]

setuptools.setup(
    name='qiskit-aqua-chemistry',
    version="0.3.0",  # this should match __init__.__version__
    description='Qiskit Aqua Chemistry: Experiment with chemistry applications on a quantum machine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/aqua-chemistry',
    author='Qiskit Aqua Chemistry Development Team',
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
    keywords='qiskit sdk quantum aqua chemistry',
    packages=setuptools.find_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    entry_points = {
        'console_scripts': [
                'qiskit_aqua_chemistry_cmd=qiskit_aqua_chemistry.command_line:main'
        ],
        'gui_scripts': [
                'qiskit_aqua_chemistry_ui=qiskit_aqua_chemistry.ui.command_line:main'
        ]
    }
)