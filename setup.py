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

long_description="""<a href="https://qiskit.org/acqua/chemistry" rel=nofollow>QISKit ACQUA Chemistry</a> 
 is a set of quantum computing algorithms, 
 tools and APIs for experimenting with real-world chemistry applications on near-term quantum devices."""
    

requirements = [
    "qiskit-acqua",
    "qiskit>=0.5.4",
    "numpy>=1.13,<1.15",
    "h5py",
    "psutil",
    "jsonschema",
    "pyobjc-core; sys_platform == 'darwin'",
    "pyobjc-framework-Cocoa; sys_platform == 'darwin'"
]

setuptools.setup(
    name='qiskit-acqua-chemistry',
    version="0.1.0",  # this should match __init__.__version__
    description='QISKit ACQUA Chemistry: Experiment with chemistry applications on a quantum machine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/QISKit/qiskit-acqua-chemistry',
    author='QISKit ACQUA Chemistry Development Team',
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
    keywords='qiskit sdk quantum acqua chemistry',
    packages=setuptools.find_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    entry_points = {
        'console_scripts': [
                'qiskit_acqua_chemistry_cmd=qiskit_acqua_chemistry.command_line:main'
        ],
        'gui_scripts': [
                'qiskit_acqua_chemistry_ui=qiskit_acqua_chemistry.ui.command_line:main'
        ]
    }
)