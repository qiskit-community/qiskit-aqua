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
from setuptools.dist import Distribution
from distutils.command.build import build
import os
import shutil
import sys

with open('README.md', 'r') as fh:
    long_description = fh.read()
    
with open('requirements.txt', 'r') as fh:
    requirements = fh.readlines()

# Gaussian files include
class GaussianBuild(build):
    
    _GAUOPEN_DIR = 'qiskit_acqua_chemistry/drivers/gaussiand/gauopen'
    _PLATFORM_DIRS = {'darwin': 'macosx_x86_64',
                      'linux': 'manylinux1_x86_64',
                      'win32': 'win_amd64',
                      'cygwin': 'win_amd64'}
    def run(self):
        super().run()
        if sys.platform not in GaussianBuild._PLATFORM_DIRS:
            print("WARNING: Missing Gaussian binaries for '{}'. It will continue.".format(sys.platform))
            return
        
        platform_dir = GaussianBuild._PLATFORM_DIRS[sys.platform]
        gaussian_binaries_dir = os.path.join(GaussianBuild._GAUOPEN_DIR,platform_dir)
        for dirpath, dirnames, filenames in os.walk(gaussian_binaries_dir):
            dirpath = os.path.normpath(dirpath)
            source_files = [os.path.join(dirpath,f) for f in filenames if not f.endswith('.DS_Store')]
            for source_file in source_files:
                target_file = source_file
                components = source_file.split(os.sep)
                try:
                    components.remove(platform_dir)
                    target_file = os.path.join(*components)
                except:
                    pass
                
                target_file = os.path.join(self.build_lib,target_file)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copyfile(source_file,target_file)
                

class BinaryDistribution(Distribution):
    """Distribution always forces binary package with platform name"""
    def has_ext_modules(self):
        return True


setuptools.setup(
    name='qiskit_acqua_chemistry',
    version="0.1.0",  # this should match __init__.__version__
    description='QISKit ACQUA Chemistry',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.ibm.com/IBMQuantum/qiskit-acqua-chemistry',
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
    keywords=['ibm', 'qiskit', 'sdk', 'quantum', 'acqua', 'chemistry'],
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
    },
    cmdclass={
        'build': GaussianBuild,
    },
    distclass=BinaryDistribution
)