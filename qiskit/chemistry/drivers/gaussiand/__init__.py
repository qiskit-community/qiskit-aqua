# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gaussian™ 16 Installation
=========================

`Gaussian™ 16 <http://gaussian.com/gaussian16/>`__ is a commercial program for
computational chemistry. This chemistry driver accesses electronic structure information
from Gaussian™ 16 via the Gaussian-supplied open-source
`interfacing code <http://www.gaussian.com/interfacing/>`__.

You should follow the installation instructions that come with your Gaussian™ 16 product.
Installation instructions can also be found online in
`Gaussian product installation support <http://gaussian.com/techsupport/#install]>`__.

Following the installation make sure the Gaussian™ 16 executable, `g16`, can be run from the
command line environment where you will be running Python and Qiskit. For example
verifying that the `g16` executable is reachable via the system environment path,
and appropriate exports, such as `GAUSS_EXEDIR`, have been configured as per
`Gaussian product installation support <http://gaussian.com/techsupport/#install]>`__.

Gaussian™ 16 Interfacing Code
-----------------------------


In the :mod:`gauopen` folder the Python part of the above interfacing code,
as needed by Qiskit's chemistry modules, has been made available. It is licensed under a
`Gaussian Open-Source Public License
<https://github.com/Qiskit/qiskit-aqua/blob/master/qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt>`_.

Part of this interfacing code --- specifically, the Fortran file `qcmatrixio.F` --- requires
compilation to a Python native extension. However, Qiskit comes with pre-built binaries
for most common platforms. If there is no pre-built binary matching your platform, then it will be
necessary to compile this file as per the instructions below.

Compiling the Fortran Interfacing Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If no prebuilt native extension binary, as supplied with Qiskit, works for your
platform, then to use the Gaussian™ 16 driver on your machine, the Fortran file `qcmatrixio.F`
must be compiled into object code that can be used by Python. This is accomplished using the
`Fortran to Python Interface Generator (F2PY) <https://docs.scipy.org/doc/numpy/f2py/>`__,
which is part of the `NumPy <http://www.numpy.org/>`__ Python library.
Specifically, on your command prompt window, change directory to the
`qiskit/chemistry/drivers/gaussiand/gauopen` directory inside the Qiskit
installation directory, and while in the Python environment created for Aqua and the chemistry
module, invoke `f2py` on `qcmatrixio.F` as explained below.


Apple macOS and Linux
~~~~~~~~~~~~~~~~~~~~~

The full syntax of the `f2py` command on macOS and Linux is as follows:

.. code:: sh

    f2py -c -m qcmatrixio qcmatrixio.F

This command will generate a file with name prefix `qcmatrixio` and extension `.so`, for example
`qcmatrixio.cpython-36m-x86_64-linux-gnu.so`.
In order for the command above to work and such file to be generated, you will need a supported
Fortran compiler installed. On macOS, you may have to download the
`GNU Compiler Collection (GCC) <https://gcc.gnu.org/>`__ and, in particular, the
`GFortran Compiler <https://gcc.gnu.org/fortran/>`__ source and compile it first,
if you do not a suitable Fortran compiler already installed.
On Linux you may be able to download and install a supported Fortran compiler via your
distribution's installer.

.. topic:: Special Notes for macOS X

    If your account is using the bash shell on a macOS X machine, you can edit the
    `.bash_profile` file in your home directory and add the following lines:

    .. code:: sh

        export GAUSS_SCRDIR=~/.gaussian
        export g16root=/Applications
        alias enable_gaussian='. $g16root/g16/bsd/g16.profile'

    The above assumes that the application Gaussian™ 16 was placed in the `/Applications` folder
    and that `~/.gaussian` is the full path to
    the selected scratch folder, where Gaussian™ 16 stores its temporary files.

    Now, before Qiskit can properly interface Gaussian™ 16, you will have to run the
    `enable_gaussian` command defined above.  This, however, may generate the following error:

    .. code:: sh

        bash: ulimit: open files: cannot modify limit: Invalid argument

    While this error is not harmful, you might want to suppress it, which can be done by entering
    the following sequence of commands on the command line:

    .. code:: sh

        echo kern.maxfiles=65536 | sudo tee -a /etc/sysctl.conf
        echo kern.maxfilesperproc=65536 | sudo tee -a /etc/sysctl.conf
        sudo sysctl -w kern.maxfiles=65536
        sudo sysctl -w kern.maxfilesperproc=65536
        ulimit -n 65536 65536

    as well as finally adding the following line to the `.bash_profile` file in your account's
    home directory:

    .. code:: sh

        ulimit -n 65536 65536

    At the end of this configuration, the `.bash_profile` in your account's home directory
    should have a section in it like in the following script snippet:

    .. code:: sh

        # Gaussian 16
        export GAUSS_SCRDIR=~/.gaussian
        export g16root=/Applications
        alias enable_gaussian='. $g16root/g16/bsd/g16.profile'
        ulimit -n 65536 65536


Microsoft Windows
~~~~~~~~~~~~~~~~~

The following steps can be used with the Intel Fortran compiler on the Microsoft Windows platform:

1. Set up the environment by running the Intel Fortran compiler batch program `ifortvars.bat`
   as follows:

   .. code:: sh

       ifortvars -arch intel64

2. Then, in this environment, issue the following command from within the `gauopen` directory:

   .. code:: sh

       f2py -c --fcompiler=intelvem -m qcmatrixio qcmatrixio.F

   Upon successful execution, the `f2py` command above will generate a file with name prefix
   `qcmatrixio` and extension `.so`, for example `qcmatrixio.cp36-win_amd64.pyd`.  However,
   in order for the `f2py` command above to work, `#ifdef` may need to be manually edited if it
   is not recognized or supported during the processing of the `f2py` command above.  For
   example, with `f2py` from Intel Visual Fortran Compiler with Microsoft Visual Studio, the
   following code snippet originally shows two occurrences of the line
   `Parameter (Len12D=8,Len4D=8)`, as shown next:

   .. code::

       `#ifdef` USE_I8
           Parameter (Len12D=8,Len4D=8)
       `#else`
           Parameter (Len12D=4,Len4D=4)
       `#endif`

   This may need to be simplified by deleting the first three lines and the last line,
   leaving just the fourth line, as follows:

   .. code::

       Parameter (Len12D=4,Len4D=4)

"""

from .gaussiandriver import GaussianDriver
from .gaussian_forces_driver import GaussianForcesDriver
from .gaussian_log_driver import GaussianLogDriver
from .gaussian_log_result import GaussianLogResult

__all__ = ['GaussianDriver',
           'GaussianForcesDriver',
           'GaussianLogDriver',
           'GaussianLogResult']
