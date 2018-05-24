# IBM Quantum Library for Chemistry

## Electronic structure driver for Gaussian 16

Gaussian 16 is a commercial program for computational chemistry, see http://gaussian.com/gaussian16/

The driver accesses the electronic structure from Gaussian 16 via the Gaussian supplied open-source interfacing code
available from Gaussian at http://www.gaussian.com/interfacing/

In the folder here 'gauopen' the Python part of the above interfacing code needed by qiskit_acqua_chemistry has been made available
here. It is licensed under a [Gaussian Open-Source Public License](./gauopen/LICENSE.txt) which can also be found in
this folder.

### Compile the Fortran interfacing code

To use the Gaussian driver on your machine the Fortran file qcmatrixIO.F must be compiled into object code that can
be used by python. This is accomplished using f2py, which is part of numpy https://docs.scipy.org/doc/numpy/f2py/

Change directory to gauopen and from your python environment use the following command. You will need a supported
Fortran compiler installed. On MacOS you may have to download GCC and the GFortan Compiler source and compiler it first
if you do not a suitable Fortran compiler installed. With Linux you may be able to download one via your distribution's
installer.

>f2py -c -m qcmatrixio qcmatrixio.F

The following can be used with the Intel Fortran e.g on Microsoft Windows platform

>f2py -c --fcompiler=intelvem -m qcmatrixio qcmatrixio.F

Note: #ifdef may need to be manually edited if its not recognised/supported during the f2py processing.
E.g on Windows with f2py using Intel Visual Fortran Compiler with Microsoft Visual Studio two occurrences of 
```
#ifdef USE_I8
      Parameter (Len12D=8,Len4D=8)
#else
      Parameter (Len12D=4,Len4D=4)
#endif
```
may need to be simplified by deleting the first three and last line above, leaving just the fourth
```
      Parameter (Len12D=4,Len4D=4)
```

On Linux/Mac you will find a file such as qcmatrixio.so is created and on Windows it could be something like this
qcmatrixio.cp36-win_amd64.pyd

### Ensure G16 is in the Path and the environment setup for G16

You should also make sure the g16 executable can be run from a command line. Make sure it's in the path and appropriate
exports such as GAUSS_EXEDIR etc have been done as per Gaussian installation instructions which may be found here
http://gaussian.com/techsupport/#install

## Input file example
To configure a molecule on which to do a chemistry experiment with qiskit_acqua_chemistry create a GAUSSIAN section in the input file
as per the example below. Here the molecule, basis set and other options are specified according to GAUSSIAN control
file, so blank lines, control line syntax etc according to Gaussian should be followed.
```
&GAUSSIAN
# rhf/sto-3g scf(conventional)

h2 molecule

0 1
H   0.0  0.0    0.0
H   0.0  0.0    0.74

&END
```
