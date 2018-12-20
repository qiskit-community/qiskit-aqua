# Qiskit Chemistry

## Electronic structure driver for Gaussian 16

Gaussian 16 is a commercial program for computational chemistry, see http://gaussian.com/gaussian16/

The driver accesses electronic structure information from Gaussian 16 via the Gaussian supplied open-source 
interfacing code available from Gaussian at http://www.gaussian.com/interfacing/

In the folder here called 'gauopen' the Python part of the above interfacing code, as needed by Qiskit Chemistry,
has been made available. It is licensed under a [Gaussian Open-Source Public License](./gauopen/LICENSE.txt) which can
also be found in this folder.

Part of this interfacing code, qcmatrixio.F, requires compilation to a Python native extension, however
Qiskit Chemistry comes with pre-built binaries for most common platforms. If there is no pre-built binary
matching your platform then it will be necessary to compile this file as per the instructions below.  

### Compiling the Fortran interfacing code

If no pre-built native extension binary, as supplied with Qiskit Chemistry, works for your platform then
to use the Gaussian driver on your machine the Fortran file qcmatrixio.F must be compiled into object code that can
be used by Python. This is accomplished using f2py, which is part of numpy https://docs.scipy.org/doc/numpy/f2py/

Change directory to gauopen and from your Python environment use the following command. You will need a supported
Fortran compiler installed. On MacOS you may have to download GCC and the GFortran Compiler source and compile it first
if you do not a suitable Fortran compiler installed. With Linux you may be able to download one via your distribution's
installer.

>f2py -c -m qcmatrixio qcmatrixio.F

The following can be used with the Intel Fortran e.g on Microsoft Windows platform. On Windows with the Intel Fortran
compiler the environment can be setup with _ifortvars.bat_ e.g. `ifortvars -arch intel64`. 

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

On Linux/Mac you will find a qcmatrixio.so file such as `qcmatrixio.cpython-36m-x86_64-linux-gnu.so` is created and on 
Windows it could be something like this `qcmatrixio.cp36-win_amd64.pyd`

### Ensure G16 is in the Path and the environment setup for G16

You should also make sure the g16 executable can be run from a command line. Make sure it's in the path and appropriate
exports such as GAUSS_EXEDIR etc have been done as per Gaussian installation instructions which may be found here:
[http://gaussian.com/techsupport/#install](http://gaussian.com/techsupport/#install).


### MacOS X notes

As an example, if your account is using the bash shell on a macOS X machine, you can edit the `.bash_profile` file
in your account's home directory and add the following lines:
```
export GAUSS_SCRDIR=~/.gaussian
export g16root=/Applications
alias enable_gaussian='. $g16root/g16/bsd/g16.profile'
```
The above assumes that Gaussian 16 was placed in the /Applications folder and that ~/.gaussian is the full path to
the selected scratch folder, where Gaussian 16 stores its temporary files. 
 
Now, before executing Qiskit Chemistry, to use it with Gaussian, you will have to run the `enable_gaussian` command.
This, however, may generate the following error:
```
bash: ulimit: open files: cannot modify limit: Invalid argument
```
While this error is not harmful, you might want to suppress it, which can be done by entering the following sequence
of commands at the command line:
```
echo kern.maxfiles=65536 | sudo tee -a /etc/sysctl.conf
echo kern.maxfilesperproc=65536 | sudo tee -a /etc/sysctl.conf
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=65536
ulimit -n 65536 65536 
```
as well as finally adding the following line to the `.bash_profile` file in your account's home directory:
```
ulimit -n 65536 65536
```

## Input file example

To configure a molecule, on which to do a chemistry experiment with Qiskit Chemistry, create a GAUSSIAN section
in the input file as per the example below. Here the molecule, basis set and other options are specified according
to the GAUSSIAN control file, so blank lines, control line syntax etc. according to Gaussian should be followed.
```
&GAUSSIAN
# rhf/sto-3g scf(conventional)

h2 molecule

0 1
H   0.0  0.0    0.0
H   0.0  0.0    0.74

&END
```
