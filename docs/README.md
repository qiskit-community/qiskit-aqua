# Automatically Generating the Qiskit Chemistry Documentation

1. Make sure you have `Sphinx` >= 1.7.6, `sphinxcontrib-fulltoc` >= 1.2.0, and `sphinxcontrib-websupport` >= 1.1.0 installed
   in the same Python environment where you have `qiskit-chemistry` installed.
2. From the `docs` folder of `qiskit-chemistry`, issue the following commands:

   - `make clean`
   - `sphinx-apidoc -f -o . ..`
   - `make html`
  
