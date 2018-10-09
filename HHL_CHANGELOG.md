# Changelog for HHL module

Here all the changes done to the qiskit aqua module are documented.

## Utils

* Added `random_hermitian`, `random_non_hermitian` methods to
    `utils/generate_random_matrix.py`. Containing additional hidden methods
    `random_diag`, `limit_paulis`, `limit_entries`. `random_hermitian,
    random_non_hermitian` are methods for matrix generation with specified
    condition number or eigen/singular values.
* Added `cnx_na.py` for preforming a multiple controlled X gate without using
    ancillary qubits.
* Added `cnu1.py`, `cnu3.py` for multiple controlled U1 and U3 operations.
* Implement
