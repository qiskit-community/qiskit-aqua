{
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 50,
        "expansion_mode": "suzuki",
        "expansion_order": 2,
        "negative_evals": (True, False),
        "num_ancillae": (4, 5, 6, 7, 8, 9, 10)
    },
    "reciprocal": {
        "name": "LOOKUP",
        "lambda_min": 0.9,
        "pat_length": (4, 5)
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    },
    "input": "test_objects/conf_specified.pkl"
}
