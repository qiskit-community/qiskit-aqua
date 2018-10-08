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
        "num_ancillae": (4, 5, 6, 7, 8, 9)
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
    "input": {
        "n": (2, 4),
        "type": "generate",
        "test_set": "general_condition",
        "condition": range(10, 110, 10)
    }
}
