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
        "negative_evals": False,
        "num_ancillae": (6, 7, 8)
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
        "n": 2,
        "type": "generate",
        "test_set": "general_condition",
        "repetition": range(10),
        "condition": range(10, 110, 10)
    }
}
