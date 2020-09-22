#imports


class AdaptVQE(GroundStateCalculation):  # same for VQEAdapt, ...
    def __init__(self, params_for_mapping):
        super().__init__(params_for_mapping)
    def compute_ground_state(driver) -> GroundStateCalculationResult:
        op = self._transform(driver)
        # different implementation similar to VQE
        result = None
        # construct GroundStateCalculationResult
        return ground_state_calculation_result
