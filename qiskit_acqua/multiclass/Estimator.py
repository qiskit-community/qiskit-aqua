class Estimator:
    def fit(self, X, y):
        raise NotImplementedError( "Should have implemented this" )

    def decision_function(self, X):
        raise NotImplementedError( "Should have implemented this" )

