from qiskit_acqua.multiclass.Estimator import Estimator
from sklearn.svm import LinearSVC

class LinearSVC_Estimator(Estimator):
    def __init__(self):
        self._estimator = LinearSVC(random_state=0)

    def fit(self, X, y):
        self._estimator.fit(X, y)

    def decision_function(self, X):
        return self._estimator.decision_function(X)

