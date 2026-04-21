import numpy as np

class LinearPolicy:
    def __init__(self, weights):
        self.weights = weights

    def act(self, observation):
        return int(np.dot(self.weights, observation) > 0)