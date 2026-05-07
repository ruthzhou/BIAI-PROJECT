import numpy as np


class LinearPolicy:

    def __init__(self, weights):

        self.weights = np.array(weights)

    def act(self, observation):

        action_value = np.dot(
            self.weights,
            observation
        )

        return int(action_value > 0)