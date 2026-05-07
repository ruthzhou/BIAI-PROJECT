import numpy as np


def uniform_crossover(parents):

    offspring = []

    for i in range(0, len(parents), 2):

        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]

        mask = np.random.randint(
            0,
            2,
            size=len(parent1)
        )

        child = np.where(
            mask,
            parent1,
            parent2
        )

        offspring.append(child)

    return np.array(offspring)