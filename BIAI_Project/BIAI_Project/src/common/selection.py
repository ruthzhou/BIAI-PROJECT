import numpy as np


def tournament_selection(
    population,
    fitness,
    k=3
):

    selected = []

    for _ in range(len(population)):

        indices = np.random.choice(
            len(population),
            k
        )

        best_index = indices[
            np.argmax(fitness[indices])
        ]

        selected.append(
            population[best_index]
        )

    return np.array(selected)