import numpy as np


def cartpole_mutation(
    population,
    gene_space,
    mutation_rate=0.1
):

    mutated_population = []

    for individual in population:

        mutated = individual.copy()

        for i in range(len(mutated)):

            if np.random.rand() < mutation_rate:

                mutated[i] += np.random.normal(
                    0,
                    0.5
                )

        mutated_population.append(mutated)

    return np.array(mutated_population)