import numpy as np

def random_mutation(population, gene_space, mutation_rate=0.1):
    for individual in population:
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.choice(gene_space)
    return population