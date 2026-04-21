import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size, num_genes, fitness_func,
                 selection_func, crossover_func, mutation_func, gene_space):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.gene_space = gene_space

        self.population = np.array([
            np.random.choice(gene_space, num_genes)
            for _ in range(pop_size)
        ])

        self.fitness_history = []

    def evaluate(self):
        return np.array([self.fitness_func(ind) for ind in self.population])

    def run(self, generations):
        for _ in range(generations):
            fitness = self.evaluate()
            self.fitness_history.append(np.max(fitness))

            parents = self.selection_func(self.population, fitness)
            offspring = self.crossover_func(parents)
            self.population = self.mutation_func(offspring, self.gene_space)

        return self.population, self.fitness_history