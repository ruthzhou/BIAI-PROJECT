import numpy as np


class GeneticAlgorithm:

    def __init__(
        self,
        pop_size,
        num_genes,
        fitness_func,
        selection_func,
        crossover_func,
        mutation_func,
        gene_space,
        mutation_rate=0.1,
        elitism_count=2
    ):

        self.pop_size = pop_size
        self.num_genes = num_genes

        self.fitness_func = fitness_func
        self.selection_func = selection_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func

        self.gene_space = gene_space

        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        # Initial population
        self.population = np.array([
            np.random.choice(gene_space, num_genes)
            for _ in range(pop_size)
        ])

        self.fitness_history = []

        self.best_solution = None
        self.best_fitness = -np.inf

    def evaluate_population(self):

        fitness = np.array([
            self.fitness_func(individual)
            for individual in self.population
        ])

        return fitness

    def run(self, generations):

        for generation in range(generations):

            # Evaluate fitness
            fitness = self.evaluate_population()

            # Sort population by fitness
            sorted_indices = np.argsort(fitness)[::-1]

            self.population = self.population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Track best solution
            current_best_fitness = fitness[0]

            if current_best_fitness > self.best_fitness:

                self.best_fitness = current_best_fitness
                self.best_solution = self.population[0].copy()

            self.fitness_history.append(self.best_fitness)

            print(
                f"Generation {generation + 1} | "
                f"Best Fitness: {self.best_fitness}"
            )

            # ELITISM
            elites = self.population[:self.elitism_count]

            # Selection
            parents = self.selection_func(
                self.population,
                fitness
            )

            # Crossover
            offspring = self.crossover_func(parents)

            # Mutation
            offspring = self.mutation_func(
                offspring,
                self.gene_space,
                self.mutation_rate
            )

            # Keep elites
            remaining = self.pop_size - self.elitism_count

            offspring = offspring[:remaining]

            self.population = np.vstack((elites, offspring))

        return (
            self.best_solution,
            self.best_fitness,
            self.fitness_history
        )