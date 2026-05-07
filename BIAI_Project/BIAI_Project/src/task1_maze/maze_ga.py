import numpy as np
import matplotlib.pyplot as plt
import os

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection
from src.common.crossover import uniform_crossover
from src.common.mutation import random_mutation

from src.task1_maze.visualization import plot_path


maze = np.array([

    [0,0,0,1,0,0,0,0,0,0],
    [1,1,0,1,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,0,1,0],
    [0,1,0,0,0,0,1,0,0,0],
    [0,1,0,1,1,0,1,1,1,0],
    [0,0,0,1,0,0,0,0,1,0],
    [1,1,0,1,0,1,1,0,1,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,1,1,1,0,0,0,1,1,0]

])

start = (0, 0)
goal = (9, 9)

moves = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0)
]


def fitness(individual):

    x, y = start

    invalid_moves = 0
    repeated_positions = 0
    reverse_moves = 0

    visited = set()
    visited.add((x, y))

    previous_gene = None

    for step_count, gene in enumerate(individual):

        if previous_gene is not None:

            if (
                (previous_gene == 0 and gene == 2)
                or
                (previous_gene == 2 and gene == 0)
                or
                (previous_gene == 1 and gene == 3)
                or
                (previous_gene == 3 and gene == 1)
            ):
                reverse_moves += 1

        previous_gene = gene

        dx, dy = moves[int(gene)]

        nx = x + dx
        ny = y + dy

        if (
            0 <= nx < len(maze)
            and
            0 <= ny < len(maze)
            and
            maze[nx][ny] == 0
        ):

            x, y = nx, ny

            if (x, y) in visited:
                repeated_positions += 1

            visited.add((x, y))

        else:
            invalid_moves += 1

        if (x, y) == goal:

            return (
                20000
                - step_count * 150
                - invalid_moves * 300
                - repeated_positions * 250
                - reverse_moves * 400
            )

    distance = abs(goal[0] - x) + abs(goal[1] - y)

    return (
        1000
        - distance * 150
        - invalid_moves * 100
        - repeated_positions * 120
        - reverse_moves * 200
    )


ga = GeneticAlgorithm(
    pop_size=500,
    num_genes=40,
    fitness_func=fitness,
    selection_func=tournament_selection,
    crossover_func=uniform_crossover,
    mutation_func=random_mutation,
    gene_space=[0, 1, 2, 3],
    mutation_rate=0.03,
    elitism_count=10
)

best_solution, best_fitness, history = ga.run(
    generations=1500
)

print("\nFINAL RESULTS")
print("Best Fitness:", best_fitness)

print("\nBest Solution:")
print(best_solution)

os.makedirs(
    "results/task1",
    exist_ok=True
)

plt.figure(figsize=(10, 5))

plt.plot(history)

plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.grid(True)

plt.savefig(
    "results/task1/fitness_history.png"
)

plt.close()

plot_path(
    maze,
    best_solution,
    start,
    goal,
    moves
)

print("\nResults saved in:")
print("results/task1/")