import numpy as np
import matplotlib.pyplot as plt
import os

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection
from src.common.crossover import uniform_crossover
from src.common.mutation import random_mutation
from src.task1_maze.visualization import plot_path

maze = np.array([
    [0,0,0,0,1,0],
    [1,1,0,1,1,0],
    [0,0,0,0,0,0],
    [0,1,1,1,1,0],
    [0,0,0,0,1,0],
    [1,1,1,0,0,0]
])

start = (0, 0)
goal = (5, 5)

moves = [(0,1),(1,0),(0,-1),(-1,0)]

def fitness(individual):
    x, y = start
    penalty = 0

    for gene in individual:
        dx, dy = moves[int(gene)]
        nx, ny = x + dx, y + dy

        if 0 <= nx < 6 and 0 <= ny < 6 and maze[nx][ny] == 0:
            x, y = nx, ny
        else:
            penalty += 1

        if (x, y) == goal:
            return 10000 - penalty

    dist = abs(goal[0]-x) + abs(goal[1]-y)
    return 1 / (dist + 1 + penalty)

ga = GeneticAlgorithm(
    pop_size=50,
    num_genes=50,
    fitness_func=fitness,
    selection_func=tournament_selection,
    crossover_func=uniform_crossover,
    mutation_func=random_mutation,
    gene_space=[0,1,2,3]
)

population, history = ga.run(500)

best_solution = population[0]

# SAVE RESULTS
os.makedirs("results/task1", exist_ok=True)

plt.plot(history)
plt.title("Fitness Over Generations")
plt.savefig("results/task1/fitness_history.png")
plt.close()

plot_path(maze, best_solution, start, goal, moves)