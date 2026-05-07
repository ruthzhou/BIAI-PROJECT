import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection
from src.common.crossover import uniform_crossover

from src.task2_cartpole.cartpole_mutation import cartpole_mutation
from src.task2_cartpole.cartpole_policy import LinearPolicy


env = gym.make("CartPole-v1")


def fitness(individual):

    policy = LinearPolicy(individual)

    observation, _ = env.reset()

    total_reward = 0

    for _ in range(500):

        action = policy.act(observation)

        observation, reward, done, truncated, _ = env.step(action)

        total_reward += reward

        if done or truncated:
            break

    return total_reward


ga = GeneticAlgorithm(
    pop_size=100,
    num_genes=4,
    fitness_func=fitness,
    selection_func=tournament_selection,
    crossover_func=uniform_crossover,
    mutation_func=cartpole_mutation,
    gene_space=[0],
    mutation_rate=0.1,
    elitism_count=5
)

best_solution, best_fitness, history = ga.run(
    generations=200
)

print("\nFINAL RESULTS")
print("Best Fitness:", best_fitness)

print("\nBest Weights:")
print(best_solution)

os.makedirs(
    "results/task2",
    exist_ok=True
)

plt.figure(figsize=(10, 5))

plt.plot(history)

plt.title("Reward Over Generations")
plt.xlabel("Generation")
plt.ylabel("Reward")

plt.grid(True)

plt.savefig(
    "results/task2/reward_history.png"
)

plt.close()

with open(
    "results/task2/best_result.txt",
    "w"
) as file:

    file.write(
        f"Best Fitness: {best_fitness}\n"
    )

    file.write(
        f"Best Weights:\n{best_solution}"
    )

print("\nTraining complete.")
print("Check results/task2/")