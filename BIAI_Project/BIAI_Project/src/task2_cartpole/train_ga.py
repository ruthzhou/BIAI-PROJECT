import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection
from src.common.crossover import uniform_crossover
from src.common.mutation import random_mutation
from src.task2_cartpole.cartpole_policy import LinearPolicy

env = gym.make("CartPole-v1")

def fitness(individual):
    policy = LinearPolicy(individual)
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(200):
        action = policy.act(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    return total_reward

ga = GeneticAlgorithm(
    pop_size=20,
    num_genes=4,
    fitness_func=fitness,
    selection_func=tournament_selection,
    crossover_func=uniform_crossover,
    mutation_func=random_mutation,
    gene_space=[-1, 0, 1]
)

population, history = ga.run(50)

os.makedirs("results/task2", exist_ok=True)

# SAVE REWARD GRAPH
plt.plot(history)
plt.title("Reward Over Generations")
plt.savefig("results/task2/reward_history.png")
plt.close()

# SAVE BEST RESULT
with open("results/task2/best_result.txt", "w") as f:
    f.write(f"Best fitness: {max(history)}\n")

print("Training complete. Check results folder.")