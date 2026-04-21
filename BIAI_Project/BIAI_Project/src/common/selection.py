import numpy as np

def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(len(population)):
        idx = np.random.choice(len(population), k)
        best = idx[np.argmax(fitness[idx])]
        selected.append(population[best])
    return np.array(selected)