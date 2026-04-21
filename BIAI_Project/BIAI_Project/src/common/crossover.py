import numpy as np

def uniform_crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[(i+1) % len(parents)]

        mask = np.random.randint(0, 2, size=len(p1))
        child = np.where(mask, p1, p2)
        offspring.append(child)

    return np.array(offspring)