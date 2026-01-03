from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import os
from functools import lru_cache
import numpy as np


wheres_waldo_locations = pd.read_csv(
    "C:/Users/Ezra/Downloads/Games/Guitar_Player/waldo/wheres-waldo-locations.csv"
)


waldo_location_map = {}
for i, record in wheres_waldo_locations.iterrows():
    key = f"B{record.Book}P{record.Page}"
    waldo_location_map[key] = (record.X, record.Y)


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

@lru_cache(maxsize=None)
def compute_fitness(agent):
    coords = np.array([waldo_location_map[loc] for loc in agent])
    diffs = coords[1:] - coords[:-1]
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def generate_random_agent():
    agent = list(waldo_location_map.keys())
    random.shuffle(agent)
    return tuple(agent)

def mutate_agent(agent_genome, max_mutations=3):
    agent_genome = list(agent_genome)
    for _ in range(random.randint(1, max_mutations)):
        i, j = random.sample(range(len(agent_genome)), 2)
        agent_genome[i], agent_genome[j] = agent_genome[j], agent_genome[i]
    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    agent_genome = list(agent_genome)
    
    if len(agent_genome) < 2:
        return tuple(agent_genome)
    
    start = random.randint(0, len(agent_genome) - 2) 
    max_length = min(20, len(agent_genome) - start)
    if max_length < 2:
        max_length = 2  
    length = random.randint(2, max_length)
    
    subset = agent_genome[start:start+length]
    agent_genome = agent_genome[:start] + agent_genome[start+length:]
    insert = random.randint(0, len(agent_genome))
    agent_genome = agent_genome[:insert] + subset + agent_genome[insert:]
    
    return tuple(agent_genome)


def generate_random_population(pop_size):
    return [generate_random_agent() for _ in range(pop_size)]

def plot_trajectory(agent_genome, filename="waldo_final.png"):
    xs, ys = zip(*[waldo_location_map[loc] for loc in agent_genome])
    plt.figure(figsize=(12.75, 8))
    plt.plot(xs, ys, "-o", markersize=5)
    plt.plot(xs[0], ys[0], "^", color="#1f77b4", markersize=10) 
    plt.plot(xs[-1], ys[-1], "v", color="#d62728", markersize=10) 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Final best path saved to {filename}")

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    fill = [x for x in parent2 if x not in child]
    i = 0
    for idx in range(len(child)):
        if child[idx] is None:
            child[idx] = fill[i]
            i += 1
    return tuple(child)


def run_genetic_algorithm(generations=500, population_size=50):
    population = generate_random_population(population_size)
    top_fraction = max(1, population_size // 10)

    for gen in range(generations):
        population_fitness = {agent: compute_fitness(agent) for agent in population}
        elitism_count = 2 
        top_agents = sorted(population_fitness, key=population_fitness.get)[:top_fraction]

        new_population = top_agents[:elitism_count]

        for agent in top_agents:
            for _ in range(2):
                new_population.append(mutate_agent(agent))
            for _ in range(2):
                new_population.append(shuffle_mutation(agent))

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(top_agents, 2)
            child = crossover(parent1, parent2)
            new_population.append(child)

            population = new_population[:population_size]

    best_agent = sorted(population, key=compute_fitness)[0]
    plot_trajectory(best_agent)
    print(compute_fitness(best_agent))
    points = [waldo_location_map[loc] for loc in best_agent]
    points_df = pd.DataFrame(points, columns=["X", "Y"])
    points_df.to_csv("points2.csv", index=False)



run_genetic_algorithm(generations=10000, population_size=100)
