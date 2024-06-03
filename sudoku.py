import random
import numpy as np

# Individual class
class Individual:
    def __init__(self, representation):
        self.representation = representation
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        return sudoku_fitness(self)

# Fitness function for Sudoku
def sudoku_fitness(individual):
    conflicts = 0
    size = 9
    subgrid_size = 3
    grid = [individual.representation[i:i + size] for i in range(0, len(individual.representation), size)]

    # Check rows for conflicts
    for row in grid:
        conflicts += len(row) - len(set(row))

    # Check columns for conflicts
    for col in range(size):
        column = [grid[row][col] for row in range(size)]
        conflicts += len(column) - len(set(column))

    # Check subgrids for conflicts
    for i in range(0, size, subgrid_size):
        for j in range(0, size, subgrid_size):
            subgrid = [grid[x][y] for x in range(i, i + subgrid_size) for y in range(j, j + subgrid_size)]
            conflicts += len(subgrid) - len(set(subgrid))

    return conflicts

# Create an individual with valid initial rows
def create_individual():
    individual = []
    for _ in range(9):
        row = list(range(1, 10))
        random.shuffle(row)
        individual.extend(row)
    return Individual(representation=individual)

# Selection Methods
def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(len(population.individuals)):
        tournament = random.sample(population.individuals, tournament_size)
        selected.append(min(tournament, key=lambda x: x.fitness))
    return selected

def roulette_wheel_selection(population):
    max_fitness = sum(ind.fitness for ind in population.individuals)
    selection_probs = [ind.fitness / max_fitness for ind in population.individuals]
    return random.choices(population.individuals, weights=selection_probs, k=len(population.individuals))

def rank_selection(population):
    sorted_individuals = sorted(population.individuals, key=lambda x: x.fitness)
    ranks = list(range(1, len(sorted_individuals) + 1))
    total_ranks = sum(ranks)
    selection_probs = [rank / total_ranks for rank in ranks]
    return random.choices(sorted_individuals, weights=selection_probs, k=len(population.individuals))

def stochastic_universal_sampling(population):
    total_fitness = sum(ind.fitness for ind in population.individuals)
    point_distance = total_fitness / len(population.individuals)
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(len(population.individuals))]

    chosen = []
    current_member = 0
    current_fitness = population.individuals[current_member].fitness
    for point in points:
        while current_fitness < point:
            current_member += 1
            current_fitness += population.individuals[current_member].fitness
        chosen.append(population.individuals[current_member])
    return chosen

def truncation_selection(population, truncation_size=0.5):
    sorted_individuals = sorted(population.individuals, key=lambda x: x.fitness)
    cutoff = int(len(sorted_individuals) * truncation_size)
    return random.choices(sorted_individuals[:cutoff], k=len(population.individuals))

def boltzmann_selection(population, temperature=1.0):
    fitnesses = np.array([ind.fitness for ind in population.individuals])
    probabilities = np.exp(-fitnesses / temperature)
    probabilities /= np.sum(probabilities)
    selected_indices = np.random.choice(len(population.individuals), size=len(population.individuals), p=probabilities)
    return [population.individuals[i] for i in selected_indices]

def linear_rank_selection(population):
    sorted_individuals = sorted(population.individuals, key=lambda x: x.fitness)
    ranks = np.arange(1, len(sorted_individuals) + 1)
    probabilities = ranks / ranks.sum()
    selected_indices = np.random.choice(len(sorted_individuals), size=len(sorted_individuals), p=probabilities)
    return [sorted_individuals[i] for i in selected_indices]

# Mutation Operators
def swap_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual.representation)), 2)
        individual.representation[idx1], individual.representation[idx2] = individual.representation[idx2], individual.representation[idx1]
    return individual

def inversion_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(individual.representation)), 2))
        individual.representation[idx1:idx2] = reversed(individual.representation[idx1:idx2])
    return individual

def scramble_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(individual.representation)), 2))
        subset = individual.representation[idx1:idx2]
        random.shuffle(subset)
        individual.representation[idx1:idx2] = subset
    return individual

def displacement_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(individual.representation)), 2))
        segment = individual.representation[idx1:idx2]
        remaining = individual.representation[:idx1] + individual.representation[idx2:]
        insert_idx = random.randint(0, len(remaining))
        individual.representation = remaining[:insert_idx] + segment + remaining[insert_idx:]
    return individual

def insertion_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(individual.representation) - 1)
        value = individual.representation.pop(idx)
        insert_idx = random.randint(0, len(individual.representation))
        individual.representation.insert(insert_idx, value)
    return individual

def gaussian_mutation(individual, mutation_rate=0.1, sigma=1.0):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(individual.representation) - 1)
        individual.representation[idx] += int(np.random.normal(0, sigma))
        individual.representation[idx] = max(1, min(9, individual.representation[idx]))
    return individual

def boundary_mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(individual.representation) - 1)
        if random.random() < 0.5:
            individual.representation[idx] = 1
        else:
            individual.representation[idx] = 9
    return individual

# Crossover Operators
def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.representation) - 1)
    child1_repr = parent1.representation[:crossover_point] + parent2.representation[crossover_point:]
    child2_repr = parent2.representation[:crossover_point] + parent1.representation[crossover_point:]
    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(parent1.representation)), 2))
    child1_repr = (parent1.representation[:point1] + parent2.representation[point1:point2] + parent1.representation[point2:])
    child2_repr = (parent2.representation[:point1] + parent1.representation[point1:point2] + parent2.representation[point2:])
    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def uniform_crossover(parent1, parent2):
    child1_repr = []
    child2_repr = []
    for i in range(len(parent1.representation)):
        if random.random() < 0.5:
            child1_repr.append(parent1.representation[i])
            child2_repr.append(parent2.representation[i])
        else:
            child1_repr.append(parent2.representation[i])
            child2_repr.append(parent1.representation[i])
    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def pmx_crossover(parent1, parent2):
    size = len(parent1.representation)
    point1, point2 = sorted(random.sample(range(size), 2))
    child1_repr = [-1] * size
    child2_repr = [-1] * size

    # Copy the segment from each parent to the child
    for i in range(point1, point2):
        child1_repr[i] = parent1.representation[i]
        child2_repr[i] = parent2.representation[i]

    def map_value(mapping, value):
        visited = set()
        while value in mapping and value not in visited:
            visited.add(value)
            value = mapping[value]
        return value

    # Create mappings for PMX
    mapping1 = {parent2.representation[i]: parent1.representation[i] for i in range(point1, point2)}
    mapping2 = {parent1.representation[i]: parent2.representation[i] for i in range(point1, point2)}

    # Fill in the rest of the values
    for i in range(size):
        if not point1 <= i < point2:
            child1_repr[i] = map_value(mapping1, parent2.representation[i])
            child2_repr[i] = map_value(mapping2, parent1.representation[i])

    # Handle remaining values that are still -1
    for i in range(size):
        if child1_repr[i] == -1:
            child1_repr[i] = parent2.representation[i]
        if child2_repr[i] == -1:
            child2_repr[i] = parent1.representation[i]

    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def cx_crossover(parent1, parent2):
    size = len(parent1.representation)
    child1_repr = [-1] * size
    child2_repr = [-1] * size

    def cycle_crossover(parent1, parent2):
        cycle = [False] * size
        start = 0
        while not cycle[start]:
            cycle[start] = True
            start = parent1.index(parent2[start])
        return cycle

    cycle = cycle_crossover(parent1.representation, parent2.representation)
    for i in range(size):
        if cycle[i]:
            child1_repr[i] = parent1.representation[i]
            child2_repr[i] = parent2.representation[i]
        else:
            child1_repr[i] = parent2.representation[i]
            child2_repr[i] = parent1.representation[i]

    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def arithmetic_crossover(parent1, parent2):
    alpha = 0.5
    child1_repr = [int(alpha * x + (1 - alpha) * y) for x, y in zip(parent1.representation, parent2.representation)]
    child2_repr = [int(alpha * y + (1 - alpha) * x) for x, y in zip(parent1.representation, parent2.representation)]
    return Individual(representation=child1_repr), Individual(representation=child2_repr)

def blend_crossover(parent1, parent2, alpha=0.5):
    child1_repr = [int(x + alpha * (y - x)) for x, y in zip(parent1.representation, parent2.representation)]
    child2_repr = [int(y + alpha * (x - y)) for x, y in zip(parent1.representation, parent2.representation)]
    return Individual(representation=child1_repr), Individual(representation=child2_repr)

# Local Search
def local_search(individual):
    # Placeholder for local search function
    # Improve individual using local search strategies
    return individual