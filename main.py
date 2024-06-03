from charles import Population
from sudoku import (
    create_individual, local_search, swap_mutation, inversion_mutation, scramble_mutation,
    displacement_mutation, insertion_mutation, gaussian_mutation, boundary_mutation,
    one_point_crossover, two_point_crossover, uniform_crossover, pmx_crossover, cx_crossover, arithmetic_crossover,
    blend_crossover,
    tournament_selection, roulette_wheel_selection, rank_selection, stochastic_universal_sampling, truncation_selection,
    boltzmann_selection, linear_rank_selection
)
import matplotlib.pyplot as plt
import numpy as np
import time

# Define the problem parameters
sol_size = 81  # 9x9 Sudoku grid
valid_set = list(range(1, 10))  # Valid numbers for Sudoku

# List of configurations to test
configs = [
    ('tournament', 'swap', 'one_point'),
    ('tournament', 'inversion', 'two_point'),
    ('roulette', 'scramble', 'uniform'),
    ('roulette', 'displacement', 'pmx'),
    ('rank', 'insertion', 'cx'),
    ('rank', 'boundary', 'blend'),
    ('sus', 'gaussian', 'arithmetic'),
    ('sus', 'swap', 'one_point'),
    ('truncation', 'inversion', 'two_point'),
    ('truncation', 'scramble', 'uniform'),
    ('boltzmann', 'displacement', 'pmx'),
    ('boltzmann', 'insertion', 'cx'),
    ('linear_rank', 'gaussian', 'arithmetic'),
    ('linear_rank', 'boundary', 'blend')
]

selection_methods = {
    'tournament': tournament_selection,
    'roulette': roulette_wheel_selection,
    'rank': rank_selection,
    'sus': stochastic_universal_sampling,
    'truncation': truncation_selection,
    'boltzmann': boltzmann_selection,
    'linear_rank': linear_rank_selection
}

mutation_methods = {
    'swap': swap_mutation,
    'inversion': inversion_mutation,
    'scramble': scramble_mutation,
    'displacement': displacement_mutation,
    'insertion': insertion_mutation,
    'gaussian': gaussian_mutation,
    'boundary': boundary_mutation
}

crossover_methods = {
    'one_point': one_point_crossover,
    'two_point': two_point_crossover,
    'uniform': uniform_crossover,
    'pmx': pmx_crossover,
    'cx': cx_crossover,
    'arithmetic': arithmetic_crossover,
    'blend': blend_crossover
}

results = []

timeout = 600  # Timeout in seconds (10 minutes)

for idx, (select, mutate, xo) in enumerate(configs):
    fitness_histories = []
    for run in range(10):  # Run each configuration 10 times
        start_time = time.time()

        # Initialize the population
        pop = Population(size=200, optim="min", sol_size=sol_size, valid_set=valid_set, repetition=True)
        pop.individuals = [create_individual() for _ in range(pop.size)]  # Use custom initialization

        # Define the operators
        selection_method = selection_methods[select]
        mutation_method = mutation_methods[mutate]
        crossover_method = crossover_methods[xo]

        # Evolve the population with adaptive mutation rate
        gens = 2000
        fitness_history = []
        for gen in range(gens):
            if time.time() - start_time > timeout:
                print(f"Timeout reached for configuration ({select}, {mutate}, {xo}), Run {run + 1}")
                break
            if gen % 50 == 0:
                print(f"Configuration ({select}, {mutate}, {xo}), Run {run + 1}, Generation {gen}")

            mut_prob = max(0.1, 1 - gen / gens)  # Adaptive mutation rate
            pop.evolve(
                gens=1,
                xo_prob=0.95,
                mut_prob=mut_prob,
                select=selection_method,
                xo=crossover_method,
                mutate=mutation_method,
                elitism=True
            )
            # Apply local search
            pop.individuals = [local_search(ind) for ind in pop.individuals]
            best_solution = min(pop.individuals, key=lambda ind: ind.fitness)
            fitness_history.append(best_solution.fitness)

            # Check for perfect solution
            if best_solution.fitness == 0:
                break

        # Pad the fitness history to ensure uniform length
        if len(fitness_history) < gens:
            fitness_history.extend([fitness_history[-1]] * (gens - len(fitness_history)))

        fitness_histories.append(fitness_history)

    avg_fitness_history = np.mean(fitness_histories, axis=0)
    results.append((idx + 1, select, mutate, xo, avg_fitness_history))

# Sort results by final fitness in ascending order
results.sort(key=lambda x: x[-1][-1])

# Plotting the results with unique colors for each line
colors = plt.cm.tab20(np.linspace(0, 1, len(configs)))

plt.figure(figsize=(12, 10)) 
for (idx, select, mutate, xo, fitness_history), color in zip(results, colors):
    label = f"#{idx}: Selection Method: {select}, Mutation Operator: {mutate}, Crossover Operator: {xo}, Fitness: {fitness_history[-1]:.2f}"
    plt.plot(fitness_history, label=label, color=color)
    # Annotate the end of each line with the configuration number
    plt.annotate(f'#{idx}', xy=(len(fitness_history) - 1, fitness_history[-1]),
                 xytext=(5, 0), textcoords='offset points', fontsize=10, color=color)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=1, fontsize='small')
plt.title('GA Performance on Sudoku Solver')
plt.tight_layout()
plt.savefig('sudoku_ga_performance.png', bbox_inches='tight')
plt.show()