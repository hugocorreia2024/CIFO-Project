import random

class Population:
    def __init__(self, size, optim, sol_size, valid_set, repetition):
        self.size = size
        self.optim = optim
        self.sol_size = sol_size
        self.valid_set = valid_set
        self.repetition = repetition
        self.individuals = []

    def evolve(self, gens, xo_prob, mut_prob, select, xo, mutate, elitism):
        for gen in range(gens):
            new_pop = []

            # Selection
            selected = select(self)

            # Crossover
            for i in range(0, len(selected), 2):
                if random.random() < xo_prob:
                    offspring1, offspring2 = xo(selected[i], selected[i + 1])
                else:
                    offspring1, offspring2 = selected[i], selected[i + 1]
                new_pop.extend([offspring1, offspring2])

            # Mutation
            for individual in new_pop:
                if random.random() < mut_prob:
                    individual = mutate(individual)

            # Elitism
            if elitism:
                elite = min(self.individuals, key=lambda ind: ind.fitness)
                if max(new_pop, key=lambda ind: ind.fitness).fitness > elite.fitness:
                    new_pop[random.randint(0, len(new_pop) - 1)] = elite

            self.individuals = new_pop

            # Evaluate fitness
            for individual in self.individuals:
                individual.evaluate_fitness()

            # Logging the best individual of the current generation
            best_individual = min(self.individuals, key=lambda ind: ind.fitness)
            print(f"Best individual of gen {gen + 1}: Fitness: {best_individual.fitness}")