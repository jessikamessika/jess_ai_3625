import pygad
import numpy as np

def himmelblaus_function(input_vector) -> float:
    '''a classic function to test performance of local search algos
    
    '''
    x, y = input_vector
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def fitness_function(ga_instance, solution, solution_idx):
    # won't use ga_instance and solution_idx but pygad wants them
    # need negative of himmelblaus cuz looking for minimum solution
    return -himmelblaus_function(solution)

# custom selection operator (random sel)
def random_selection(fitness, num_parents, ga_instance):
    # random: won't base parent choices based on fitness
    # replacement False cuz once you've selected parent, can't select again
    selection_idx = np.random.choice(len(fitness), size=num_parents, replace=False)
    # return selected indvs and indexes
    return ga_instance.population[selection_idx], selection_idx

# create GA Genetic Algo instance
ga = pygad.pygad.GA(
    # can provide stopping criteria
    # info to provide: see slides for GA step-by-step
    num_generations=100,
    num_parents_mating=10, # how many will be picked out for crossover for next gen
    sol_per_pop=25, # how many solutions at a time
    fitness_func=fitness_function,
    # either provide starting pop or info on how to randomly generate
    init_range_high=5,
    init_range_low=-5,
    num_genes=2,
    parent_selection_type=random_selection,
    crossover_type='single_point',
    mutation_type='random',
    mutation_percent_genes=0.1, # 10% mutate
    )
ga.run()
solution, solution_fitness, solution_idx = ga.best_solution()
print(solution, solution_fitness)