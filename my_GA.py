
import random
import numpy as np
import math


def xavier_initialization(pop_size, layer_sizes):
    """
      Xavier (Glorot) uniform initialisation.

      Each weight matrix Wl is drawn from Uniform[-√(6/(fan_in+fan_out)),
      +√(6/(fan_in+fan_out))].  Biases are initialised to zero.

      Rationale: Xavier init keeps signal variance roughly constant across
      layers, reducing the risk of vanishing/exploding signals at start.
      For weight-optimisation this gives a better starting population than
      pure random uniform.
      """
    population = []

    for _ in range(pop_size):
        individual = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            limit = math.sqrt(6 / (fan_in + fan_out))

            # weights between current layer and next layer
            for _ in range(fan_in * fan_out):
                individual.append(random.uniform(-limit, limit))

            # biases of the next layer
            for _ in range(fan_out):
                individual.append(0)

        population.append(individual)

    return population

#selection methods
def roulette_selection(population, fitnesses, maximization = True):
    S = sum(fitnesses)

    alpha = random.uniform(0, S)
    iS =0
    j=0
    while iS < alpha and j < len(population):
        iS +=fitnesses[j]
        j+=1
    # Return the individual whose cumulative fitness passes alpha
    return population[j - 1]

def  tournament_selection( population, fitnesses, maximization= True, k=3):
    tournament = random.sample(range(len(population)), k)
    best = tournament[0]
    j=1
    for j in range(1, len(tournament)):
        i1, i2 = best, tournament[j]
        if fitnesses[i1] > fitnesses[i2]:
            best = i1
        else:
            best = i2

    return population[best] #best index

def rank_selection(population,fitnesses, maximization = True):
    n=  len(population)
    indices = list(range(n))
    ranked_indices = sorted(indices, key=lambda i: fitnesses[i])

    v= 1/(n-2.001) #formula in document
    selected= ranked_indices[0]

    for i in range(n):
        alpha = random.uniform(0,v)
        for j in range(n):
            rank_j =j+1
            p_j= rank_j / (n * (n+1))
            if p_j <= alpha:
                selected =ranked_indices[j]
                break
    return population[selected]

#crossover


def blend_crossover(p1, p2 , alpha=0.5):
    """
        Blend (BLX-α) crossover: sample uniformly from an extended interval
        [min - α·d, max + α·d] for each gene.
        """
    child1= []
    child2=[]
    for i in range(len(p1)):
        parent1= p1[i]
        parent2= p2[i]

        d= abs(p2[i]-p1[i])
        lower = min(p1[i], p2[i]) - alpha * d
        upper = max(p1[i], p2[i]) + alpha * d

        g1 = random.uniform(lower, upper)
        g2 = random.uniform(lower, upper)

        child1.append(g1)
        child2.append(g2)

    return child1, child2

def arithmetic_crossover(p1, p2, alpha =None):
    """
    Arithmetic crossover is used in case of real-value encoding.
    Arithmetic crossover operator linearly combines the two
    parent chromosomes. Two chromosomes are selected
    randomly for crossover and produce two offsprings.
        child1 = a * p1 + (1 - a) * p2
        child2 = a * p2 + (1 - a) * p1
    """
    if alpha is None:
        a = random.uniform(0,1)
    else:
        a=alpha

    c1 = [a * p1[i] + (1 - a) * p2[i] for i in range(len(p1))]
    c2 = [a * p2[i] + (1 - a) * p1[i] for i in range(len(p1))]
    return c1, c2

#mutation
def non_uniform_mutation(individual, generation,generations, mutation_rate, b=2):
    """
    Non-uniform mutation for real-valued chromosomes.

    At the beginning of the GA, the mutation can make larger changes.
    As generations pass, the mutation becomes smaller, helping fine-tune
    the solution instead of randomly destroying good weights.

    Higher b makes mutation shrink faster.
    """
    mutated= individual.copy()
    for j in range(len(mutated)):
        if random.random() < mutation_rate:
            r = random.uniform(0, 1)
            delta = (1 - r ** ((1 - generation / generations) ** b))
            mutated[j] += delta if random.random() < 0.5 else -delta
    return mutated

def uniform_mutation( individual, mutation_rate):
    mutated= individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate: #decides wheter the gene mutates
            mutated[i] = random.uniform(-1, 1)
    return mutated

def gaussian_mutation(individual, mutation_rate, sigma=0.1):
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = mutated[i] + random.gauss(0, sigma) # new gene = old gene + small random change
            if mutated[i] < -1.0:
                mutated[i] = -1.0

            if mutated[i] > 1.0:
                mutated[i] = 1.0

    return mutated

def get_best(population, fitnesses, n=1, maximization= True):
    """
    Return the n best individuals (copies) from the population,
    sorted from best to worst.
    """
    paired  = list(zip(population, fitnesses))
    paired.sort(key=lambda x: x[1], reverse=maximization)
    return [list(ind) for ind, _ in paired[:n]]

def random_initialization(pop_size, n_params):
    return [[random.uniform(-1, 1) for _ in range(n_params)] for _ in range(pop_size)]

def genetic_algorithm(fitness_func,
                      init ,
                      selector,
                      mutator,
                      crossover,
                      pop_size = None,
                      generations= None,
                      mutation_rate = None ,
                      crossover_rate= 0.9,
                      sigma=0.1,
                      elitism=2 ,
                      b= 2,
                      layer_sizes = [22, 10, 10, 1],
                      maximization =True
                       ):
    if init == 'xavier':
        population = xavier_initialization(pop_size, layer_sizes)
    else:
        # Calculate total parameters
        n_params = sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
                       for i in range(len(layer_sizes) - 1))
        population = random_initialization(pop_size, n_params)

#Parameter Tuning Method (0.03 Mutation, 0.9 Crossover)
    #population = initial_pop_generator(individual_generator, pop_size)

    fitnesses = [fitness_func(ind) for ind in population]

    best_idx = fitnesses.index(max(fitnesses)) # value of the best fitness
    best_ind = population[best_idx].copy()
    best_fit = fitnesses[best_idx]

    history = [best_fit]

    for generation in range(generations):
        offspring = [] # empty population (new candidate solutions)

        elites= get_best(population, fitnesses, n=elitism, maximization= maximization)
        offspring.extend(elites)

        while len(offspring) < pop_size:
                p1= selector( population, fitnesses = fitnesses, maximization = maximization)

                p2= selector(population, fitnesses = fitnesses, maximization = maximization)

                if random.random() <= crossover_rate:
                    child1 , child2 = crossover(p1,p2)
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                if mutator== 'gaussian':
                    child1= gaussian_mutation(child1, mutation_rate,sigma)
                    child2= gaussian_mutation(child2, mutation_rate, sigma)
                elif mutator == 'non_uniform':
                    child1 = non_uniform_mutation(child1, generation , generations, mutation_rate, b )
                    child2= non_uniform_mutation(child2, generation,generations, mutation_rate, b)
                else: #uniform
                    child1=uniform_mutation(child1, mutation_rate)
                    child2=uniform_mutation(child2, mutation_rate)

                offspring.extend([child1, child2])

        # now that I have my offspring population filled it is time to 'evolve'
        # children replace their parents
        population = [child for child in offspring[:pop_size]]

        # evaluating the offspring
        fitnesses = [fitness_func(ind) for ind in population]

        gen_best_idx = fitnesses.index(max(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = population[gen_best_idx].copy()

        history.append(best_fit)

    # after evolution concluded, return the best individual found overall
    return best_ind, best_fit, history