import random
from utils import find_best_neighbor
import numpy as np

def hill_climber(generate_initial_solution, neighborhood_function, fitness_function,
                 maximize=True):
    # Generate an initial solution
    current_solution = generate_initial_solution()
    print("Initial Solution:", current_solution)
    print("Initial Fitness:", fitness_function(current_solution))

    # Iterate. We can start with an infinite loop and break when we can no longer climb
    while True:
        # Find the best neighbor and its fitness
        best_neighbor = find_best_neighbor(current_solution, neighborhood_function, fitness_function, maximize)

        if maximize:
            # If the best neighbor is better than the current solution, move to it
            if fitness_function(best_neighbor) > fitness_function(current_solution):
                current_solution = best_neighbor
            else:
                break  # No better neighbor found, we are at a local optimum
        else: # We are minimizing
            if fitness_function(best_neighbor) < fitness_function(current_solution):
                current_solution = best_neighbor
            else:
                break  # No better neighbor found, we are at a local optimum
        print("Current Solution:", current_solution)
        print("Current Fitness:", fitness_function(current_solution))

    return current_solution


def simulated_annealing(initial_sol_generator,
                        fitness_function,
                        n_iterations,
                        neighbour_generator,
                        T,
                        alpha,
                        verbose=False,
                        maximize = True):

    # starting by shooting in the dark (i.e., generating a random solution)
    current_sol = initial_sol_generator()
    current_fit = fitness_function(current_sol)

    # printing on console, if verbose is active
    if verbose:
        print(f"Initial solution: {current_sol}")
        print(f"Initial Fitness: {current_fit}")

    # because I am willing to choose a frog over a prince, I need to keep track
    # of the best solution I ever found. At the end, I will return this overall best and not
    # the current solution (which might be worse than the overall best)

    best_found = current_sol
    best_found_fit = current_fit

    # starting my fitness history list (with the first initial solution):
    history = [best_found_fit]


    # having the stopping criteria of the number of iterations
    for iteration in range(n_iterations):

        # generating a random neighbour
        neighbour = neighbour_generator(current_sol)
        neighbour_fit = fitness_function(neighbour)

        # deciding whether or not I "move" to this neighbour

        # I will always move to the neighbour if it's better than the current solution
        if (maximize and neighbour_fit > current_fit) or (not maximize and (neighbour_fit < current_fit)):
            # updating the current solution if I am to move to the neighbour
            current_sol = neighbour
            current_fit = neighbour_fit
        # I still have a probability of moving to the neighbour even if it's not better than the current sol
        else:
            # calculating the acceptance probability (i.e., probability of accepting kissing a frog)
            # first, computing the absolute fitness difference
            fitness_diff = abs(neighbour_fit - current_fit)

            # then using the difference to calc the probability
            acceptance_prob = np.exp(-fitness_diff / T)

            # now that I have my acceptance probability, I see if I am to move to the neighbour
            # or not...
            if np.random.rand() < acceptance_prob:
                # updating the current solution if I am to move to the neighbour
                current_sol = neighbour
                current_fit = neighbour_fit

        # printing the current solution of the iteration if verbose
        if verbose:
            print(f'Iteration: {iteration + 1}: Current Solution: {current_sol}')
            print(f'Iteration: {iteration + 1}: Current Fitness: {current_fit}')

        # updating the temperature
        T *= alpha

        # updating the best found history log (if applicable)
        if (maximize and current_fit > best_found_fit) or (not maximize and (current_fit < best_found_fit)):
            best_found = current_sol
            best_found_fit = current_fit

        # updating my fitness logging history
        history.append(best_found_fit)

    # if verbose, print the final solution
    if verbose:
        print(f'Final Solution: {best_found}')
        print(f'Final Fitness: {best_found_fit}')

    # returning the overall best solution that was found during my learning
    return best_found, history


def genetic_algorithm(individual_generator,
                      initial_pop_generator,
                      fit_func,
                      selector,
                      mutator,
                      xover_operator,
                      pop_size,
                      p_mut,
                      p_xover,
                      n_gens,
                      maximization=True):

        """TASK: LOG HISTORY OF BEST IN POPULATION AND RETURN IT IN THE END"""

        ## Gen 0 - Initializing the algorithm: ##

        # creating and evaluating an initial random population
        population = initial_pop_generator(individual_generator, pop_size=pop_size)
        pop_fits = [fit_func(ind) for ind in population]

        ## Running the evolution ##

        # n_gens == n_iterations
        for generation in range(n_gens):

            # creating an empty offspring population
            offspring = []

            # I will add individuals to the offspring population until
            # it's full (aka has pop_size number of individuals)
            while len(offspring) < pop_size:

                # choosing two (parent) individuals from the population
                parent1 = selector(population=population,
                                   fitnesses=pop_fits,
                                   maximization=maximization)

                parent2 = selector(population=population,
                                   fitnesses=pop_fits,
                                   maximization=maximization)

                # choosing between reproduction and xover:
                if random.random() <= p_xover:
                    # obtaining the offspring through xover
                    child1, child2 = xover_operator(parent1, parent2)
                else:
                    # offspring will be a copy of the parents
                    child1 = parent1.copy() # or [bit for bit in parent1]
                    child2 = parent2.copy() # or [bit for bit in parent2]

                # you're not just 50% mom + 50% dad! You're your own person!!
                # we must mutate...
                child1 = mutator(child1, p_mut=p_mut)
                child2 = mutator(child2, p_mut=p_mut)

                # adding the two children into the offspring population
                offspring.extend([child1, child2]) # the same as appending child1 then appending child2

            # now that I have my offspring population filled its time to 'evolve'
            # the circle of time... children replace their parents... oh... the beauty of nature...
            # aka replacing the current population with the offspring population
            population = [child for child in offspring]
            pop_fits = [fit_func(ind) for ind in population] # evaluating the offspring (aka the new individuals of the pop)

        # after evolution concluded, I locate and return the best individual in the population
        return population[np.argmax(pop_fits)] if maximization else population[np.argmin(pop_fits)]


