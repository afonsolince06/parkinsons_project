import random
import numpy as np


def tournament_selection(population, fitnesses, tournament_size=2, maximization=True):
    """
    Imagining the KS problem with 3 items and pop_size = 4:

    example:
    population = [[1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

    fitnesses = [-20, 1000, 200, 0]

    tournament_size = 2

    """
    # building my tournament by randomly choosing tournament_size number of individuals from the population:
    tournament = [random.randint(0, len(population) - 1) for _ in range(tournament_size)]

    # now that I have the index of the competitors, I need to check their fitnesses
    fitnesses_of_competitors = [fitnesses[idx] for idx in tournament]

    # finding out WHERE in the fitness list is the winner
    if maximization:
        winner_idx = tournament[np.argmax(fitnesses_of_competitors)]
    else:
        winner_idx = tournament[np.argmin(fitnesses_of_competitors)]

    # returning from the population the corresponding winner based on the index from tournament (wow!)
    return population[winner_idx]
