from algorithms import hill_climber, simulated_annealing, genetic_algorithm
from knapsack_problem import (generate_random_solution, calculate_fitness, generate_neighborhood_v1,
                              generate_neighborhood_v2, generate_random_neighbour)
from utils import run_multiple_times, plot_history
from GAs_operators.population import create_pop
from GAs_operators.selectors import tournament_selection
from GAs_operators.mutators import bit_flip_mutation
from GAs_operators.crossover_operators import one_point_xover

if __name__ == "__main__":

   ga_params = {
       "individual_generator": generate_random_solution,
        "initial_pop_generator": create_pop,
        "fit_func": calculate_fitness,
        "selector": tournament_selection,
        "mutator": bit_flip_mutation,
        "xover_operator": one_point_xover,
        "pop_size": 100,
        "p_mut": 0.01,
        "p_xover":0.8,
        "n_gens": 100,
        "maximization":True
   }

   best = genetic_algorithm(**ga_params)

   print(f'Best found Solution: {best}')
   print(f'Best found Solution Fitness: {calculate_fitness(best)}')




