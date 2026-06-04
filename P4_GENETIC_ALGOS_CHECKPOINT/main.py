from algorithms import hill_climber, simulated_annealing
from knapsack_problem import (generate_random_solution, calculate_fitness, generate_neighborhood_v1,
                              generate_neighborhood_v2, generate_random_neighbour)
from utils import run_multiple_times, plot_history

if __name__ == "__main__":

    sa_params1 = {
        "initial_sol_generator": generate_random_solution,
        "fitness_function": calculate_fitness,
        "n_iterations": 100,
        "neighbour_generator": generate_random_neighbour,
        "T": 1000,
        "alpha": 0.95,
        "verbose": False,
        "maximize": True
    }

    sa_params2 = {
        "initial_sol_generator": generate_random_solution,
        "fitness_function": calculate_fitness,
        "n_iterations": 100,
        "neighbour_generator": generate_random_neighbour,
        "T": 100,
        "alpha": 0.75,
        "verbose": False,
        "maximize": True
    }

    # running for 30 times (default value) SA on both configurations:
    avg_history_params1 = run_multiple_times(algorithm=simulated_annealing,
                                             param_dict=sa_params1)

    avg_history_params2 = run_multiple_times(algorithm=simulated_annealing,
                                             param_dict=sa_params2)

    # plotting the average results across the 30 (default value) runs
    plot_history(avg_history_params1, title="Slow Cooking (T = 1000; alpha = 0.95)")
    plot_history(avg_history_params2, title="Fast Cooking (T = 100; alpha = 0.75)")






