import numpy as np
import matplotlib.pyplot as plt

def to_binary(x):
    return bin(x)[2:].zfill(4)

def to_decimal(binary_str):

    return int(binary_str, 2)

def find_best_neighbor(current_solution, neighborhood_function, fitness_function, 
                       maximize=True):
    
    # Generate the neighborhood
    neighborhood = neighborhood_function(current_solution)
    # Generate the fitnesses
    fitnesses = [fitness_function(neighbor) for neighbor in neighborhood]

    # Find the best neighbor's index
    if maximize:
        best_index = np.argmax(fitnesses)
    else:
        best_index = np.argmin(fitnesses)
    
    # Associate the index with the neighbor and its fitness
    best_neighbor = neighborhood[best_index]

    return best_neighbor


def run_multiple_times(algorithm, param_dict, n_runs = 30):

    """This function receives an algorithm function, the corresponding parameter dictionary and the desired
    number of runs.

    It then executes the provided algorithm n_runs number of times, keeping track of all the histories outputted
    by the algorithm. It returns a list called avg_history, representing the average history values per iteration
    for all the runs."""

    """
    -----------------------------------------------------
    
    n_runs = 2
    n_iters = 3
    
    h1 = [4, 18, 4]
    h2 = [10, 2, 4]

    all_histories = [[4, 18, 4],  [10, 2, 4]]

    avg_history = [7, 10, 4]
    
    """

    # creating an empty list for all the histories
    all_histories = []

    # running the provided algorithm n_runs number of times
    for run in range(n_runs):
        best_found, history = algorithm(**param_dict)

        # adding the history obtained in this run to the overall log
        all_histories.append(history)

    # creating the empty average_history list
    avg_hist = []

    # obtaining the number of iterations from the history list:
    n_iter = len(all_histories[0])

    # getting all the history values for each iteration
    for it in range(n_iter):

        # getting all the values for the corresponding iteration
        all_vals = [history[it] for history in all_histories] # """ n_vals = [4, 10] in first iteration, then [18, 2] .... """

        # averaging all of the values
        avg_it = sum(all_vals) / n_runs # or len(all_vals)...

        # adding the average to the averages list
        avg_hist.append(avg_it)

    return avg_hist

def plot_history(history_lt, title="Fitness Throughout the Iterations"):
    plt.plot(history_lt)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage
    x = 2
    print(f"The binary representation of {x} is: {to_binary(x)}")