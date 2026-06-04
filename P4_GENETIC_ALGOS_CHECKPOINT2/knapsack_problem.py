import random
from ks_data import values, volumes, capacity

# We can use len() of values or volumes to know the number of items

def generate_random_solution():
    return [random.randint(0, 1) for _ in range(len(values))]

def calculate_fitness(solution):
    # Use comprehensions with ifs to get weights and values of the included items

    # First, the volume as we need to verify validity
    total_volume = sum(volumes[i] for i in range(len(solution)) if solution[i] == 1)

    if total_volume > capacity:
        # If invalid, penalize
        return -total_volume
    else:
        # If valid, return the total value
        total_value = sum(values[i] for i in range(len(solution)) if solution[i] == 1)
        return total_value
    
def generate_neighborhood_v1(solution):
    neighborhood = []
    for i in range(len(solution)):
        neighbor = [bit for bit in solution]  # Create a copy of the solution
        neighbor[i] = 1 - neighbor[i]  # This formula works by flipping the bit, but an if works
        neighborhood.append(neighbor)
    return neighborhood

def generate_neighborhood_v2(solution):
    neighborhood = []
    for _ in range(10):
        neighbor = [bit for bit in solution]
        # Randomly select two different indices to switch
        index1 = random.randint(0, len(solution) - 1)
        index2 = random.randint(0, len(solution) - 1)
        # Switch the inclusion status of the two items
        neighbor[index1] = 1 - neighbor[index1]
        neighbor[index2] = 1 - neighbor[index2]
        neighborhood.append(neighbor)
    return neighborhood

def generate_random_neighbour(solution):

    """Receives a solution as input and returns a random neighbour"""

    # creating a copy of the solution
    neighbour = solution.copy() # or... [bit for bit in solution]

    # choosing a random bit to flip
    to_flip = random.randint(0, len(solution) - 1)

    # actually flipping the bit in the chosen index (i.e., update the neighbour)
    neighbour[to_flip] = 1 - neighbour[to_flip]

    # returning the neighbour
    return neighbour

