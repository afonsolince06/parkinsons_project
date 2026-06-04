import random

def one_point_xover(parent1, parent2):

    # randomly picking a (valid) xover point:
    xover_point = random.randint(1, len(parent1) - 1)

    # when parent1 and parent 2 love each other very much...
    child1 = parent1[:xover_point] + parent2[xover_point:]
    child2 = parent2[:xover_point] + parent1[xover_point:]

    # returning the offspring
    return child1, child2

