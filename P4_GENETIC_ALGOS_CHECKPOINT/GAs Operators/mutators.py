import random

def bit_flip_mutation(individual, p_mut):
    # making sure the modded individual doesn't change the original
    # individual

    modded = [bit for bit in individual] # or... individual.copy()

    # going bit by bit and seeing if mutation is to be applied:
    for i in range(len(individual)):

        # seeing if the bit is to be mutated:
        if random.random() <= p_mut:
            modded[i] = 1 - modded[i]

    # at the end, returning the modded individual:
    return modded

    # this could have been easily done in ONE LINE:
    # return [1 - bit if random.random() <= p_mut else bit for bit in individual]

