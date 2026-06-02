def create_pop(ind_generator , pop_size=100):
    # creating a list of (pop_size number of) individuals
    return [ind_generator() for _ in range(pop_size)]

def evaluate_pop(fit_func, population):
    """Receives a list of individuals (aka a population)
    and a fitness evaluating function and returns
    the corresponding fitness list"""
    # evaluating all the individuals in the population
    return [fit_func(ind) for ind in population]