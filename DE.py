def roullete_selction(population, fitnesses, maximization = True):
    """
        Select one individual from the population using roulette wheel selection.

        Args:
            population (list): List of individuals (vectors θ).
            fitnesses (list or array): Fitness value for each individual.

        Returns:
            ind: Selected individual (copy of vector).
        """
    S = sum(fitnesses)
    alpha = random.uniform(0, S)
    iS =0
    j=0
    while iS < alpha and j < len(population):
        iS +=fitnesses[j]
        j+=1
    # Return the individual whose cumulative fitness passes alpha
    return population[j - 1].copy()

def  tournament_selection( population, fitnesses, maximization= True  ,k=3):
    tournament =random.sample( population, k)
    best= tournament[0]
    j=1
    while j < len(tournament):
        i1= best
        i2=tournament[j]

        if fitnesses(i1) > fitnesses(i2):
            best= i1
        else:
            best= i2
        j+=1
    return best

def rank_selection(population, fitnesses):
    n=  len(population)

    ranked_pop= sorted(population, key= fitnesses)

    v= 1/(n-2.001) #formula in document
    selected= ranked_pop[0]

    for i in range(n):
        alpha = random.uniform(0,v)
        for j in range(n):
            rank_j =j+1
            p_j= rank_j / (n * (n+1))
            if p_j <= alpha:
                selected =ranked_pop[j]
                break
    return selected

#ver isto com o stor

def genetic_algorithm(individual_generator, initial_pop_generator, fitness_func, selector, mutator, crossover,  pop_size = 50, generations= 100,  mutation_rate = 0.05 , crossover_rate= 0.8, maximization =True ):

    population = initial_pop_generator(individual_generator, pop_size)
    pop_fits= [fitness_func(ind) for ind in population]

    sel_m = selector.lower().strip()
    cx_m = crossover.lower().strip()
    mut_m = mutator.lower().strip()

    # --- Initialisation ---
    population = _init_population(
        pop_size, n_params, init_method,
        input_size, hidden1_size, hidden2_size, output_size
    )
    fitnesses = np.array([fitness_func(ind, *fitness_args)
                          for ind in population])

    best_idx = int(np.argmax(fitnesses)) # value of the best fitness
    best_ind = population[best_idx].copy()
    best_fit = fitnesses[best_idx]

    history = []

    initial_best_f = max(pop_fits) #maximization problem
    history.append(initial_best_f)

    for generation in range(generations):
        offspring = [] # empty population (new candidate solutions)
        while len(offspring) < pop_size:
                par1= selector( population, fitnesses = pop_fits, maximization = maximization)

                par2 = selector(population, fitnesses = pop_fits, maximization = maximization)

