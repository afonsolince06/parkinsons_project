import numpy as np
import random

def pso(fitness_func,
    n_particles,
    n_params,
    n_iterations,
    fitness_args=(),
    w=None,
    c1=None,
    c2=None,
    low=-1.0,
    high=1.0,
    seed=None):

    if seed is not None:
        np.random.seed(seed)

    positions=[]
    velocities=[]
    p_bests=[]
    p_bests_fit=[]
    for particle in range(n_particles):
        position= np.random.uniform(low, high, n_params)
        p_best= position.copy()
        p_best_fit= fitness_func(p_best)
        velocity = np.random.uniform( -abs(high - low), abs(high - low), n_params)

        positions.append(position)
        velocities.append(velocity)
        p_bests.append(p_best)
        p_bests_fit.append(p_best_fit)

    positions = np.array(positions)
    velocities = np.array(velocities)
    p_bests = np.array(p_bests)
    p_bests_fit = np.array(p_bests_fit)

    best_idx = 0

    for i in range(1, len(p_bests_fit)):
        if p_bests_fit[i] > p_bests_fit[best_idx]:
            best_idx = i
    g_best = p_bests[best_idx].copy()
    g_best_fit = p_bests_fit[best_idx]

    print("Initial global best fitness:", g_best_fit)
    history=[]
    for iteration in range(n_iterations):
        for particle in range(n_particles):
            r1 = np.random.uniform(0, 1, n_params)
            r2 = np.random.uniform(0, 1, n_params)
            velocities[particle] =  (w * velocities[particle] + c1 * r1 * (p_bests[particle] -positions[particle]) + c2 * r2 * (g_best - positions[particle]))

            positions[particle] = positions[particle] + velocities[particle]
            for j in range(n_params):
                if positions[particle][j] < low:
                    positions[particle][j] = low
                elif positions[particle][j] > high:
                    positions[particle][j] = high

            current_fit = fitness_func(positions[particle], *fitness_args)

            # Update personal best
            if current_fit > p_bests_fit[particle]:
                p_bests[particle] = positions[particle].copy()
                p_bests_fit[particle] = current_fit

            # Update global best
            if current_fit > g_best_fit:
                g_best = positions[particle].copy()
                g_best_fit = current_fit

        history.append(g_best_fit)

        print("Iteration", iteration + 1,"Best fitness:", round(g_best_fit, 4))

    return g_best, g_best_fit, history