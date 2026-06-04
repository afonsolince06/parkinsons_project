import numpy as np
import pandas as pd
from my_parkinsons_problem import fitness_function, compute_n_params
from my_pso import pso

import time
import random
seed_run = random.randint(0, 1_000_000)

import pandas as pd
df = pd.read_csv("parkinsons_preprocessed.csv")
X = df.drop(columns=["status"]).values.astype(float)
y = df["status"].values.astype(int)

# Definir função fitness compatível
def fitness_fn(solution):
    return fitness_function(solution, X, y, input_size=22, hidden_sizes=(10,10), output_size=1)

# Hiperparâmetros do grid
n_particles_list = [ 30, 50, 100]
n_iterations_list = [30, 50, 100]
w_list = [0.5, 0.7, 0.9]
c1_list = [1.0, 1.5, 2.0]
c2_list = [1.0, 1.5, 2.0]

n_runs = 3  # repetições por combinação

# Preparar DataFrame de resultados
results_pso = pd.DataFrame(columns=[
    'n_particles', 'n_iterations', 'w', 'c1', 'c2',
    'mean_fitness', 'std_fitness', 'all_fitnesses'
])

n_params = compute_n_params(22, (10, 10), 1)


# Grid search
for n_particles in n_particles_list:
    for n_iter in n_iterations_list:
        for w in w_list:
            for c1 in c1_list:
                for c2 in c2_list:
                    all_fit = []
                    for run in range(n_runs):
                        start_time = time.time()
                        _, best_fit, _ = pso(
                            fitness_func=fitness_fn,
                            n_particles=n_particles,
                            n_params=n_params,
                            n_iterations=n_iter,
                            w=w, c1=c1, c2=c2,
                            low=-1.0, high=1.0,
                            seed=seed_run
                        )
                        all_fit.append(best_fit)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Run {run + 1} completed in {elapsed_time:.2f} seconds")  # opcional

                    results_pso = pd.concat([
                        results_pso,
                        pd.DataFrame([{
                            'n_particles': n_particles,
                            'n_iterations': n_iter,
                            'w': w,
                            'c1': c1,
                            'c2': c2,
                            'mean_fitness': np.mean(all_fit),
                            'std_fitness': np.std(all_fit),
                            'all_fitnesses': all_fit
                        }])
                    ], ignore_index=True)

# Guardar resultados para análise
results_pso.to_csv("pso_gridsearch_full.csv", index=False)
print("Grid search para PSO concluído!")