import numpy as np
import pandas as pd
import time

from parkinsons_problem import fitness_function, compute_n_params
from differential_evolution import differential_evolution

df = pd.read_csv("parkinsons_preprocessed.csv")
X  = df.drop(columns=["status"]).values.astype(float)
y  = df["status"].values.astype(int)

def fitness_fn(solution):
    return fitness_function(solution, X, y,
                            input_size=22,
                            hidden_sizes=(10, 10),
                            output_size=1)

n_params = compute_n_params(22, (10, 10), 1)

pop_size    = [50, 100]
n_generations  = [50, 100]
F_list          = [0.5, 0.8]
CR_list          = [0.7, 0.9]

n_runs = 3

results_de = pd.DataFrame(columns=[
    'pop_size', 'generations', 'F', 'CR',
    'mean_fitness', 'std_fitness', 'all_fitnesses'
])

total_combinations = (len(pop_size) * len(n_generations) *
                      len(F_list) * len(CR_list))
combo_count  = 0
global_start = time.time()

print(f"Total de combinações: {total_combinations}  |  runs por combinação: {n_runs}\n")


for pop in pop_size:
    for ngen in n_generations:
        for F in F_list:
            for CR in CR_list:
                combo_count += 1
                all_fit = []

                for run in range(n_runs):
                    run_start = time.time()

                    _, best_fit, _ = differential_evolution(
                        fitness_func  = fitness_fn,
                        n_params      = n_params,
                        pop_size      = pop,
                        generations   = ngen,
                        bounds        = (-1.0, 1.0),
                        F             = F,
                        CR            = CR,
                        maximization  = True,
                        seed          = run * 7 + combo_count
                    )

                    all_fit.append(best_fit)
                    elapsed = time.time() - run_start
                    print(f"    run {run + 1}/{n_runs}  fitness={best_fit:.4f}  "
                          f"({elapsed:.1f}s)")

                mean_fit = np.mean(all_fit)
                std_fit  = np.std(all_fit)

                elapsed_total = time.time() - global_start
                avg_per_combo = elapsed_total / combo_count
                remaining     = (total_combinations - combo_count) * avg_per_combo

                print(f"[{combo_count}/{total_combinations}] "
                      f"pop={pop}  gen={ngen}  F={F}  CR={CR}  →  "
                      f"mean={mean_fit:.4f} ± {std_fit:.4f}  |  "
                      f"decorrido={elapsed_total/60:.1f}min  "
                      f"restante~{remaining/60:.1f}min\n")

                results_de = pd.concat([
                    results_de,
                    pd.DataFrame([{
                        'pop_size'     : pop,
                        'generations'  : ngen,
                        'F'            : F,
                        'CR'           : CR,
                        'mean_fitness' : mean_fit,
                        'std_fitness'  : std_fit,
                        'all_fitnesses': all_fit
                    }])
                ], ignore_index=True)

                results_de.to_csv("de_gridsearch_full.csv", index=False)

print(f"\nGrid search DE completed in {(time.time() - global_start)/60:.1f} min.")

print("\nTop 5 combos:")
top5 = results_de.sort_values(by='mean_fitness', ascending=False).head(5)
print(top5[['pop_size', 'generations', 'F', 'CR',
            'mean_fitness', 'std_fitness']].to_string(index=False))
