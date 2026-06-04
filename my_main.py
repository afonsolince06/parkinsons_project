import pandas as pd
import matplotlib.pyplot as plt
import ast

# Caminho do CSV
csv_path = "pso_gridsearch_full.csv"
pso_results = pd.read_csv(csv_path)

# Top 5 melhores combinações
top5 = pso_results.sort_values(by='mean_fitness', ascending=False).head(5)

plt.figure(figsize=(12, 6))

for i, (_, row) in enumerate(top5.iterrows(), 1):
    all_runs = row['all_fitnesses']

    # Se não for lista, transforme em lista
    if isinstance(all_runs, str):
        all_runs = ast.literal_eval(all_runs)
    if isinstance(all_runs, float):
        all_runs = [all_runs]

    # Garantir que cada run é uma lista
    if isinstance(all_runs[0], float):
        all_runs = [all_runs]

    for run_idx, run_history in enumerate(all_runs, 1):
        plt.plot(
            range(1, len(run_history) + 1),
            run_history,
            linestyle='-' if run_idx == 1 else '--',
            alpha=0.7,
            label=f"Top {i} Run {run_idx}" if run_idx == 1 else None
        )

plt.xlabel("Iteração")
plt.ylabel("Fitness (F1-Score)")
plt.title("Convergência do PSO - Top 5 combinações do Grid Search")
plt.grid(True)
plt.legend(fontsize=8, loc='lower right')
plt.show()