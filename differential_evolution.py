import numpy as np
import random
import pandas as pd
import time
from my_parkinsons_problem import fitness_function, compute_n_params


def differential_evolution(
    fitness_func,
    n_params,
    pop_size=30,
    generations=100,
    bounds=(-1.0, 1.0),
    F=0.8,
    CR=0.9,
    maximization=True,
    seed=None,
):
 

   
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    
    # serve para converer (low, high) num array uniforme para todas as dimensões.
    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple, np.ndarray)):
        low  = np.full(n_params, bounds[0], dtype=float)
        high = np.full(n_params, bounds[1], dtype=float)
    else:
        # bounds é uma lista de tuplos [(l1,h1), ..., (ln,hn)]---
        bounds = list(bounds)
        low  = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)

    
    # Cada linha é uma pessoa: um vetor real de comprimento n_params, que foi iniciado ao mesmo tempo dentro dos bounds.
    population = np.random.uniform(low, high, size=(pop_size, n_params))

    #  Avaliar o fitness de toda a população inicial 
    fitness = np.array([fitness_func(ind) for ind in population])

    # Identificar o melhor para começar
    if maximization:
        best_idx = np.argmax(fitness)
    else:
        best_idx = np.argmin(fitness)

    best_individual = population[best_idx].copy()
    best_fitness    = fitness[best_idx]
    history         = [best_fitness]  # rregista a evolução do mehor fitness

    # Ciclo principal de evolução 
    for gen in range(generations):

        for i in range(pop_size):

            # Mutação (DE/rand/1) 
            # Selecionar 3 índices distintos e diferentes de i.---
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)

            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Vetor mutante: perturbação aleatória na direção (b - c).---
            mutant = a + F * (b - c)

            # Garantir que o mutante fica dentro dos bounds (clipping).---
            mutant = np.clip(mutant, low, high)

            # Crossover binomial 
            # Para cada gene, sorteia um número entre 0 e 1, se for menor que CR então gene vem do vetor mutante, caso contrário o gene vem do indivíduo original
            
            j_rand    = random.randint(0, n_params - 1)
            cross_mask = (np.random.uniform(0, 1, n_params) < CR)
            cross_mask[j_rand] = True  # garante pelo menos um gene do mutante

            trial = np.where(cross_mask, mutant, population[i])

            #  Seleção greedy 
            # O trial vector substitui o indivíduo atual só se tiver fitness igual ou melhor.
            trial_fitness = fitness_func(trial)

            if maximization:
                if trial_fitness >= fitness[i]:
                    population[i] = trial
                    fitness[i]    = trial_fitness
            else:
                if trial_fitness <= fitness[i]:
                    population[i] = trial
                    fitness[i]    = trial_fitness

            # Atualizar o melhor global 
            if maximization:
                if trial_fitness > best_fitness:
                    best_fitness    = trial_fitness
                    best_individual = trial.copy()
            else:
                if trial_fitness < best_fitness:
                    best_fitness    = trial_fitness
                    best_individual = trial.copy()

        # Guardar o melhor fitness de cada geração
        history.append(best_fitness)

        
        print(f"  [DE] Geração {gen + 1:>4}/{generations} | Melhor fitness: {best_fitness:.6f}")

    return best_individual, best_fitness, history



# EXEMPLO DE UTILIZAÇÃO — teste independente do projeto


def _example_sphere_function():
    """
    Exemplo simples com a função Sphere (minimização):
        f(x) = sum(x_i^2)
    O mínimo global é 0, em x = [0, 0, ..., 0].

    Executar apenas se este ficheiro for corrido diretamente:
        python differential_evolution.py
    """
    print("=" * 60)
    print("Exemplo: minimizar a função Sphere em 5 dimensões")
    print("  f(x) = x1² + x2² + x3² + x4² + x5²")
    print("  Mínimo global: f(0,...,0) = 0")
    print("=" * 60)

    def sphere(x):
        return -np.sum(x ** 2)   # negativo porque a função maximiza por omissão

    best, fitness, history = differential_evolution(
        fitness_func = sphere,
        n_params     = 5,
        pop_size     = 20,
        generations  = 50,
        bounds       = (-5.0, 5.0),
        F            = 0.8,
        CR           = 0.9,
        maximization = True,   # maximizar -f(x) ≡ minimizar f(x)
        seed         = 42,
    )

    print("\nResultados:")
    print(f"  Melhor vetor  : {np.round(best, 6)}")
    print(f"  Melhor fitness: {fitness:.8f}  (esperado ≈ 0.0)")
    print(f"  Convergiu em {len(history)} gerações")
    return best, fitness, history


def _template_integration():
    """
    Template (não executável diretamente) que mostra como ligar
    differential_evolution à fitness_function do projeto real.

    Pressupõe que já existem no projeto:
        - fitness_function(weights) → float
        - n_params                  → int   (número de pesos da MLP)
    """

    
    

    
    DE_CONFIG = {
        "pop_size"     : 30,
        "generations"  : 100,
        "bounds"       : (-1.0, 1.0),
        "F"            : 0.8,
        "CR"           : 0.9,
        "maximization" : True,   # porque a fitness_function devolve F1/accuracy
        "seed"         : 42,
    }

   
    pass


if __name__ == "__main__":
    _example_sphere_function()

