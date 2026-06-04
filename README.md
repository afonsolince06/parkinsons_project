# Parkinson's Disease Neural Network Optimization

This project implements Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) to optimize the weights of a neural network for Parkinson's Disease classification.

Instead of training the neural network through standard backpropagation, both algorithms search directly for the best weight vector θ. Each candidate solution represents all the weights and biases of the neural network.

## Dataset

The project uses the Oxford Parkinson's Disease Detection dataset.

The dataset contains:

- 195 voice recordings
- 22 acoustic features
- 147 Parkinson's disease cases
- 48 healthy control cases
- 1 binary target variable: `status`

The task is to classify each observation as either Parkinson's disease or healthy control.

## Algorithms

The project includes:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- MLP neural network with externally optimized weights
- Grid search experiments
- Streamlit application for interactive visualization

## Project Structure

Main files:

- `app.py`: Streamlit application
- `genetic_algorithm_c.py`: Genetic Algorithm implementation
- `pso_c.py`: Particle Swarm Optimization implementation
- `parkinsons_problem_c.py`: Neural network problem formulation
- `parkinsons_preprocessed.csv`: Preprocessed Parkinson's dataset
- `grid_search_results.csv`: Experimental results for the one-hidden-layer model
- `grid_search_plot.png`: Grid search visualization
- `genetic_algorithm_hidden.py`: Genetic Algorithm for the two-hidden-layer extension
- `pso_hidden.py`: PSO for the two-hidden-layer extension
- `parkinsons_hidden.py`: Neural network formulation for the two-hidden-layer extension
- `grid_search_results_hidden.csv`: Experimental results for the two-hidden-layer model
- `grid_search_plot_hidden.png`: Grid search visualization for the two-hidden-layer model


## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Optional experiment scripts:

```bash
python main_c.py
python main_hidden.py
```

## Streamlit Application

The Streamlit application allows users to:

- Explore the dataset
- Visualize feature distributions
- Run GA and PSO interactively
- Compare convergence curves
- Compare accuracy, recall, precision and F1-score
- View confusion matrices
- Learn about the algorithms and the dataset

## Evaluation

The project compares GA and PSO using several metrics:

- Accuracy
- Recall
- Precision
- F1-score

Since the dataset is imbalanced, recall alone can produce degenerate solutions where the model predicts almost all samples as Parkinson's disease. Therefore, F1-score is used as the main comparison metric, while recall, precision and accuracy are also reported.

## Authors

Optimization Algorithms Project  
NOVA IMS — 2025/2026