# 🔬 MetaSR — Metallurgical Symbolic Regression

MetaSR is a cluster-guided symbolic regression platform for discovering interpretable metallurgical equations from numeric datasets.

## Live Application

Use MetaSR here:

**https://theormets.github.io/MetaSR/**

## Workflow

1. Upload a numeric CSV dataset.
2. Enter the target column name.
3. Probe all available grammar clusters.
4. Select the best-fitting grammar.
5. Run symbolic regression using PySR.
6. Return candidate equations with performance and similarity metrics.

## Application Modes

### Auto-Probe

MetaSR evaluates all 13 grammar clusters and selects the most suitable grammar for the uploaded dataset.

Two probe modes are available:

- **Fast mode:** lightweight grammar-aware heuristic.
- **Research mode:** grammar-guided PySR evaluation across all clusters.

### Select by Reference Equation

Users who know the expected governing physics can search the reference-equation database and run symbolic regression with the selected equation's grammar cluster.

## Features

- Cluster-guided symbolic regression
- Thirteen metallurgical grammar clusters
- PySR and SymbolicRegression.jl equation discovery
- Automatic feature engineering
- R², adjusted R² and RMSE evaluation
- Grammar compatibility analysis
- Equation similarity search
- Reference library containing 1,324 equations
- CSV dataset support
- Physics-informed grammar selection

## Technology

- Python
- Gradio
- PySR
- SymbolicRegression.jl
- Julia
- pandas
- NumPy
- scikit-learn
- Docker
- GitHub Actions
- GitHub Pages

## Deployment Architecture

GitHub contains the source repository and provides the public GitHub Pages entry point.

The interactive application requires Python, Julia and PySR, which cannot execute directly on GitHub Pages. Therefore, the existing Hugging Face Space provides the computational backend and is embedded in the GitHub Pages website.

```text
GitHub Pages
    ↓
Embedded Gradio interface
    ↓
Hugging Face Space
    ↓
Python + Julia + PySR
