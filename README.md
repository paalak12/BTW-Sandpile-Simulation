# Bak-Tang-Wiesenfeld Sandpile Model

## Project Overview
This project implements the Bak-Tang-Wiesenfeld (BTW) sandpile model to investigate self-organized criticality (SOC) using computational simulation.

The model consists of a two-dimensional square lattice where grains are added randomly. When a site reaches the critical height of four grains, it topples and distributes one grain to each of its four nearest neighbours.

Open boundary conditions are used, allowing grains to dissipate from the system.

## Objectives
- Simulate self-organized criticality.
- Study avalanche-size distributions.
- Investigate the effect of lattice size.
- Analyse power-law behaviour using CCDF and maximum likelihood estimation.
- Visualize the sandpile in 2D and 3D.

## Implementation
- Python
- NumPy
- Matplotlib
- SciPy
- Pandas

## Experiments
1. Evolution of mean lattice height.
2. Avalanche-size distribution.
3. Comparison of lattice sizes (20 × 20, 40 × 40 and 60 × 60).
4. Complementary cumulative distribution function (CCDF).
5. Discrete power-law fitting.

## How to Run

Install dependencies:

`pip install -r requirements.txt`

Run the primary simulation:

`python sandpile.py`

Run lattice-size experiments:

`python experiments/lattice_comparison.py`

Run avalanche statistical analysis:

`python experiments/avalanche_statistics.py`

## Repository Structure
- experiments/: Numerical experiments and statistical analysis.
- figures/: Generated visualizations.
- results/: Simulation data and fitting results.
- animations/: Sandpile animations.

## Reproducibility
Random seeds are specified in the simulation scripts to make the experiments reproducible.
