# Evolutionary Strategy Function Approximator

This project uses an Evolutionary Strategy (ES) algorithm—a type of computational intelligence inspired by biological evolution—to find the optimal parameters for a complex mathematical function.

## Goal

We want to find the best possible coefficients for the target function.

The algorithm searches for parameters $(a, b, c)$ that minimize the Mean Squared Error (MSE) between the function's output and the actual values in our dataset (data/ES_data_26.dat).

$$f(i) = a \cdot (i^2 - b \cdot \cos(c \cdot \pi \cdot i))$$

## How It Works

At its core, it's a (150, 750)-ES with self-adaptive mutation.

Population Cycle: We start with 150 parent solutions (genes). They generate 750 children (5 per parent).

Survival: Only the best individuals from the child pool (\mu, \lambda)-selection survive to become the parents of the next generation.

Self-Adaptation: The 'genes' (parameters $a, b, c$) and their respective mutation rates $(\sigma_a, \sigma_b, \sigma_c)$ evolve

Stop Condition: Evolution runs for a maximum of 200 generations or until the improvement between generations is less than $1 \cdot 10^{-5}$.

# Get Started

You only need Python 3.x. No external libraries are required.

## Clone and Run

Get the code

`
git clone <repository-url>
cd topic_8

# Run with the script

./run.sh

# ...or directly with Python

python3 src/main.py

## Configuration

If you need to tweak the core parameters (like population size or max iterations), they are located in src/constants.py.

## Project Structure

```
.
├── README.md
├── data
│   └── ES_data_26.dat # Input data file
├── run.sh # Convenience run script
└── src
    ├── constants.py # Core parameters
    ├── data.py # Data loading utility
    ├── genes.py # Gene and GenePool classes
    └── main.py # Entrypoint
```

## Results

After multiple runs, the algorithm consistently finds parameters with an MSE of around 0.22
