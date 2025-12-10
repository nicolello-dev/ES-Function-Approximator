from math import cos, pi
from data import load_data
from genes import GenePool, Gene
from constants import MAX_ITERATIONS, POPULATION_SIZE, DATA_FILE_NAME


def f(gene: Gene) -> float:
    return gene.a * (gene.i**2 - gene.b * cos(gene.c * pi * gene.i))


data = load_data(DATA_FILE_NAME)
gene_pool = GenePool(POPULATION_SIZE, data, f)


def main():
    while not gene_pool.stop_condition_met():
        gene_pool.evolve()
        print(f"Iteration {gene_pool.current_iteration}/{MAX_ITERATIONS} completed.")

    print("Evolution complete.")
    print(
        f"Best gene: {gene_pool.genes[0]}\n- Fitness: {gene_pool.genes[0].evaluate(gene_pool.inputs, gene_pool.targets)}"
    )


if __name__ == "__main__":
    main()
