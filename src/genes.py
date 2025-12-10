from dataclasses import dataclass
from math import cos, exp, pi, sqrt
from random import uniform, gauss
from typing import Callable, List, Tuple
from constants import (
    APPROXIMATION_THRESHOLD,
    CHILDREN_SIZE,
    MAX_ITERATIONS,
    POPULATION_SIZE,
)


def mse(predictions: List[float], targets: List[float]) -> float:
    """
    Calculate the Mean Squared Error between predictions and targets.

    :param predictions: List of predicted values
    :type predictions: list[float]
    :param targets: List of target values
    :type targets: list[float]
    :return: Mean Squared Error
    :rtype: float
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    error = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    return error


@dataclass
class Gene:
    """
    A single gene in the model. You should not instantiate this directly.
    Use GenePool instead
    """

    a: float
    b: float
    c: float
    sigma_a: float
    sigma_b: float
    sigma_c: float
    f: Callable[["Gene"], float]

    mse: float | None = None

    def _evaluate(self, i: float) -> float:
        return self.a * (i**2 - self.b * cos(self.c * pi * i))

    def evaluate(self, inputs: List[float], targets: List[float]) -> float:
        """
        :seealso: evaluate

        :param self: Description
        :param inputs: List of input values
        :param targets: List of target values
        :return: The Mean Squared Error between the gene's output and the target outputs
        """
        if self.mse is not None:
            return self.mse
        self.mse = mse([self._evaluate(i) for i in inputs], targets)
        return self.mse

    def get_offsprings(self, n: int) -> list["Gene"]:
        """
        Generates n offspring genes from this gene.
        Implements Gaussian mutation.

        :param n: Number of offspring genes to generate
        :type n: int
        :return: List of generated offspring genes
        :rtype: list[Gene]
        """
        tau_1 = 1 / sqrt(2 * 3)
        r_sigma_1 = tau_1 * gauss(0, 1)
        tau_2 = 1 / sqrt(2 * sqrt(3))

        offsprings = [
            Gene(
                a=self.a + gauss(0, self.sigma_a),
                b=self.b + gauss(0, self.sigma_b),
                c=self.c + gauss(0, self.sigma_c),
                sigma_a=self.sigma_a * exp(r_sigma_1 + tau_2 * gauss(0, 1)),
                sigma_b=self.sigma_b * exp(r_sigma_1 + tau_2 * gauss(0, 1)),
                sigma_c=self.sigma_c * exp(r_sigma_1 + tau_2 * gauss(0, 1)),
                f=self.f,
            )
            for _ in range(n)
        ]

        return offsprings


class GenePool:
    """
    This class manages a pool of genes
    """

    data: List[Tuple[float, float]]
    inputs: List[float]
    targets: List[float]

    f: Callable[[Gene], float]

    genes: list[Gene]
    current_iteration: int
    previous_genes: list[Gene] | None

    def __init__(
        self, n: int, data: List[Tuple[float, float]], f: Callable[[Gene], float]
    ) -> None:
        """
        Docstring for __init__

        :param self: Description
        :param n: The number of parent genes to be generated
        :type n: int
        :param data: The raw data for the function to be approximated
        :type data: List[Tuple[float, float]]
        :param f: A function f(x) that takes in a Gene and returns a float. This is the function to be approximated
        :type f: Callable[[Gene], float]
        """
        self.data = data
        self.inputs = [pair[0] for pair in data]
        self.targets = [pair[1] for pair in data]
        self.f = f
        self.genes = GenePool.gen_genes(n, f)
        self.previous_genes = None
        self.current_iteration = 0

    def gen_genes(n: int, f: Callable[[Gene], float]) -> list[Gene]:
        """
        Generates a list of Gene objects with default parameters.

        :param n: Number of Gene objects to generate
        :type n: int
        :return: List of Gene objects
        :rtype: list[Gene]
        """
        return [
            Gene(
                a=uniform(-10, 10),
                b=uniform(-10, 10),
                c=uniform(-10, 10),
                sigma_a=uniform(0, 10),
                sigma_b=uniform(0, 10),
                sigma_c=uniform(0, 10),
                f=f,
            )
            for _ in range(n)
        ]

    def select_genes(self, n: int, genes: list[Gene]) -> list[Gene]:
        """
        Selects the n best Gene objects from the provided list based on their evaluation (lower is better).

        :param n: Number of Gene objects to select
        :type n: int
        :param genes: List of Gene objects
        :type genes: list[Gene]
        :return: List of selected Gene objects
        :rtype: list[Gene]
        """
        sorted_genes = sorted(
            genes, key=lambda gene: gene.evaluate(self.inputs, self.targets)
        )
        return sorted_genes[:n]

    def generate_offsprings_from_parents(self) -> list[Gene]:
        """
        Generates the offsprings from the parents.

        :param self: Description
        """
        selected_genes = self.select_genes(CHILDREN_SIZE, self.genes)
        offsprings = [
            offspring for gene in selected_genes for offspring in gene.get_offsprings(5)
        ]
        return offsprings

    def stop_condition_met(self) -> bool:
        """
        Returns whether the stop condition is met from the gene's side

        :param self: A GenePool
        :return: Whether the learning loop should stop
        :rtype: bool
        """

        if self.current_iteration >= MAX_ITERATIONS:
            return True

        if self.previous_genes is None:
            return False

        # Genes are already sorted by fitness
        min_parents = self.previous_genes[0].evaluate(self.inputs, self.targets)
        min_offsprings = self.genes[0].evaluate(self.inputs, self.targets)

        return abs(min_parents - min_offsprings) < APPROXIMATION_THRESHOLD

    def evolve(self):
        """
        Evolves the gene pool to the next generation.

        :param self: Description
        """
        offsprings = self.select_genes(
            POPULATION_SIZE, self.generate_offsprings_from_parents()
        )

        self.previous_genes = self.genes
        self.genes = offsprings
        self.current_iteration += 1
