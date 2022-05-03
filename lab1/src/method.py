import numpy as np
from dataclasses import dataclass
from typing import Optional


def is_square_matrix(matrix: np.array) -> bool:
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def is_strictly_diagonally_dominant(matrix: np.ndarray) -> bool:
    """ Check if given matrix is strictly diagonally dominant,
        i.e. for any i: |a[i][i]| > sum |a[i][j]|, j != i """
    assert is_square_matrix(matrix)
    abs_values = np.abs(matrix)
    return np.all(2 * np.diagonal(abs_values) > np.sum(abs_values, axis=1))


def magnify_to_strictly_diagonally_dominant(matrix: np.ndarray):
    """ Increase the diagonals so that the matrix becomes strictly diagonally dominant.
    (integrality is preserved, increase factor is no more than the matrix linear size) """
    assert is_square_matrix(matrix)
    np.fill_diagonal(matrix, np.sum(np.abs(matrix) + (matrix.diagonal() == 0), axis=1))


@dataclass
class LinearSystem:
    coefficients: np.ndarray
    constants: np.ndarray

    @classmethod
    def empty(cls, size: int, dtype=np.float):
        return cls(coefficients=np.empty((size, size), dtype),
                   constants=np.empty((size, 1), dtype))

    @classmethod
    def from_augmented(cls, augmented: np.ndarray):
        coefficients = augmented[:, :-1]
        assert is_square_matrix(coefficients)
        return cls(np.copy(coefficients), np.copy(augmented[:, -1]))


def rearrange_to_strictly_diagonally_dominant(system: LinearSystem) -> Optional[LinearSystem]:
    """ Rearrange the rows of the matrix in such a way
        that it becomes strictly diagonally dominant """
    column_maxima_indexes = np.argmax(np.abs(system.coefficients), axis=0)
    if len(column_maxima_indexes) != len(np.unique(column_maxima_indexes)):
        return None
    system.coefficients = system.coefficients[np.ix_(column_maxima_indexes)]
    system.constants = system.constants[np.ix_(column_maxima_indexes)]
    if is_strictly_diagonally_dominant(system.coefficients):
        return system


@dataclass
class IterativeMethodResult:
    has_converged: bool
    iterations_count: int
    solution: Optional[np.ndarray] = None
    errors: Optional[np.ndarray] = None


def solve_jacobi(system: LinearSystem, initial_solution_approximation: np.ndarray,
                 max_error: float, iteration_limit: int) -> IterativeMethodResult:
    """ Solve a system of linear equations by the (iterative) Jacobi method """
    solution = initial_solution_approximation
    diagonals = np.diagonal(system.coefficients).reshape(solution.shape)
    for i in range(iteration_limit):
        errors = (system.constants - np.matmul(system.coefficients, solution)) / diagonals
        if np.all(np.abs(errors) <= max_error):
            return IterativeMethodResult(has_converged=True, iterations_count=i,
                                         solution=solution, errors=errors)
        solution += errors
    return IterativeMethodResult(has_converged=False, iterations_count=iteration_limit)
