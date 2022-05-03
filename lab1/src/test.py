from __future__ import annotations
import timeit
from method import *


def generate_random_array(shape: int | tuple[int, ...], scale: float,
                          integral: bool, positive: bool) -> np.ndarray:
    matrix = (2 * np.random.random(shape) - 1) * scale
    if integral:
        matrix = np.fix(matrix)
    if positive:
        np.abs(matrix, out=matrix)
    return matrix


def generate_random_system(size: int, scale: int, integral: bool, positive: bool) -> LinearSystem:
    augmented = generate_random_array((size, size + 1), scale, integral, positive)
    return LinearSystem.from_augmented(augmented)


def generate_test_case(size: int, scale: int = 100, integral: bool = False,
                       positive: bool = False) -> tuple[LinearSystem, np.ndarray]:
    solution = generate_random_array(size, scale/size, integral, positive)
    coefficients = generate_random_array((size, size), scale/size, integral, positive)
    magnify_to_strictly_diagonally_dominant(coefficients)
    constants = np.matmul(coefficients, solution)
    return LinearSystem(coefficients, constants), solution


def execution_time(*args: tuple[LinearSystem, np.ndarray, np.float64, int], number: int = 0) -> float:
    system, initial, accuracy, iterations = args
    setup = 'from method import solve_jacobi'
    target = 'solve_jacobi(system, initial, accuracy, iterations)'
    timer = timeit.Timer(target, setup, globals=locals())
    if number == 0:
        number, _ = timer.autorange()
    return min(timer.repeat(repeat=5, number=number)) / number
