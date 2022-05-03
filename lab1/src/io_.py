from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Optional, Any
import sys
import numpy as np
from method import LinearSystem, IterativeMethodResult


def decimal_width(numbers: np.ndarray | int) -> np.ndarray | int:
    return np.floor(np.maximum(0, np.log10(np.maximum(1, np.abs(numbers))))).astype(np.int64) + 1 + (numbers < 0)


def try_parse(data: str, converter: Callable) -> Optional[Any]:
    try:
        return converter(data)
    except ValueError:
        return None


def foolproof_read(prompt: str, converter: Callable, reprompt: Optional[str] = None) -> Any:
    value = try_parse(input(prompt), converter)
    while value is None:
        value = try_parse(input(reprompt or prompt), converter)
    return value


def read_system_interactively(sep: str = ' ', decimal: str = '.') -> LinearSystem:
    size = foolproof_read('Enter matrix size: ', int, 'Invalid value. Enter again: ')
    augmented = np.empty((size, size + 1), np.float64)

    def parse_line(line: str) -> Optional[Iterable[np.float64]]:
        values = line.replace(decimal, '.').split(sep)
        return map(np.float64, values) if len(values) == (size + 1) else None

    for i in range(size):
        row = foolproof_read(f'row {i + 1}: ', parse_line)
        for j, value in enumerate(row):
            augmented[i][j] = value
    return LinearSystem.from_augmented(augmented)


def read_system_from_file(path: str, sep: str = ' ', decimal: str = '.') -> LinearSystem:
    with open(path, 'r') as file:
        size = int(file.readline())
        system = LinearSystem.empty(size, np.float64)
        for i in range(size):
            row = file.readline().replace(decimal, '.').split(sep)
            for j in range(size):
                system.coefficients[i][j] = np.float64(row[j])
            system.constants[i] = np.float64(row[size])
        return system


def report_system(system: LinearSystem, precision: int = 3):
    size = len(system.constants)
    width = (decimal_width(system.coefficients).max(axis=0) + precision + (precision != 0),
             decimal_width(system.constants).max() + 1 + precision)
    coefs_pattern = ' '.join(f'{{: >{width[0][i]}.{precision}f}}' for i in range(size))
    row_pattern = '\t' + coefs_pattern + f' | {{: >{width[1]}.{precision}f}}'
    for coef_row, const in zip(system.coefficients, system.constants.flat):
        print(row_pattern.format(*coef_row, const))


def report_result(result: IterativeMethodResult, solution_precision: int = 3, error_precision: int = 1):
    if not result.has_converged:
        print(f'Method has not converged after {result.iterations_count} steps')
    else:
        print(f'Method converged in {result.iterations_count} steps')
        width = decimal_width(result.solution).max() + 1 + solution_precision
        pattern = f'\t{{: >{width}.{solution_precision}f}} ({{:+.{error_precision}e}})'
        for value, error in zip(result.solution.flat, result.errors.flat):
            print(pattern.format(value, error))


def report_error(description: str):
    print(description, file=sys.stderr)
    exit(1)


def report_failure(description: str):
    print(description, 'Method is not applicable', sep='\n')
    exit()
