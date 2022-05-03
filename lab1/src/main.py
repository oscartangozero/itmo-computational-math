#!/usr/bin/env python

import argparse
from test import *
from io_ import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-l', '--limit', type=int, default=10 ** 9, metavar='L', help='maximum number of iterations')
parser.add_argument('-a', '--accuracy', type=float, default=0.000001, metavar='E', help='maximum acceptable error')

INITIAL_APPROXIMATION_TYPES = {
    'zeros': lambda system: np.zeros_like(system.constants),
    'constants': lambda system: np.copy(system.constants),
}
parser.add_argument('--init', choices=INITIAL_APPROXIMATION_TYPES.keys(), default='zeros',
                    metavar='I', help='type of initial approximation')

input_method = parser.add_argument_group('input method').add_mutually_exclusive_group()
input_method.add_argument('-f', '--file', type=str, metavar='path')
input_method.add_argument('-g', '--generate', type=int, metavar='size')

gen_options = parser.add_argument_group('generator options')
gen_options.add_argument('-i', '--integral', action='store_true', help='are the generated values purely integers')
gen_options.add_argument('-p', '--positive', action='store_true', help='are the generated values purely positive')
gen_options.add_argument('-s', '--scale', type=int, default=100, metavar='factor',
                         help='upper bound of the absolute values of the coefficient matrix and the solution vector')

parse_options = parser.add_argument_group('parser options')
parse_options.add_argument('--separator', type=str, default=None, metavar='s')
parse_options.add_argument('--decimal', type=str, default='.', metavar='d')

args = parser.parse_args()
initial = INITIAL_APPROXIMATION_TYPES[args.init]
if args.generate is not None:
    system, solution = generate_test_case(args.generate, args.scale, args.integral, args.positive)
    solution_precision = 0 if args.integral else 3
    print('Linear system generated: ')
    report_system(system, solution_precision)
    result = solve_jacobi(system, initial(system), args.accuracy, args.limit)
    report_result(result, solution_precision)
else:
    system = None
    if args.file is None:
        system = read_system_interactively(args.separator, args.decimal)
    else:
        try:
            system = read_system_from_file(args.file, args.separator, args.decimal)
        except ValueError:
            report_error('Error: invalid file format')
        except OSError:
            report_error('Error: file is inaccessible')

    if not is_strictly_diagonally_dominant(system.coefficients):
        if np.linalg.det(system.coefficients) == 0:
            matrix_rank = np.linalg.matrix_rank(system.coefficients)
            augmented = np.hstack((system.coefficients, system.constants))
            augmented_rank = np.linalg.matrix_rank(augmented)
            if matrix_rank == augmented_rank:
                report_failure("System has infinitely many solutions")
            else:
                report_failure("Linear system has no solution")
        print('Coefficients matrix is not strictly diagonally dominant')
        system = rearrange_to_strictly_diagonally_dominant(system)
        if system is not None:
            print('Successfully rearranged: ')
            report_system(system)
        else:
            report_failure('Unable to rearrange')

    result = solve_jacobi(system, initial(system), args.accuracy, args.limit)
    report_result(result)
