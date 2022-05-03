#!/usr/bin/env python
import subprocess
import sys
from collections.abc import Iterable

import numpy as np

ROOTS = sys.argv[1:]

TARGET = 'examples.tex'

EXAMPLES_WIDTH = 0.9
EXAMPLES_TEMPLATE = r'''
\documentclass[lab1-report.tex]{subfiles}

\begin{document}
    \begin{examples}{%.2f}{%.2f}{%.2f}
%s
    \end{examples}
\end{document}
'''

EXAMPLE_COLUMNS = 3
EXAMPLE_MIN_COLUMN_WIDTH = 8
EXAMPLE_TEMPLATE = r'        \example{\strut{}%s}{\strut{}%s}{\strut{}%s}'


def count_width(text: str) -> int:
    width, start = 0, 0
    while True:
        end = text.find('\n', start)
        if end == -1:
            return max(width, len(text) - start)
        width = max(width, end - start)
        start = end + 1


def tex_escape(text: str) -> str:
    return text.rstrip().replace('\n', r'\newline\strut{}').replace(' ', r'~')


def render_example(root: str) -> tuple[str, list[int]]:
    columns, width = [], []
    input_path = root + '/input.txt'
    with open(input_path, 'r') as file:
        data = file.read()
        columns.append(tex_escape(data))
        width.append(count_width(data))
    with open(root + '/output.txt', 'r') as file:
        data = file.read()
        columns.append(tex_escape(data))
        width.append(count_width(data))
    data = subprocess.run(['src/main.py', '-f', input_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    columns.append(tex_escape(data))
    width.append(count_width(data))
    return (EXAMPLE_TEMPLATE % tuple(columns)), width


def render_examples(roots: Iterable[str]) -> str:
    examples = []
    columns = np.ones(EXAMPLE_COLUMNS)
    columns *= EXAMPLE_MIN_COLUMN_WIDTH
    for example, width in map(render_example, roots):
        examples.append(example)
        np.maximum(columns, width, out=columns)
    columns *= EXAMPLES_WIDTH / columns.sum()
    return EXAMPLES_TEMPLATE % (*columns, '\n'.join(examples))


if __name__ == '__main__':
    with open('examples.tex', 'w') as file:
        file.write(render_examples(ROOTS))
