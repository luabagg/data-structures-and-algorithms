# Data Structures and Algorithms

This repository contains my studies on data structures, algorithms, and numerical methods.

The goal is to keep the organization **language-agnostic**: each topic lives under
`algorithms/`, and implementations in different languages sit side-by-side in the same
topic folder. Right now the repo has Python and C implementations, but new languages can
be added without moving existing files.

## Repository structure

```
algorithms/
  hello_world/          # First steps in each language
  commons/              # Shared helpers (e.g. array utilities)
  sorting/              # Sorting algorithms
  general/              # Small general-purpose exercises
  numerical_methods/    # Numerical analysis and scientific computing
    curve_fitting/
    integration/
    linear_system/
    polynomial_interpolation/
    root_finding/
out/                    # Compiled C binaries (ignored by git)
```

## Running and compiling

### Python

Create a virtual environment and install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run any module from the repository root using its dotted package path:

```sh
python3 -m algorithms.hello_world.hello_world
python3 -m algorithms.sorting.bubble_sort
python3 -m algorithms.numerical_methods.integration.trapezoidal_rule
```

Or use the Makefile helper:

```sh
make MODULE=algorithms.sorting.bubble_sort runpy
```

### C

Compile all available C programs:

```sh
make
```

Run a specific binary:

```sh
make BIN=bubble_sort run
```

Clean compiled artifacts:

```sh
make clean
```

## To do

- Linked List
- Hash Table
- Binary Tree

## Contributing

Contributions are welcome. When adding a new algorithm, create a topic folder under the
appropriate `algorithms/` sub-directory and place each language implementation there.
Keep file names in `snake_case` and match the algorithm name across languages so it is
easy to compare implementations.

See [AGENTS.md](./AGENTS.md) for the project's AI assistant guidelines.

## Useful links

- [Computer Science Roadmap](https://roadmap.sh/computer-science)
