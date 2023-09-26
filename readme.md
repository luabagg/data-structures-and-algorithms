# Data Structures and Algorithms

This repository contain my studies about data structures and algorithms.

## Useful links

- [Computer Science Roadmap](https://roadmap.sh/computer-science)

## To Do

- Linked List
- Hash Table
- Binary Tree

## Running and Compiling

### C

To compile the files, use the available Makefile.

Example:

```sh
make
# or to compile a specific file
make helloworld
```

To run, simply execute the run command specifying the binary:

```sh
make BIN=helloworld run
```

### Python

First, install the requirements:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, execute the python module:

```sh
make MODULE=helloworld runpy
```

You can use any available module as "folder.subfolder.filename" - "sorting.bubblesort.bubblesort"

## Contributing
This project is an open-source project, and contributions from other developers are welcome. If you encounter any issues or have suggestions for improvement, please submit them on the project's GitHub page.
Any new implementation is welcome, and you can choose your preferred language.