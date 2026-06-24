# C options
CC = gcc
CFLAGS = -g -Wall -Wextra
INCLUDE = -I algorithms/commons
OUTPUT_DIR = out
RM = rm -f

# Python options
MODULE = algorithms.hello_world.hello_world

# Default binary to run
BIN = hello_world

default: all

all: $(OUTPUT_DIR)/hello_world $(OUTPUT_DIR)/bubble_sort

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/hello_world: algorithms/hello_world/hello_world.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $<

$(OUTPUT_DIR)/bubble_sort: algorithms/sorting/bubble_sort.c algorithms/commons/array_commons.c | $(OUTPUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^

# make BIN=<binary> run
run: all
	./$(OUTPUT_DIR)/$(BIN)

# make MODULE=<dotted.module.path> runpy
runpy:
	python3 -m $(MODULE)

clean:
	$(RM) $(OUTPUT_DIR)/*
