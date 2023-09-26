# C options
CC = gcc
CFLAGS = -g
OUTPUT_DIR=out
RM = rm -f
BIN=helloworld
INLCUDE_FILES=commons/arraycommons.c

# Python options
MODULE=helloworld

default: all

# make BIN=<binary> run
run:
	./$(OUTPUT_DIR)/$(BIN)

runpy:
	python3 -m $(MODULE)

all: helloworld bubblesort

helloworld:
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/helloworld helloworld.c

bubblesort:
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/bubblesort sorting/bubblesort/bubblesort.c $(INLCUDE_FILES)

clean:
	$(RM) $(OUTPUT_DIR)/*