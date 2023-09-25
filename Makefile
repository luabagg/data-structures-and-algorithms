CC = gcc
CFLAGS = -g
OUTPUT_DIR=out
RM = rm -f
BIN=hello-world
INLCUDE_FILES=algorithms/arraycommons.c

default: all

# make BIN=<binary> run
run:
	./$(OUTPUT_DIR)/$(BIN)

all: hello-world bubble-sort

hello-world:
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/hello-world hello-world.c

bubble-sort:
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/bubble-sort algorithms/bubble-sort.c $(INLCUDE_FILES)

clean:
	$(RM) $(OUTPUT_DIR)/*