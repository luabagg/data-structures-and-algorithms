CC = gcc
CFLAGS = -g
OUTPUT_DIR=out
RM = rm -f

default: all

all: hello-world

hello-world: hello-world.c	
	$(CC) $(CFLAGS) -o $(OUTPUT_DIR)/hello-world hello-world.c

clean:
	$(RM) $(OUTPUT_DIR)/*