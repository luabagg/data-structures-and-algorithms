CC = gcc
CFLAGS = -g
RM = rm -f

default: all

all: hello-world

hello-world: hello-world.c	
	$(CC) $(CFLAGS) -o hello-world hello-world.c

clean:
	$(RM) hello-world