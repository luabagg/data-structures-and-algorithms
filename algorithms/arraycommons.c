#include "arraycommons.h"

void populateArray(int *buf, int array_size) {
    srand(time(NULL));

    int i;
    for (i = 0; i < array_size; i++) {
        // Generate number between 0 to 99
        buf[i] = rand() % 100;
    }
}

int printArray(int array[], int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", array[i]);
	}
	printf("\n\n");
}