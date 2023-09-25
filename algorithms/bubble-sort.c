#include <stdio.h>
#include "arraycommons.h"
#include <stdbool.h>

int swap(int *prev, int *next) {
	int temp = *prev;
	*prev = *next;
	*next = temp;
}

int sort(int array[], int array_size) {
	bool swapped;
	int totalIt = 0;
	int real_array_size = array_size - 1;
	for (int i = 0; i < real_array_size; i++) {
		swapped = false;
		for (int j = 0; j < real_array_size - i; j++) {
			if (array[j] > array[j + 1]) {
				swap(&array[j], &array[j + 1]);
				swapped = true;
			}
			totalIt++;
		}
		if (swapped == false) {
            break;
		}
	}

	printf("Loop ran for %d iterations \n\n", totalIt);
}

int main() {
	int array_size;

	printf("Enter the size of array::");
	scanf("%d", &array_size);

	int array[array_size];

	populateArray(array, array_size);

	printf("Before sorting: \n");
	printArray(array, array_size);

	int res = sort(array, array_size);
	printf("Sorted array is: \n");
	printArray(array, array_size);
}