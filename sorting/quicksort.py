from commons import arraycommons

# Python program for implementation of Quicksort Sort

# This implementation utilizes pivot as the last element in the nums list
# It has a pointer to keep track of the elements smaller than the pivot
# At the very end of partition() function, the pointer is swapped with the pivot
# to come up with a "sorted" nums relative to the pivot

total_it: int = 0

# Function to find the partition position
def partition(arr, low, high):
    global total_it

    # choose the rightmost element as pivot
    pivot = arr[high]

    # pointer for greater element
    i = low - 1

    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if arr[j] <= pivot:
            total_it += 1

            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            (arr[i], arr[j]) = (arr[j], arr[i])

    # Swap the pivot element with the greater element specified by i
    (arr[i + 1], arr[high]) = (arr[high], arr[i + 1])

    # Return the position from where partition is done
    return i + 1

def sort(arr, low, high):
    if low < high:
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pivot = partition(arr, low, high)

        # Recursive call on the left of pivot
        arr = sort(arr, low, pivot - 1)

        # Recursive call on the right of pivot
        arr = sort(arr, pivot + 1, high)

    return arr

array_size = int(input("Enter the size of array:"))
array = arraycommons.populateArray(array_size)

print("Before sorting:")
arraycommons.printArray(array)

array = sort(array, 0, array_size - 1)

print(f"Loop ran for {total_it} iterations \n")

print("Sorted array is:")
arraycommons.printArray(array)
