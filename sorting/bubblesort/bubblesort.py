from commons import arraycommons

def bubbleSort(arr):
    swapped = False
    totalIt = 0
    real_array_size = len(arr) - 1
    for i in range(real_array_size):
        for j in range(0, real_array_size-i):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        totalIt += 1

        if not swapped:
            break

    print("Loop ran for %d iterations \n" % totalIt)
    return arr

array_size = int(input("Enter the size of array:"))
array = arraycommons.populateArray(array_size)

print("Before sorting:")
arraycommons.printArray(array)

array = bubbleSort(array)

print("Sorted array is:")
arraycommons.printArray(array)