from commons import arraycommons

def sort(arr):
    swapped = False
    total_it = 0
    real_array_size = len(arr) - 1
    for i in range(real_array_size):
        for j in range(0, real_array_size-i):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            total_it += 1
        if not swapped:
            break

    print(f"Loop ran for {total_it} iterations \n")
    return arr

array_size = int(input("Enter the size of array:"))
array = arraycommons.populateArray(array_size)

print("Before sorting:")
arraycommons.printArray(array)

array = sort(array)

print("Sorted array is:")
arraycommons.printArray(array)
