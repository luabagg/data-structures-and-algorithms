from algorithms.commons import array_commons


def sort(arr):
    swapped = False
    total_it = 0
    real_array_size = len(arr) - 1
    for i in range(real_array_size):
        for j in range(0, real_array_size - i):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            total_it += 1
        if not swapped:
            break

    print(f"Loop ran for {total_it} iterations \n")
    return arr


if __name__ == "__main__":
    array_size = int(input("Enter the size of array:"))
    array = array_commons.populate_array(array_size)

    print("Before sorting:")
    array_commons.print_array(array)

    array = sort(array)

    print("Sorted array is:")
    array_commons.print_array(array)
