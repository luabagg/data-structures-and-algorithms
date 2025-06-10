def print_matrix(a, b):
    """
    Print the augmented matrix [A|b] in a readable format.
    """
    n = len(b)
    for i in range(n):
        row = "  ".join(f"{a[i][j]:8.4f}" for j in range(n))
        print(f"[ {row} ] | {b[i]:8.4f}")
    print()

def gauss_elimination_verbose(a, b):
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.
    Prints each step of the elimination and back substitution process.

    Parameters:
    a -- Coefficient matrix (list of lists, will be modified in-place)
    b -- Right-hand side vector (list, will be modified in-place)

    Returns:
    x -- Solution vector (list)
    """
    n = len(b)

    print("Initial augmented matrix:")
    print_matrix(a, b)

    # Forward elimination
    for i in range(n):
        # Partial pivoting: find the row with the largest value in column i
        max_row = i + max(range(n - i), key=lambda k: abs(a[i + k][i]))
        if abs(a[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        # Swap rows if needed
        if max_row != i:
            a[i], a[max_row] = a[max_row], a[i]
            b[i], b[max_row] = b[max_row], b[i]
            print(f"Swapped row {i} with row {max_row}")
            print_matrix(a, b)

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]
            print(f"Eliminated row {j} using row {i} with factor {factor:.4f}")
            print_matrix(a, b)

    # Back substitution
    x = [0 for _ in range(n)]
    print("Back substitution:")
    for i in reversed(range(n)):
        s = sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / a[i][i]
        print(f"x[{i}] = {x[i]:.4f}")

    print("\nFinal solution:", x)
    return x

if __name__ == "__main__":
    A = [
        [1, 2, 1, 1, 2],
        [0, 1, 2, 2, 3],
        [1, 0, 2, 3, 2],
        [1, 1, 1, 2, 1],
        [2, 1, 0, 1, 1]
    ]
    B = [31, 31, 27, 23, 22]

    solution = gauss_elimination_verbose([row[:] for row in A], B[:])