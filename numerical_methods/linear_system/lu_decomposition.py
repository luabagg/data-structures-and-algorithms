import numpy as np

def lu_decomposition_pivot(A):
    """
    Perform LU decomposition with partial pivoting on matrix A.
    Decomposes PA = LU, where P is a permutation matrix, L is lower triangular, and U is upper triangular.

    Parameters:
    A -- Square matrix (numpy.ndarray)

    Returns:
    P -- Permutation matrix (numpy.ndarray)
    L -- Lower triangular matrix (numpy.ndarray)
    U -- Upper triangular matrix (numpy.ndarray)
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    P = np.eye(n)

    for i in range(n):
        # Find the index of the row with the largest absolute value in column i
        pivot = np.argmax(np.abs(U[i:, i])) + i
        if U[pivot, i] == 0:
            raise ValueError("Singular matrix.")
        # Swap rows in U
        U[[i, pivot]] = U[[pivot, i]]
        # Swap rows in P
        P[[i, pivot]] = P[[pivot, i]]
        # Swap rows in L (only for previously computed columns)
        if i > 0:
            L[[i, pivot], :i] = L[[pivot, i], :i]
        # Elimination process
        for j in range(i+1, n):
            m = U[j, i] / U[i, i]
            L[j, i] = m
            U[j] = U[j] - m * U[i]
    return P, L, U

def solve_with_lu(P, L, U, b):
    """
    Solve the linear system Ax = b using LU decomposition with pivoting.

    Parameters:
    P, L, U -- The factors of A such that PA = LU
    b -- The right-hand side vector

    Returns:
    result -- Dictionary with keys:
        'solution'   : Solution vector (numpy.ndarray)
        'residual'   : Final residual norm (float)
    """
    # Step 1: Apply the permutation to b
    Pb = np.dot(P, b)

    # Step 2: Forward substitution to solve Ly = Pb
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]

    # Step 3: Back substitution to solve Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf) if 'A' in globals() else None
    return {'solution': x, 'residual': residual}

if __name__ == "__main__":
    # Example usage
    A = np.array([
        [3, -2, 5, 1],
        [-6, 4, -8, 1],
        [9, -8, 19, 1],
        [6, -4, -6, 15]
    ], dtype=float)
    P, L, U = lu_decomposition_pivot(A)
    print("P =\n", P)
    print("\nL =\n", L)
    print("\nU =\n", U)
    print("\nVerification: P·A =\n", np.dot(P, A))
    print("\nL·U =\n", np.dot(L, U))
    # Define a right-hand side vector b
    b = np.array([7, -9, 23, 11], dtype=float)
    # Solve the system Ax = b
    result = solve_with_lu(P, L, U, b)
    print("\nSolution x =\n", result['solution'])
    print("\nFinal residual norm =", result['residual'])
    print("\nVerification: A·x =\n", np.dot(A, result['solution']))