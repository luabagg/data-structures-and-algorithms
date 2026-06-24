import numpy as np

def gauss_jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve the linear system Ax = b using the Gauss-Jacobi iterative method.

    Parameters:
    A : numpy.ndarray
        Coefficient matrix (n x n)
    b : numpy.ndarray
        Right-hand side vector (n,)
    x0 : numpy.ndarray, optional
        Initial guess for the solution (n,). If None, uses zeros.
    tol : float, optional
        Tolerance for the stopping criterion (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)

    Returns:
    result : dict
        Dictionary with keys:
            'solution'   : Solution vector (numpy.ndarray)
            'iterations' : Number of iterations performed (int)
            'converged'  : Boolean indicating if the method converged (bool)
            'residual'   : Final residual norm (float)
    """
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    x_new = np.zeros_like(x)

    for iteration in range(1, max_iter + 1):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            residual = np.linalg.norm(np.dot(A, x_new) - b, ord=np.inf)
            return {
                'solution': x_new,
                'iterations': iteration,
                'converged': True,
                'residual': residual
            }
        x[:] = x_new
    residual = np.linalg.norm(np.dot(A, x_new) - b, ord=np.inf)
    return {
        'solution': x_new,
        'iterations': max_iter,
        'converged': False,
        'residual': residual
    }

# Example usage
if __name__ == "__main__":
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8.]])
    B = np.array([6., 25., -11., 15.])
    x0 = np.zeros(4)
    result = gauss_jacobi(A, B, x0, tol=1e-8, max_iter=1000)
    print("Solution:", result['solution'])
    print("Iterations:", result['iterations'])
    print("Converged:", result['converged'])
    print("Final residual norm:", result['residual'])
    print("Verification: A·x =", np.dot(A, result['solution']))
    print("b =", B)
