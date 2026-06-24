import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()

    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        print(f"Iteration {iteration}: x = {x_old}")
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
            print(f"Updated x[{i}] = {x[i]} using s1 = {s1}, s2 = {s2}")

        delta = np.linalg.norm(x - x_old, ord=np.inf)
        residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
        print(f"it: {iteration}, residual: {residual}, delta: {delta}")
        if delta < tol:
            return {
                'solution': x,
                'iterations': iteration,
                'converged': True,
                'residual': residual
            }

    residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
    return {
        'solution': x,
        'iterations': max_iter,
        'converged': False,
        'residual': residual
    }

# Exemplo de uso
if __name__ == "__main__":
    A = np.array([[10.0, 2.0, 1.0],
                  [1.0, 5.0, 1.0],
                  [2.0, 3.0, 10.0]])
    b = np.array([7.0, -8.0, 6])
    x0 = np.array([0.70, -1.6, 0.6])
    resultado = gauss_seidel(A, b, x0, tol=0.05, max_iter=100)
    print("Solução:", resultado['solution'])
    print("Iterações:", resultado['iterations'])
    print("Convergiu?", resultado['converged'])
    print("Resíduo final:", resultado['residual'])