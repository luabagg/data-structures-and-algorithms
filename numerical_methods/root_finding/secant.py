import math
import numpy


def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Find a root of the function f(x) = 0 using the Secant method.

    Parameters:
    f        -- Function for which the root is sought (callable)
    x0, x1   -- Initial approximations (floats)
    tol      -- Tolerance for stopping criterion (float, default 1e-6)
    max_iter -- Maximum number of iterations (int, default 100)

    Returns:
    result -- Dictionary with keys:
        'root'           : Approximated root (float)
        'function_value' : Value of f at the root (float)
        'iterations'     : Number of iterations performed (int)
        'converged'      : Boolean indicating if the method converged (bool)
    """
    fx0 = f(x0)
    fx1 = f(x1)
    iteration = 0
    while iteration < max_iter:
        # Check if the denominator is too close to zero
        if abs(fx1 - fx0) < tol:
            return {
                'root': x1,
                'iterations': iteration,
                'function_value': fx1,
                'converged': abs(fx1) < tol
            }
        # Secant method formula
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        # Update values for next iteration
        x0, x1 = x1, x_new
        fx0, fx1 = fx1, f(x_new)
        iteration += 1
        # Check for convergence
        if abs(fx1) < tol or (iteration > 0 and abs(x1 - x0) < tol):
            break
    return {
        'root': x1,
        'iterations': iteration,
        'function_value': fx1,
        'converged': iteration < max_iter
    }


if __name__ == "__main__":
    def example_function(x):
        return numpy.cos(x) - x

    result = secant_method(example_function, 0.5, 1, tol=1e-4)
    print(f"Approximate root: {result['root']}")
    print(f"Function value at the root: {result['function_value']}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")