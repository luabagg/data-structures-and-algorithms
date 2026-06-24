def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find a root of the function f(x) = 0 in the interval [a, b] using the Bisection method.

    Parameters:
    f        -- Continuous function for which the root is sought (callable)
    a, b     -- Interval endpoints (floats), must satisfy f(a)*f(b) < 0
    tol      -- Tolerance for stopping criterion (float, default 1e-6)
    max_iter -- Maximum number of iterations (int, default 100)

    Returns:
    result -- Dictionary with keys:
        'root'           : Approximated root (float)
        'function_value' : Value of f at the root (float)
        'iterations'     : Number of iterations performed (int)
        'converged'      : Boolean indicating if the method converged (bool)
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Bolzano's Theorem is not guaranteed: f(a) and f(b) must have opposite signs.")

    for iteration in range(1, max_iter + 1):
        c = (a + b) / 2  # Midpoint
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return {
                'root': c,
                'function_value': fc,
                'iterations': iteration,
                'converged': True
            }
        if f(a) * fc < 0:
            b = c  # Root is in [a, c]
        else:
            a = c  # Root is in [c, b]
    # If max_iter reached
    c = (a + b) / 2
    fc = f(c)
    return {
        'root': c,
        'function_value': fc,
        'iterations': max_iter,
        'converged': False
    }

# Example usage
if __name__ == "__main__":
    f = lambda x: x**3 + x**2 - 10
    result = bisection(f, 0, 2, 1e-8)
    print(f"Approximate root: {result['root']:.10f}")
    print(f"Function value at the root: {result['function_value']}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")