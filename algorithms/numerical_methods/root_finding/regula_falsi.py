import math

def false_position(f, a, b, tol=1e-6, max_iter=100):
    """
    Find a root of the function f(x) = 0 in the interval [a, b] using the Regula Falsi (False Position) method.

    Parameters:
    f        -- Function for which the root is sought (callable)
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
        raise ValueError("The function must change sign in the interval [a, b].")
    
    iter_count = 0
    fa = f(a)
    fb = f(b)
    error = tol + 1
    c_previous = None
    c = a  # Initialize c in case the loop does not run
    fc = f(c)
    
    while error > tol and iter_count < max_iter:
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        if c_previous is not None:
            error = abs(c - c_previous)
        else:
            error = abs(fc)

        if abs(fc) < tol:
            break
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
        c_previous = c
        iter_count += 1
    
    return {
        "root": c,
        "function_value": fc,
        "iterations": iter_count,
        "converged": abs(fc) < tol or error < tol,
    }

def example1(x):
    """
    Example function: f(x) = x + 3*cos(x) - exp(x)
    """
    return x + 3*math.cos(x) - math.exp(x)

if __name__ == "__main__":
    result = false_position(example1, 0, 1, tol=0.001)
    print(f"Approximate root: {result['root']}")
    print(f"Function value at the root: {result['function_value']}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")