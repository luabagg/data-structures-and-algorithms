import math
import numpy


def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant Method to find the root of a function f(x).
    
    Parameters:
    f -- Function for which we want to find a root
    x0, x1 -- Initial approximations
    tol -- Tolerance (stopping criterion)
    max_iter -- Maximum number of iterations
    
    Returns:
    Dictionary containing:
    - root: Approximated root
    - iterations: Number of iterations performed
    - function_value: Value of f(x) at the approximated root
    - converged: Boolean indicating if the method converged
    """
    
    fx0 = f(x0)
    fx1 = f(x1)

    print(f"Initial approximations: x0 = {x0}, f(x0) = {fx0}")
    print(f"Initial approximations: x1 = {x1}, f(x1) = {fx1}")

    
    iteration = 0
    
    while iteration < max_iter:
        # Check if the denominator is too close to zero

        if abs(fx1 - fx0) < tol:
            return {
                "root": x1,
                "iterations": iteration,
                "function_value": fx1,
                "converged": abs(fx1) < tol
            }

        # Secant method formula
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        print(f"New approximation: {x_new}, f(x_new): {f(x_new)}")
        
        # Update values for next iteration
        x0, x1 = x1, x_new
        fx0, fx1 = fx1, f(x_new)
        
        iteration += 1
        print("error", abs(fx1 - fx0))

        # Check for convergence
        if abs(fx1) < tol or (iteration > 0 and abs(x1 - x0) < tol):
            break
    
    return {
        "root": x1,
        "iterations": iteration,
        "function_value": fx1,
        "converged": iteration < max_iter
    }

# Example usage
def example_function(x):
    return numpy.cos(x) - x

result = secant_method(example_function, .5, 1, tol=1e-4)

print(f"Approximate root: {result['root']}")
print(f"Function value at the root: {result['function_value']}")
print(f"Number of iterations: {result['iterations']}")
print(f"Converged: {result['converged']}")