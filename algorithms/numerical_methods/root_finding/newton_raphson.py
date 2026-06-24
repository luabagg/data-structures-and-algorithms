from sympy import symbols, diff, lambdify

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Find a root of the function f(x) = 0 using the Newton-Raphson method.

    Parameters:
    f        -- Function for which the root is sought (callable)
    df       -- Derivative of f (callable)
    x0       -- Initial guess (float)
    tol      -- Tolerance for stopping criterion (float, default 1e-6)
    max_iter -- Maximum number of iterations (int, default 100)

    Returns:
    result -- Dictionary with keys:
        'root'           : Approximated root (float)
        'function_value' : Value of f at the root (float)
        'iterations'     : Number of iterations performed (int)
        'converged'      : Boolean indicating if the method converged (bool)
    """
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"Zero derivative at x = {x}. Method failed.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return {
                'root': x_new,
                'function_value': f(x_new),
                'iterations': i,
                'converged': True
            }
        x = x_new
    return {
        'root': x,
        'function_value': f(x),
        'iterations': max_iter,
        'converged': False
    }

if __name__ == "__main__":
    x = symbols('x')
    # Example: f(x) = x^2 - 2 (root: sqrt(2))
    f_expr = x**2 - 2
    df_expr = diff(f_expr, x)
    # Convert symbolic expressions to numeric functions (for use in the method)
    f = lambdify(x, f_expr, 'math')
    df = lambdify(x, df_expr, 'math')
    print(f"Function: {f_expr}")
    print(f"Derivative: {df_expr}")
    result = newton_raphson(f, df, x0=1.25, tol=1e-4, max_iter=100)
    print(f"Approximate root: {result['root']}")
    print(f"Function value at the root: {result['function_value']}")
    print(f"Number of iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
