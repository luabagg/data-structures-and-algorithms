from sympy import symbols, diff, lambdify

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            raise ValueError(f"Zero derivative at x = {x}. Method failed.")
        
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            print(f"Converged in {i + 1} iterations.")
            return x_new
        
        x = x_new

    raise ValueError("Maximum number of iterations exceeded.")

x = symbols('x')

# Example: f(x) = x^2 - 2 (root: sqrt(2))
f_expr = x**2 - 2
df_expr = diff(f_expr, x)

# Convert symbolic expressions to numeric functions (for use in the method)
f = lambdify(x, f_expr, 'math')
df = lambdify(x, df_expr, 'math')

print(f"Function: {f_expr}")
print(f"Derivative: {df_expr}")

root = newton_raphson(f, df, x0=1.25, tol=1e-4, max_iter=100)
print(f"The approximate root is: {root}")
