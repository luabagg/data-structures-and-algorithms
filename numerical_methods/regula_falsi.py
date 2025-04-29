import math

def false_position(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("The function must change sign in the interval [a, b]")
    
    iter_count = 0
    fa = f(a)
    fb = f(b)
    error = tol + 1
    c_previous = None
    
    while error > tol and iter_count < max_iter:
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        if c_previous is not None:
            error = abs(c - c_previous)

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
        "converged": iter_count < max_iter,
    }

def example1(x):
    return x + 3*math.cos(x) - math.exp(x)


result = false_position(example1, 0, 1, tol=0.001)
print(f"Approximate root: {result['root']}")
print(f"Function value at the root: {result['function_value']}")
print(f"Number of iterations: {result['iterations']}")