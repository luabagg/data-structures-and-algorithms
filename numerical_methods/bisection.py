def bisection(f, a, b, tol=1e-6, max_iter=100):
     """
     Bisection Method to find a zero of f(x) in the interval [a, b].

     Parameters:
     f -- Continuous function f(x)
     a, b -- Initial interval [a, b] such that f(a) * f(b) < 0
     tol -- Stopping criterion (maximum allowed error)
     max_iter -- Maximum number of iterations

     Returns:
     Approximate root of f(x)
     """

     if f(a) * f(b) >= 0:
       raise ValueError("Bolzano's Theorem is not guaranteed: f(a) and f(b) must have opposite signs.")

     for _ in range(max_iter):
       c = (a + b) / 2 # Midpoint
       if abs(f(c)) < tol or (b - a) / 2 < tol:
         return c

       elif f(a) * f(c) < 0:
         b = c # Root is in [a, c]
       else:
         a = c # Root is in [c, b]

     return (a + b) / 2 # Final approximation of the root

# Example of use
f = lambda x: x**3 + x**2 - 10

root = bisection(f, 0, 2, 1e-8)

print(f"Approximate root: {root:.10f}")