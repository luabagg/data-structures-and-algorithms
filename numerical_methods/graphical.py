import numpy as np
import matplotlib.pyplot as plt

# Defining the functions f(x) and g(x)
def f(x):
    return x**3 - 2*x + 1

def g(x):
    return x**2 - 1

# Calculating values of h(x) = f(x) - g(x)
x = np.linspace(-2, 2, 1000)
h = f(x) - g(x)

# Plotting the graph
plt.plot(x, f(x), label='f(x) = x^3 - 2x + 1')
plt.plot(x, g(x), label='g(x) = x^2 - 1')
plt.plot(x, h, label='h(x) = f(x) - g(x)')
plt.axhline(0, color='gray', linestyle='--')

# Finding the zeros of h(x)
zeros = np.roots(np.poly1d(h))

# Plotting the roots
for zero in zeros:
    if np.isreal(zero):
        plt.plot([zero], [0], 'ro')  # Red points indicate the zeros

plt.legend()
plt.grid(True)
plt.title('Graph of f(x) = x^3 - 2x + 1 and g(x) = x^2 - 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("The zeros of function h(x) are:", [round(zero, 4) for zero in zeros])