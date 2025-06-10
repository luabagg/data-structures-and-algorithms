import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Function f(x) = x^3 - 2x + 1."""
    return x**3 - 2*x + 1

def g(x):
    """Function g(x) = x^2 - 1."""
    return x**2 - 1

def plot_graphical_method(f, g, x_range=(-2, 2), num_points=1000):
    """
    Plot f(x), g(x), and h(x) = f(x) - g(x) over a specified range and mark the zeros of h(x).

    Parameters:
    f, g      -- Functions to plot
    x_range   -- Tuple (min, max) for x-axis
    num_points-- Number of points in the plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    h = f(x) - g(x)

    plt.plot(x, f(x), label='f(x) = x^3 - 2x + 1')
    plt.plot(x, g(x), label='g(x) = x^2 - 1')
    plt.plot(x, h, label='h(x) = f(x) - g(x)')
    plt.axhline(0, color='gray', linestyle='--')

    # Find approximate zeros of h(x) using sign changes
    zero_indices = np.where(np.diff(np.sign(h)))[0]
    zeros = []
    for idx in zero_indices:
        # Linear interpolation for a better zero estimate
        x0, x1 = x[idx], x[idx+1]
        y0, y1 = h[idx], h[idx+1]
        zero = x0 - y0 * (x1 - x0) / (y1 - y0)
        zeros.append(zero)
        plt.plot([zero], [0], 'ro')  # Mark the zero

    plt.legend()
    plt.grid(True)
    plt.title('Graphical Method: f(x), g(x), and h(x) = f(x) - g(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    print("The approximate zeros of h(x) are:", [round(z, 4) for z in zeros])

if __name__ == "__main__":
    plot_graphical_method(f, g)