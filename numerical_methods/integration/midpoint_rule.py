"""
Midpoint Rule for Numerical Integration

The midpoint rule approximates the definite integral of a function by
dividing the integration interval into subintervals and using the value
of the function at the midpoint of each subinterval.

Basic Midpoint Rule:
∫[a to b] f(x) dx ≈ (b-a) * f((a+b)/2)

Composite Midpoint Rule (n subintervals):
∫[a to b] f(x) dx ≈ h * ∑f(xᵢ₊₁/₂)
where h = (b-a)/n and xᵢ₊₁/₂ = a + (i+0.5)*h

Error: O(h²) for smooth functions
The midpoint rule has the same order of accuracy as the trapezoidal rule
but often gives better results for smooth functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Union
import math


def midpoint_rule_basic(func: Callable[[float], float],
                       a: float,
                       b: float) -> float:
    """
    Basic midpoint rule using the midpoint of the interval.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        
    Returns:
        Approximate value of the integral
    """
    midpoint = (a + b) / 2
    return (b - a) * func(midpoint)


def midpoint_rule(func: Callable[[float], float],
                 a: float,
                 b: float,
                 n: int) -> float:
    """
    Composite midpoint rule with n subintervals.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        
    Returns:
        Approximate value of the integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    
    h = (b - a) / n
    total = 0.0
    
    # Evaluate function at midpoint of each subinterval
    for i in range(n):
        midpoint = a + (i + 0.5) * h
        total += func(midpoint)
    
    return h * total


def midpoint_rule_vectorized(func: Callable[[np.ndarray], np.ndarray],
                           a: float,
                           b: float,
                           n: int) -> float:
    """
    Vectorized midpoint rule for better performance with large n.
    
    Args:
        func: Vectorized function to integrate (accepts numpy arrays)
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        
    Returns:
        Approximate value of the integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    
    h = (b - a) / n
    
    # Generate all midpoints at once
    i = np.arange(n)
    midpoints = a + (i + 0.5) * h
    
    # Evaluate function at all midpoints
    values = func(midpoints)
    
    return h * np.sum(values)


def adaptive_midpoint(func: Callable[[float], float],
                     a: float,
                     b: float,
                     tolerance: float = 1e-6,
                     max_iterations: int = 20) -> Tuple[float, int]:
    """
    Adaptive midpoint rule that automatically adjusts the number of subintervals
    until the desired tolerance is achieved.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        tolerance: Desired accuracy
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Tuple of (integral_value, number_of_subintervals_used)
    """
    n = 1
    old_integral = midpoint_rule_basic(func, a, b)
    
    for iteration in range(max_iterations):
        n *= 2
        new_integral = midpoint_rule(func, a, b, n)
        
        # Check convergence
        error_estimate = abs(new_integral - old_integral) / 3  # Error estimate for midpoint rule
        
        if error_estimate < tolerance:
            return new_integral, n
        
        old_integral = new_integral
    
    print(f"Warning: Maximum iterations ({max_iterations}) reached. "
          f"Error estimate: {error_estimate:.2e}")
    return new_integral, n


def richardson_extrapolation_midpoint(func: Callable[[float], float],
                                    a: float,
                                    b: float,
                                    n: int) -> float:
    """
    Richardson extrapolation to improve midpoint rule accuracy.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals for the coarser grid
        
    Returns:
        Improved approximation using Richardson extrapolation
    """
    M_h = midpoint_rule(func, a, b, n)      # Coarse grid
    M_h2 = midpoint_rule(func, a, b, 2*n)   # Fine grid (h/2)
    
    # Richardson extrapolation: R = M(h/2) + [M(h/2) - M(h)]/3
    return M_h2 + (M_h2 - M_h) / 3


def midpoint_error_estimate(func: Callable[[float], float],
                          a: float,
                          b: float,
                          n: int,
                          second_derivative_max: float = None) -> float:
    """
    Estimate the error in midpoint rule integration.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        second_derivative_max: Maximum value of |f''(x)| on [a,b]
        
    Returns:
        Error bound estimate
    """
    h = (b - a) / n
    
    if second_derivative_max is None:
        # Estimate second derivative maximum numerically
        x_points = np.linspace(a, b, 1000)
        h_small = (b - a) / 999
        
        second_derivatives = []
        for i in range(1, len(x_points) - 1):
            # Central difference approximation for second derivative
            f_pp = (func(x_points[i+1]) - 2*func(x_points[i]) + func(x_points[i-1])) / (h_small**2)
            second_derivatives.append(abs(f_pp))
        
        second_derivative_max = max(second_derivatives) if second_derivatives else 1.0
    
    # Error bound: |E| ≤ (b-a)h²/24 * max|f''(x)|
    error_bound = (b - a) * h**2 * second_derivative_max / 24
    return error_bound


def plot_midpoint_approximation(func: Callable[[float], float],
                               a: float,
                               b: float,
                               n: int,
                               title: str = "Midpoint Rule Approximation") -> None:
    """
    Visualize the midpoint rule approximation.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        title: Plot title
    """
    # Generate points for smooth curve
    x_smooth = np.linspace(a, b, 1000)
    y_smooth = [func(x) for x in x_smooth]
    
    # Generate points for rectangles
    h = (b - a) / n
    
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='f(x)')
    
    # Plot rectangles
    for i in range(n):
        x_left = a + i * h
        x_right = a + (i + 1) * h
        midpoint = a + (i + 0.5) * h
        height = func(midpoint)

        # Rectangle
        rect_x = [x_left, x_right, x_right, x_left, x_left]
        rect_y = [0, 0, height, height, 0]

        plt.fill(rect_x, rect_y, alpha=0.3, color='red', edgecolor='red', linewidth=1)

        # Mark the midpoint
        plt.plot(midpoint, height, 'ro', markersize=4)
    
    # Calculate and display the integral
    integral_approx = midpoint_rule(func, a, b, n)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\nn = {n}, Approximate Integral = {integral_approx:.6f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def convergence_analysis_midpoint(func: Callable[[float], float],
                                a: float,
                                b: float,
                                exact_value: float = None,
                                max_n: int = 1024) -> None:
    """
    Analyze the convergence of the midpoint rule as n increases.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        exact_value: Exact value of the integral (if known)
        max_n: Maximum number of subintervals to test
    """
    n_values = [2**i for i in range(1, int(math.log2(max_n)) + 1)]
    approximations = []
    errors = []
    h_values = []
    
    print("Convergence Analysis for Midpoint Rule")
    print("=" * 60)
    print(f"{'n':<8} {'h':<12} {'Approximation':<15} {'Error':<12} {'Error/h²':<12}")
    print("-" * 60)
    
    for n in n_values:
        h = (b - a) / n
        approx = midpoint_rule(func, a, b, n)
        
        approximations.append(approx)
        h_values.append(h)
        
        if exact_value is not None:
            error = abs(approx - exact_value)
            error_over_h2 = error / (h**2)
            errors.append(error)
            print(f"{n:<8} {h:<12.6f} {approx:<15.8f} {error:<12.2e} {error_over_h2:<12.2e}")
        else:
            print(f"{n:<8} {h:<12.6f} {approx:<15.8f} {'N/A':<12} {'N/A':<12}")
    
    # Plot convergence if exact value is known
    if exact_value is not None and len(errors) > 1:
        plt.figure(figsize=(12, 5))
        
        # Error vs h
        plt.subplot(1, 2, 1)
        plt.loglog(h_values, errors, 'go-', label='Actual Error')
        plt.loglog(h_values, [h**2 * errors[0] / h_values[0]**2 for h in h_values],
                  'r--', label='O(h²) Reference')
        plt.xlabel('h (step size)')
        plt.ylabel('Error')
        plt.title('Error vs Step Size (Midpoint Rule)')
        plt.grid(True)
        plt.legend()
        
        # Approximation convergence
        plt.subplot(1, 2, 2)
        plt.semilogx(n_values, approximations, 'go-', label='Midpoint Rule')
        plt.axhline(y=exact_value, color='r', linestyle='--', label=f'Exact = {exact_value:.8f}')
        plt.xlabel('Number of Subintervals (n)')
        plt.ylabel('Integral Approximation')
        plt.title('Convergence to Exact Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def example_usage():
    """
    Demonstrate the midpoint rule with various examples.
    """
    print("=== Midpoint Rule for Numerical Integration ===\n")
    
    # Example 1: Simple polynomial
    print("Example 1: ∫₀² x² dx = 8/3 ≈ 2.666667")
    
    def f1(x):
        return x**2
    
    exact1 = 8/3
    a1, b1 = 0, 2
    
    # Test with different numbers of subintervals
    for n in [1, 2, 4, 8, 16]:
        approx = midpoint_rule(f1, a1, b1, n)
        error = abs(approx - exact1)
        print(f"n = {n:2d}: Approximation = {approx:.8f}, Error = {error:.2e}")
    
    # Example 2: Trigonometric function
    print(f"\nExample 2: ∫₀^π sin(x) dx = 2")
    
    def f2(x):
        return math.sin(x)
    
    exact2 = 2.0
    a2, b2 = 0, math.pi
    
    approx_basic = midpoint_rule_basic(f2, a2, b2)
    approx_composite = midpoint_rule(f2, a2, b2, 10)
    approx_adaptive, n_adaptive = adaptive_midpoint(f2, a2, b2, tolerance=1e-6)
    approx_richardson = richardson_extrapolation_midpoint(f2, a2, b2, 8)
    
    print(f"Basic rule:      {approx_basic:.8f}, Error = {abs(approx_basic - exact2):.2e}")
    print(f"Composite (n=10): {approx_composite:.8f}, Error = {abs(approx_composite - exact2):.2e}")
    print(f"Adaptive:        {approx_adaptive:.8f}, Error = {abs(approx_adaptive - exact2):.2e}, n = {n_adaptive}")
    print(f"Richardson:      {approx_richardson:.8f}, Error = {abs(approx_richardson - exact2):.2e}")
    
    # Example 3: Comparison with trapezoidal rule
    # Example 3: Vectorized computation
    print(f"\nExample 3: Vectorized computation performance test")
    
    def f3_scalar(x):
        return math.sin(x) * math.exp(-x)
    
    def f3_vector(x):
        return np.sin(x) * np.exp(-x)
    
    n_large = 10000
    
    # Time comparison would go here in a real implementation
    result_scalar = midpoint_rule(f3_scalar, 0, 5, n_large)
    result_vector = midpoint_rule_vectorized(f3_vector, 0, 5, n_large)
    
    print(f"Scalar version:    {result_scalar:.8f}")
    print(f"Vectorized version: {result_vector:.8f}")
    print(f"Difference:        {abs(result_scalar - result_vector):.2e}")
    
    # Example 4: Error estimation
    print(f"\nExample 4: Error estimation")
    
    def f4(x):
        return x**3
    
    a4, b4, n4 = 0, 1, 4
    approx4 = midpoint_rule(f4, a4, b4, n4)
    exact4 = 0.25  # ∫₀¹ x³ dx = 1/4
    actual_error = abs(approx4 - exact4)
    
    # For f(x) = x³, f''(x) = 6x, so max|f''(x)| on [0,1] is 6
    estimated_error = midpoint_error_estimate(f4, a4, b4, n4, second_derivative_max=6)
    
    print(f"∫₀¹ x³ dx with n = {n4}")
    print(f"Approximation: {approx4:.8f}")
    print(f"Exact value:   {exact4:.8f}")
    print(f"Actual error:  {actual_error:.2e}")
    print(f"Error bound:   {estimated_error:.2e}")


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1: x² function
    def f1(x):
        return x**2
    plot_midpoint_approximation(f1, 0, 2, 8, "Midpoint Rule: ∫₀² x² dx")
    
    # Convergence analysis
    def f2(x):
        return math.exp(x)
    exact2 = math.exp(1) - 1
    convergence_analysis_midpoint(f2, 0, 1, exact2, max_n=256)