"""
Trapezoidal Rule for Numerical Integration

The trapezoidal rule approximates the definite integral of a function by 
dividing the integration interval into subintervals and approximating the 
area under the curve using trapezoids.

Basic Trapezoidal Rule:
∫[a to b] f(x) dx ≈ (b-a)/2 * [f(a) + f(b)]

Composite Trapezoidal Rule (n subintervals):
∫[a to b] f(x) dx ≈ h/2 * [f(x₀) + 2∑f(xᵢ) + f(xₙ)]
where h = (b-a)/n and xᵢ = a + i*h

Error: O(h²) for smooth functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Union
import math


def trapezoidal_rule_basic(func: Callable[[float], float], 
                          a: float, 
                          b: float) -> float:
    """
    Basic trapezoidal rule using only two points (single trapezoid).
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        
    Returns:
        Approximate value of the integral
    """
    return (b - a) * (func(a) + func(b)) / 2


def trapezoidal_rule(func: Callable[[float], float], 
                    a: float, 
                    b: float, 
                    n: int) -> float:
    """
    Composite trapezoidal rule with n subintervals.
    
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
    
    # Calculate sum: f(a) + 2∑f(xᵢ) + f(b)
    total = func(a) + func(b)  # End points
    
    # Add interior points with coefficient 2
    for i in range(1, n):
        x_i = a + i * h
        total += 2 * func(x_i)
    
    return h * total / 2


def trapezoidal_rule_data(x_data: List[float], 
                         y_data: List[float]) -> float:
    """
    Trapezoidal rule for discrete data points.
    
    Args:
        x_data: List of x-coordinates (must be sorted)
        y_data: List of y-coordinates corresponding to x_data
        
    Returns:
        Approximate value of the integral
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if len(x_data) < 2:
        raise ValueError("At least 2 data points are required")
    
    # Check if x_data is sorted
    if not all(x_data[i] <= x_data[i+1] for i in range(len(x_data)-1)):
        raise ValueError("x_data must be sorted in ascending order")
    
    integral = 0.0
    
    # Apply trapezoidal rule to each interval
    for i in range(len(x_data) - 1):
        h = x_data[i + 1] - x_data[i]
        integral += h * (y_data[i] + y_data[i + 1]) / 2
    
    return integral


def adaptive_trapezoidal(func: Callable[[float], float], 
                        a: float, 
                        b: float, 
                        tolerance: float = 1e-6,
                        max_iterations: int = 20) -> Tuple[float, int]:
    """
    Adaptive trapezoidal rule that automatically adjusts the number of subintervals
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
    old_integral = trapezoidal_rule_basic(func, a, b)
    
    for iteration in range(max_iterations):
        n *= 2
        new_integral = trapezoidal_rule(func, a, b, n)
        
        # Check convergence using Richardson extrapolation estimate
        error_estimate = abs(new_integral - old_integral) / 3  # Error estimate for trapezoidal rule
        
        if error_estimate < tolerance:
            return new_integral, n
        
        old_integral = new_integral
    
    print(f"Warning: Maximum iterations ({max_iterations}) reached. "
          f"Error estimate: {error_estimate:.2e}")
    return new_integral, n


def richardson_extrapolation_trapezoidal(func: Callable[[float], float], 
                                        a: float, 
                                        b: float, 
                                        n: int) -> float:
    """
    Richardson extrapolation to improve trapezoidal rule accuracy.
    Combines T(h) and T(h/2) to get a more accurate result.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals for the coarser grid
        
    Returns:
        Improved approximation using Richardson extrapolation
    """
    T_h = trapezoidal_rule(func, a, b, n)      # Coarse grid
    T_h2 = trapezoidal_rule(func, a, b, 2*n)   # Fine grid (h/2)
    
    # Richardson extrapolation: R = T(h/2) + [T(h/2) - T(h)]/3
    return T_h2 + (T_h2 - T_h) / 3


def trapezoidal_error_estimate(func: Callable[[float], float], 
                              a: float, 
                              b: float, 
                              n: int,
                              second_derivative_max: float = None) -> float:
    """
    Estimate the error in trapezoidal rule integration.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        second_derivative_max: Maximum value of |f''(x)| on [a,b]. 
                              If None, estimated numerically.
        
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
    
    # Error bound: |E| ≤ (b-a)h²/12 * max|f''(x)|
    error_bound = (b - a) * h**2 * second_derivative_max / 12
    return error_bound


def plot_trapezoidal_approximation(func: Callable[[float], float], 
                                  a: float, 
                                  b: float, 
                                  n: int,
                                  title: str = "Trapezoidal Rule Approximation") -> None:
    """
    Visualize the trapezoidal rule approximation.
    
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
    
    # Generate points for trapezoids
    h = (b - a) / n
    x_trap = [a + i * h for i in range(n + 1)]
    y_trap = [func(x) for x in x_trap]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'f(x)')
    
    # Plot trapezoids
    for i in range(n):
        x_left = x_trap[i]
        x_right = x_trap[i + 1]
        y_left = y_trap[i]
        y_right = y_trap[i + 1]
        
        # Trapezoid vertices
        trap_x = [x_left, x_right, x_right, x_left, x_left]
        trap_y = [0, 0, y_right, y_left, 0]
        
        plt.fill(trap_x, trap_y, alpha=0.3, color='red', edgecolor='red', linewidth=1)
    
    # Plot the points
    plt.plot(x_trap, y_trap, 'ro', markersize=6, label='Sample Points')
    
    # Calculate and display the integral
    integral_approx = trapezoidal_rule(func, a, b, n)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\nn = {n}, Approximate Integral = {integral_approx:.6f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def convergence_analysis(func: Callable[[float], float], 
                        a: float, 
                        b: float, 
                        exact_value: float = None,
                        max_n: int = 1024) -> None:
    """
    Analyze the convergence of the trapezoidal rule as n increases.
    
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
    
    print("Convergence Analysis for Trapezoidal Rule")
    print("=" * 60)
    print(f"{'n':<8} {'h':<12} {'Approximation':<15} {'Error':<12} {'Error/h²':<12}")
    print("-" * 60)
    
    for n in n_values:
        h = (b - a) / n
        approx = trapezoidal_rule(func, a, b, n)
        
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
        plt.loglog(h_values, errors, 'bo-', label='Actual Error')
        plt.loglog(h_values, [h**2 * errors[0] / h_values[0]**2 for h in h_values], 
                  'r--', label='O(h²) Reference')
        plt.xlabel('h (step size)')
        plt.ylabel('Error')
        plt.title('Error vs Step Size')
        plt.grid(True)
        plt.legend()
        
        # Approximation convergence
        plt.subplot(1, 2, 2)
        plt.semilogx(n_values, approximations, 'go-', label='Trapezoidal Rule')
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
    Demonstrate the trapezoidal rule with various examples.
    """
    print("=== Trapezoidal Rule for Numerical Integration ===\n")
    
    # Example 1: Simple polynomial
    print("Example 1: ∫₀² x² dx = 8/3 ≈ 2.666667")
    
    def f1(x):
        return x**2
    
    exact1 = 8/3
    a1, b1 = 0, 2
    
    # Test with different numbers of subintervals
    for n in [1, 2, 4, 8, 16]:
        approx = trapezoidal_rule(f1, a1, b1, n)
        error = abs(approx - exact1)
        print(f"n = {n:2d}: Approximation = {approx:.8f}, Error = {error:.2e}")
    
    # Example 2: Trigonometric function
    print(f"\nExample 2: ∫₀^π sin(x) dx = 2")
    
    def f2(x):
        return math.sin(x)
    
    exact2 = 2.0
    a2, b2 = 0, math.pi
    
    approx_basic = trapezoidal_rule_basic(f2, a2, b2)
    approx_composite = trapezoidal_rule(f2, a2, b2, 10)
    approx_adaptive, n_adaptive = adaptive_trapezoidal(f2, a2, b2, tolerance=1e-6)
    approx_richardson = richardson_extrapolation_trapezoidal(f2, a2, b2, 8)
    
    print(f"Basic rule:      {approx_basic:.8f}, Error = {abs(approx_basic - exact2):.2e}")
    print(f"Composite (n=10): {approx_composite:.8f}, Error = {abs(approx_composite - exact2):.2e}")
    print(f"Adaptive:        {approx_adaptive:.8f}, Error = {abs(approx_adaptive - exact2):.2e}, n = {n_adaptive}")
    print(f"Richardson:      {approx_richardson:.8f}, Error = {abs(approx_richardson - exact2):.2e}")
    
    # Example 3: Data points
    print(f"\nExample 3: Integration from discrete data points")
    
    # Generate some data points from f(x) = e^x on [0, 1]
    x_data = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_data = [math.exp(x) for x in x_data]
    exact3 = math.exp(1) - 1  # ∫₀¹ e^x dx = e - 1
    
    approx_data = trapezoidal_rule_data(x_data, y_data)
    error_data = abs(approx_data - exact3)
    
    print(f"Data points: {list(zip(x_data, y_data))}")
    print(f"∫₀¹ e^x dx ≈ {approx_data:.8f}")
    print(f"Exact value = {exact3:.8f}")
    print(f"Error = {error_data:.2e}")
    
    # Example 4: Error estimation
    print(f"\nExample 4: Error estimation")
    
    def f4(x):
        return x**3
    
    a4, b4, n4 = 0, 1, 4
    approx4 = trapezoidal_rule(f4, a4, b4, n4)
    exact4 = 0.25  # ∫₀¹ x³ dx = 1/4
    actual_error = abs(approx4 - exact4)
    
    # For f(x) = x³, f''(x) = 6x, so max|f''(x)| on [0,1] is 6
    estimated_error = trapezoidal_error_estimate(f4, a4, b4, n4, second_derivative_max=6)
    
    print(f"∫₀¹ x³ dx with n = {n4}")
    print(f"Approximation: {approx4:.8f}")
    print(f"Exact value:   {exact4:.8f}")
    print(f"Actual error:  {actual_error:.2e}")
    print(f"Error bound:   {estimated_error:.2e}")
    
    # Example 5: Convergence analysis
    print(f"\nExample 5: Convergence analysis for ∫₀¹ e^x dx")
    
    def f5(x):
        return math.exp(x)
    
    exact5 = math.exp(1) - 1
    convergence_analysis(f5, 0, 1, exact5, max_n=512)


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1: x² function
    def f1(x):
        return x**2
    plot_trapezoidal_approximation(f1, 0, 2, 8, "Trapezoidal Rule: ∫₀² x² dx")
    
    # Plot Example 2: sin(x) function
    import math
    def f2(x):
        return math.sin(x)
    plot_trapezoidal_approximation(f2, 0, math.pi, 6, "Trapezoidal Rule: ∫₀^π sin(x) dx")