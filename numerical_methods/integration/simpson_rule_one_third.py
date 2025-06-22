"""
Simpson's 1/3 Rule for Numerical Integration

Simpson's 1/3 rule approximates the definite integral by fitting parabolas
through consecutive sets of three points and integrating the parabolas.
It provides much higher accuracy than trapezoidal and midpoint rules.

Basic Simpson's 1/3 Rule (2 intervals):
∫[a to b] f(x) dx ≈ (b-a)/6 * [f(a) + 4f((a+b)/2) + f(b)]

Composite Simpson's 1/3 Rule (n intervals, n must be even):
∫[a to b] f(x) dx ≈ h/3 * [f(x₀) + 4∑f(x₂ᵢ₋₁) + 2∑f(x₂ᵢ) + f(xₙ)]
where h = (b-a)/n

Error: O(h⁴) for smooth functions - much better than O(h²) methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Union
import math


def simpson_13_rule_basic(func: Callable[[float], float], 
                         a: float, 
                         b: float) -> float:
    """
    Basic Simpson's 1/3 rule using three points (two intervals).
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        
    Returns:
        Approximate value of the integral
    """
    midpoint = (a + b) / 2
    return (b - a) / 6 * (func(a) + 4 * func(midpoint) + func(b))


def simpson_13_rule(func: Callable[[float], float], 
                   a: float, 
                   b: float, 
                   n: int) -> float:
    """
    Composite Simpson's 1/3 rule with n subintervals (n must be even).
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals (must be even)
        
    Returns:
        Approximate value of the integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even for Simpson's 1/3 rule")
    
    h = (b - a) / n
    
    # Calculate the sum according to Simpson's 1/3 formula
    total = func(a) + func(b)  # End points
    
    # Add coefficients for interior points
    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 1:  # Odd indices get coefficient 4
            total += 4 * func(x_i)
        else:  # Even indices get coefficient 2
            total += 2 * func(x_i)
    
    return h * total / 3


def simpson_13_rule_data(x_data: List[float], 
                        y_data: List[float]) -> float:
    """
    Simpson's 1/3 rule for discrete data points.
    
    Args:
        x_data: List of x-coordinates (must be sorted, odd number of points)
        y_data: List of y-coordinates corresponding to x_data
        
    Returns:
        Approximate value of the integral
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if len(x_data) < 3:
        raise ValueError("At least 3 data points are required")
    
    if len(x_data) % 2 == 0:
        raise ValueError("Odd number of data points required for Simpson's 1/3 rule")
    
    # Check if x_data is sorted
    if not all(x_data[i] <= x_data[i+1] for i in range(len(x_data)-1)):
        raise ValueError("x_data must be sorted in ascending order")
    
    # Check if points are equally spaced (within tolerance)
    n = len(x_data) - 1
    h_expected = (x_data[-1] - x_data[0]) / n
    tolerance = 1e-10
    
    for i in range(len(x_data) - 1):
        h_actual = x_data[i + 1] - x_data[i]
        if abs(h_actual - h_expected) > tolerance:
            raise ValueError("Data points must be equally spaced for Simpson's 1/3 rule")
    
    # Apply Simpson's 1/3 rule
    h = h_expected
    total = y_data[0] + y_data[-1]  # End points
    
    for i in range(1, len(y_data) - 1):
        if i % 2 == 1:  # Odd indices get coefficient 4
            total += 4 * y_data[i]
        else:  # Even indices get coefficient 2
            total += 2 * y_data[i]
    
    return h * total / 3


def adaptive_simpson_13(func: Callable[[float], float], 
                       a: float, 
                       b: float, 
                       tolerance: float = 1e-6,
                       max_iterations: int = 20) -> Tuple[float, int]:
    """
    Adaptive Simpson's 1/3 rule that automatically adjusts the number of subintervals.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        tolerance: Desired accuracy
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Tuple of (integral_value, number_of_subintervals_used)
    """
    n = 2  # Start with 2 intervals (minimum for Simpson's 1/3)
    old_integral = simpson_13_rule_basic(func, a, b)
    
    for iteration in range(max_iterations):
        n *= 2
        new_integral = simpson_13_rule(func, a, b, n)
        
        # Check convergence using Richardson extrapolation estimate
        error_estimate = abs(new_integral - old_integral) / 15  # Error estimate for Simpson's rule
        
        if error_estimate < tolerance:
            return new_integral, n
        
        old_integral = new_integral
    
    print(f"Warning: Maximum iterations ({max_iterations}) reached. "
          f"Error estimate: {error_estimate:.2e}")
    return new_integral, n


def richardson_extrapolation_simpson_13(func: Callable[[float], float], 
                                       a: float, 
                                       b: float, 
                                       n: int) -> float:
    """
    Richardson extrapolation to improve Simpson's 1/3 rule accuracy.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals for the coarser grid (must be even)
        
    Returns:
        Improved approximation using Richardson extrapolation
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's 1/3 rule")
    
    S_h = simpson_13_rule(func, a, b, n)      # Coarse grid
    S_h2 = simpson_13_rule(func, a, b, 2*n)   # Fine grid (h/2)
    
    # Richardson extrapolation: R = S(h/2) + [S(h/2) - S(h)]/15
    return S_h2 + (S_h2 - S_h) / 15


def simpson_13_error_estimate(func: Callable[[float], float], 
                             a: float, 
                             b: float, 
                             n: int,
                             fourth_derivative_max: float = None) -> float:
    """
    Estimate the error in Simpson's 1/3 rule integration.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals
        fourth_derivative_max: Maximum value of |f⁽⁴⁾(x)| on [a,b]
        
    Returns:
        Error bound estimate
    """
    h = (b - a) / n
    
    if fourth_derivative_max is None:
        # Estimate fourth derivative maximum numerically
        x_points = np.linspace(a, b, 1000)
        h_small = (b - a) / 999
        
        fourth_derivatives = []
        # Use finite differences to approximate fourth derivative
        for i in range(2, len(x_points) - 2):
            # Fourth derivative approximation using 5-point stencil
            f_4 = (func(x_points[i+2]) - 4*func(x_points[i+1]) + 6*func(x_points[i]) 
                   - 4*func(x_points[i-1]) + func(x_points[i-2])) / (h_small**4)
            fourth_derivatives.append(abs(f_4))
        
        fourth_derivative_max = max(fourth_derivatives) if fourth_derivatives else 1.0
    
    # Error bound: |E| ≤ (b-a)h⁴/180 * max|f⁽⁴⁾(x)|
    error_bound = (b - a) * h**4 * fourth_derivative_max / 180
    return error_bound


def plot_simpson_13_approximation(func: Callable[[float], float], 
                                 a: float, 
                                 b: float, 
                                 n: int,
                                 title: str = "Simpson's 1/3 Rule Approximation") -> None:
    """
    Visualize Simpson's 1/3 rule approximation with parabolic segments.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals (must be even)
        title: Plot title
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even")
    
    # Generate points for smooth curve
    x_smooth = np.linspace(a, b, 1000)
    y_smooth = [func(x) for x in x_smooth]
    
    # Generate points for Simpson's rule
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [func(x) for x in x_points]
    
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='f(x)')
    
    # Plot parabolic segments
    for i in range(0, n, 2):  # Process every two intervals
        x0, x1, x2 = x_points[i], x_points[i+1], x_points[i+2]
        y0, y1, y2 = y_points[i], y_points[i+1], y_points[i+2]
        
        # Create parabola through three points
        x_parabola = np.linspace(x0, x2, 100)
        
        # Lagrange interpolation for parabola
        parabola = []
        for x in x_parabola:
            # L0(x) * y0 + L1(x) * y1 + L2(x) * y2
            L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
            L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
            L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
            y = L0 * y0 + L1 * y1 + L2 * y2
            parabola.append(y)
        
        plt.plot(x_parabola, parabola, 'r--', linewidth=1.5, alpha=0.7)
        
        # Fill area under parabola
        plt.fill_between(x_parabola, 0, parabola, alpha=0.2, color='red')
    
    # Plot the sample points
    plt.plot(x_points, y_points, 'ro', markersize=6, label='Sample Points')
    
    # Calculate and display the integral
    integral_approx = simpson_13_rule(func, a, b, n)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\nn = {n}, Approximate Integral = {integral_approx:.6f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def convergence_analysis_simpson_13(func: Callable[[float], float], 
                                   a: float, 
                                   b: float, 
                                   exact_value: float = None,
                                   max_n: int = 256) -> None:
    """
    Analyze the convergence of Simpson's 1/3 rule as n increases.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        exact_value: Exact value of the integral (if known)
        max_n: Maximum number of subintervals to test
    """
    # Only use even values of n
    n_values = [2**i for i in range(1, int(math.log2(max_n)) + 1) if 2**i <= max_n]
    approximations = []
    errors = []
    h_values = []
    
    print("Convergence Analysis for Simpson's 1/3 Rule")
    print("=" * 65)
    print(f"{'n':<8} {'h':<12} {'Approximation':<15} {'Error':<12} {'Error/h⁴':<12}")
    print("-" * 65)
    
    for n in n_values:
        h = (b - a) / n
        approx = simpson_13_rule(func, a, b, n)
        
        approximations.append(approx)
        h_values.append(h)
        
        if exact_value is not None:
            error = abs(approx - exact_value)
            error_over_h4 = error / (h**4)
            errors.append(error)
            print(f"{n:<8} {h:<12.6f} {approx:<15.8f} {error:<12.2e} {error_over_h4:<12.2e}")
        else:
            print(f"{n:<8} {h:<12.6f} {approx:<15.8f} {'N/A':<12} {'N/A':<12}")
    
    # Plot convergence if exact value is known
    if exact_value is not None and len(errors) > 1:
        plt.figure(figsize=(12, 5))
        
        # Error vs h
        plt.subplot(1, 2, 1)
        plt.loglog(h_values, errors, 'mo-', label='Actual Error')
        plt.loglog(h_values, [h**4 * errors[0] / h_values[0]**4 for h in h_values], 
                  'r--', label='O(h⁴) Reference')
        plt.xlabel('h (step size)')
        plt.ylabel('Error')
        plt.title('Error vs Step Size (Simpson 1/3)')
        plt.grid(True)
        plt.legend()
        
        # Approximation convergence
        plt.subplot(1, 2, 2)
        plt.semilogx(n_values, approximations, 'mo-', label="Simpson's 1/3 Rule")
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
    Demonstrate Simpson's 1/3 rule with various examples.
    """
    print("=== Simpson's 1/3 Rule for Numerical Integration ===\n")
    
    # Example 1: Polynomial that Simpson's rule integrates exactly
    print("Example 1: ∫₀² x³ dx = 4 (Simpson's rule should be exact)")
    
    def f1(x):
        return x**3
    
    exact1 = 4.0
    a1, b1 = 0, 2
    
    # Test with different numbers of subintervals
    for n in [2, 4, 6, 8]:
        approx = simpson_13_rule(f1, a1, b1, n)
        error = abs(approx - exact1)
        print(f"n = {n:2d}: Approximation = {approx:.10f}, Error = {error:.2e}")
    
    # Example 2: Trigonometric function
    print(f"\nExample 2: ∫₀^π sin(x) dx = 2")
    
    def f2(x):
        return math.sin(x)
    
    exact2 = 2.0
    a2, b2 = 0, math.pi
    
    approx_basic = simpson_13_rule_basic(f2, a2, b2)
    approx_composite = simpson_13_rule(f2, a2, b2, 10)
    approx_adaptive, n_adaptive = adaptive_simpson_13(f2, a2, b2, tolerance=1e-8)
    approx_richardson = richardson_extrapolation_simpson_13(f2, a2, b2, 8)
    
    print(f"Basic rule:      {approx_basic:.10f}, Error = {abs(approx_basic - exact2):.2e}")
    print(f"Composite (n=10): {approx_composite:.10f}, Error = {abs(approx_composite - exact2):.2e}")
    print(f"Adaptive:        {approx_adaptive:.10f}, Error = {abs(approx_adaptive - exact2):.2e}, n = {n_adaptive}")
    print(f"Richardson:      {approx_richardson:.10f}, Error = {abs(approx_richardson - exact2):.2e}")
    
    # Example 3: Data points
    print(f"\nExample 3: Integration from discrete data points")
    
    # Generate equally spaced data points from f(x) = x² on [0, 2]
    n_data = 8  # This gives 9 points (odd number required)
    x_data = [i * 2.0 / n_data for i in range(n_data + 1)]
    y_data = [x**2 for x in x_data]
    exact3 = 8/3
    
    approx_data = simpson_13_rule_data(x_data, y_data)
    error_data = abs(approx_data - exact3)
    
    print(f"Data points: {list(zip([f'{x:.2f}' for x in x_data], [f'{y:.2f}' for y in y_data]))}")
    print(f"∫₀² x² dx ≈ {approx_data:.8f}")
    print(f"Exact value = {exact3:.8f}")
    print(f"Error = {error_data:.2e}")
    
    # Example 4: Comparison with other methods
    # Example 4: Error estimation
    print(f"\nExample 4: Error estimation")
    
    def f4(x):
        return x**4
    
    a4, b4, n4 = 0, 1, 4
    approx4 = simpson_13_rule(f4, a4, b4, n4)
    exact4 = 0.2  # ∫₀¹ x⁴ dx = 1/5
    actual_error = abs(approx4 - exact4)
    
    # For f(x) = x⁴, f⁽⁴⁾(x) = 24, so max|f⁽⁴⁾(x)| on [0,1] is 24
    estimated_error = simpson_13_error_estimate(f4, a4, b4, n4, fourth_derivative_max=24)
    
    print(f"∫₀¹ x⁴ dx with n = {n4}")
    print(f"Approximation: {approx4:.10f}")
    print(f"Exact value:   {exact4:.10f}")
    print(f"Actual error:  {actual_error:.2e}")
    print(f"Error bound:   {estimated_error:.2e}")


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1: x³ function
    def f1(x):
        return x**3
    plot_simpson_13_approximation(f1, 0, 2, 6, "Simpson's 1/3 Rule: ∫₀² x³ dx")
    
    # Convergence analysis
    def f2(x):
        return math.exp(x)
    exact2 = math.exp(1) - 1
    convergence_analysis_simpson_13(f2, 0, 1, exact2, max_n=128)