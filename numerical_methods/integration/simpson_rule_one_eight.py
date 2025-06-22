"""
Simpson's 3/8 Rule for Numerical Integration

Simpson's 3/8 rule approximates the definite integral by fitting cubic polynomials
through consecutive sets of four points and integrating the cubics.
It has the same order of accuracy as Simpson's 1/3 rule (O(h⁴)) but uses
different coefficients.

Basic Simpson's 3/8 Rule (3 intervals):
∫[a to b] f(x) dx ≈ 3h/8 * [f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]
where h = (b-a)/3

Composite Simpson's 3/8 Rule (n intervals, n must be multiple of 3):
∫[a to b] f(x) dx ≈ 3h/8 * [f(x₀) + 3∑f(x₃ᵢ₊₁) + 3∑f(x₃ᵢ₊₂) + 2∑f(x₃ᵢ) + f(xₙ)]
where h = (b-a)/n

Error: O(h⁴) for smooth functions - same as Simpson's 1/3 rule
Sometimes more accurate than 1/3 rule for certain functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Union
import math


def simpson_38_rule_basic(func: Callable[[float], float], 
                         a: float, 
                         b: float) -> float:
    """
    Basic Simpson's 3/8 rule using four points (three intervals).
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        
    Returns:
        Approximate value of the integral
    """
    h = (b - a) / 3
    x1 = a + h
    x2 = a + 2 * h
    
    return 3 * h / 8 * (func(a) + 3 * func(x1) + 3 * func(x2) + func(b))


def simpson_38_rule(func: Callable[[float], float], 
                   a: float, 
                   b: float, 
                   n: int) -> float:
    """
    Composite Simpson's 3/8 rule with n subintervals (n must be multiple of 3).
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals (must be multiple of 3)
        
    Returns:
        Approximate value of the integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    
    if n % 3 != 0:
        raise ValueError("Number of subintervals must be multiple of 3 for Simpson's 3/8 rule")
    
    h = (b - a) / n
    
    # Calculate the sum according to Simpson's 3/8 formula
    total = func(a) + func(b)  # End points
    
    # Add coefficients for interior points
    for i in range(1, n):
        x_i = a + i * h
        if i % 3 == 0:  # Points at positions 3, 6, 9, ... get coefficient 2
            total += 2 * func(x_i)
        else:  # Points at positions 1, 2, 4, 5, 7, 8, ... get coefficient 3
            total += 3 * func(x_i)
    
    return 3 * h * total / 8


def simpson_38_rule_data(x_data: List[float], 
                        y_data: List[float]) -> float:
    """
    Simpson's 3/8 rule for discrete data points.
    
    Args:
        x_data: List of x-coordinates (must be sorted, number of points = 3k+1)
        y_data: List of y-coordinates corresponding to x_data
        
    Returns:
        Approximate value of the integral
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if len(x_data) < 4:
        raise ValueError("At least 4 data points are required")
    
    if (len(x_data) - 1) % 3 != 0:
        raise ValueError("Number of intervals (points - 1) must be multiple of 3 for Simpson's 3/8 rule")
    
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
            raise ValueError("Data points must be equally spaced for Simpson's 3/8 rule")
    
    # Apply Simpson's 3/8 rule
    h = h_expected
    total = y_data[0] + y_data[-1]  # End points
    
    for i in range(1, len(y_data) - 1):
        if i % 3 == 0:  # Points at positions 3, 6, 9, ... get coefficient 2
            total += 2 * y_data[i]
        else:  # Points at positions 1, 2, 4, 5, 7, 8, ... get coefficient 3
            total += 3 * y_data[i]
    
    return 3 * h * total / 8


def adaptive_simpson_38(func: Callable[[float], float], 
                       a: float, 
                       b: float, 
                       tolerance: float = 1e-6,
                       max_iterations: int = 20) -> Tuple[float, int]:
    """
    Adaptive Simpson's 3/8 rule that automatically adjusts the number of subintervals.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        tolerance: Desired accuracy
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Tuple of (integral_value, number_of_subintervals_used)
    """
    n = 3  # Start with 3 intervals (minimum for Simpson's 3/8)
    old_integral = simpson_38_rule_basic(func, a, b)
    
    for iteration in range(max_iterations):
        n *= 3  # Increase by factor of 3 to maintain divisibility
        new_integral = simpson_38_rule(func, a, b, n)
        
        # Check convergence using Richardson extrapolation estimate
        error_estimate = abs(new_integral - old_integral) / 15  # Error estimate for Simpson's rule
        
        if error_estimate < tolerance:
            return new_integral, n
        
        old_integral = new_integral
    
    print(f"Warning: Maximum iterations ({max_iterations}) reached. "
          f"Error estimate: {error_estimate:.2e}")
    return new_integral, n


def richardson_extrapolation_simpson_38(func: Callable[[float], float], 
                                       a: float, 
                                       b: float, 
                                       n: int) -> float:
    """
    Richardson extrapolation to improve Simpson's 3/8 rule accuracy.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals for the coarser grid (must be multiple of 3)
        
    Returns:
        Improved approximation using Richardson extrapolation
    """
    if n % 3 != 0:
        raise ValueError("n must be multiple of 3 for Simpson's 3/8 rule")
    
    S_h = simpson_38_rule(func, a, b, n)      # Coarse grid
    S_h2 = simpson_38_rule(func, a, b, 2*n)   # Fine grid (h/2)
    
    # Richardson extrapolation: R = S(h/2) + [S(h/2) - S(h)]/15
    return S_h2 + (S_h2 - S_h) / 15


def simpson_38_error_estimate(func: Callable[[float], float], 
                             a: float, 
                             b: float, 
                             n: int,
                             fourth_derivative_max: float = None) -> float:
    """
    Estimate the error in Simpson's 3/8 rule integration.
    
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
    
    # Error bound: |E| ≤ (b-a)h⁴/80 * max|f⁽⁴⁾(x)|
    # Note: Different constant than Simpson's 1/3 rule (80 vs 180)
    error_bound = (b - a) * h**4 * fourth_derivative_max / 80
    return error_bound


def plot_simpson_38_approximation(func: Callable[[float], float], 
                                 a: float, 
                                 b: float, 
                                 n: int,
                                 title: str = "Simpson's 3/8 Rule Approximation") -> None:
    """
    Visualize Simpson's 3/8 rule approximation with cubic segments.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        n: Number of subintervals (must be multiple of 3)
        title: Plot title
    """
    if n % 3 != 0:
        raise ValueError("Number of subintervals must be multiple of 3")
    
    # Generate points for smooth curve
    x_smooth = np.linspace(a, b, 1000)
    y_smooth = [func(x) for x in x_smooth]
    
    # Generate points for Simpson's 3/8 rule
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [func(x) for x in x_points]
    
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='f(x)')
    
    # Plot cubic segments
    for i in range(0, n, 3):  # Process every three intervals
        if i + 3 <= n:
            x_seg = x_points[i:i+4]  # 4 points for cubic
            y_seg = y_points[i:i+4]
            
            # Create cubic polynomial through four points using Lagrange interpolation
            x_cubic = np.linspace(x_seg[0], x_seg[3], 100)
            
            cubic = []
            for x in x_cubic:
                # Lagrange interpolation for cubic
                y = 0
                for j in range(4):
                    L_j = 1
                    for k in range(4):
                        if k != j:
                            L_j *= (x - x_seg[k]) / (x_seg[j] - x_seg[k])
                    y += L_j * y_seg[j]
                cubic.append(y)
            
            plt.plot(x_cubic, cubic, 'r--', linewidth=1.5, alpha=0.7)
            
            # Fill area under cubic
            plt.fill_between(x_cubic, 0, cubic, alpha=0.2, color='red')
    
    # Plot the sample points
    plt.plot(x_points, y_points, 'ro', markersize=6, label='Sample Points')
    
    # Calculate and display the integral
    integral_approx = simpson_38_rule(func, a, b, n)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\nn = {n}, Approximate Integral = {integral_approx:.6f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def convergence_analysis_simpson_38(func: Callable[[float], float], 
                                   a: float, 
                                   b: float, 
                                   exact_value: float = None,
                                   max_n: int = 243) -> None:
    """
    Analyze the convergence of Simpson's 3/8 rule as n increases.
    
    Args:
        func: Function to integrate
        a: Lower limit of integration
        b: Upper limit of integration
        exact_value: Exact value of the integral (if known)
        max_n: Maximum number of subintervals to test
    """
    # Only use multiples of 3
    n_values = [3**i for i in range(1, int(math.log(max_n, 3)) + 1) if 3**i <= max_n]
    approximations = []
    errors = []
    h_values = []
    
    print("Convergence Analysis for Simpson's 3/8 Rule")
    print("=" * 65)
    print(f"{'n':<8} {'h':<12} {'Approximation':<15} {'Error':<12} {'Error/h⁴':<12}")
    print("-" * 65)
    
    for n in n_values:
        h = (b - a) / n
        approx = simpson_38_rule(func, a, b, n)
        
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
        plt.loglog(h_values, errors, 'co-', label='Actual Error')
        plt.loglog(h_values, [h**4 * errors[0] / h_values[0]**4 for h in h_values], 
                  'r--', label='O(h⁴) Reference')
        plt.xlabel('h (step size)')
        plt.ylabel('Error')
        plt.title('Error vs Step Size (Simpson 3/8)')
        plt.grid(True)
        plt.legend()
        
        # Approximation convergence
        plt.subplot(1, 2, 2)
        plt.semilogx(n_values, approximations, 'co-', label="Simpson's 3/8 Rule")
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
    Demonstrate Simpson's 3/8 rule with various examples.
    """
    print("=== Simpson's 3/8 Rule for Numerical Integration ===\n")
    
    # Example 1: Polynomial that Simpson's rule integrates exactly
    print("Example 1: ∫₀³ x³ dx = 81/4 = 20.25 (Simpson's rule should be exact)")
    
    def f1(x):
        return x**3
    
    exact1 = 81/4
    a1, b1 = 0, 3
    
    # Test with different numbers of subintervals
    for n in [3, 6, 9, 12]:
        approx = simpson_38_rule(f1, a1, b1, n)
        error = abs(approx - exact1)
        print(f"n = {n:2d}: Approximation = {approx:.10f}, Error = {error:.2e}")
    
    # Example 2: Trigonometric function
    print(f"\nExample 2: ∫₀^π sin(x) dx = 2")
    
    def f2(x):
        return math.sin(x)
    
    exact2 = 2.0
    a2, b2 = 0, math.pi
    
    # Adjust interval to be multiple of 3 for exact π
    # Use scaled version: ∫₀³ sin(πx/3) dx = 6/π
    def f2_scaled(x):
        return math.sin(math.pi * x / 3)
    
    exact2_scaled = 6 / math.pi
    
    approx_basic = simpson_38_rule_basic(f2_scaled, 0, 3)
    approx_composite = simpson_38_rule(f2_scaled, 0, 3, 12)
    approx_adaptive, n_adaptive = adaptive_simpson_38(f2_scaled, 0, 3, tolerance=1e-8)
    approx_richardson = richardson_extrapolation_simpson_38(f2_scaled, 0, 3, 9)
    
    print(f"Scaled function: ∫₀³ sin(πx/3) dx = 6/π ≈ {exact2_scaled:.10f}")
    print(f"Basic rule:      {approx_basic:.10f}, Error = {abs(approx_basic - exact2_scaled):.2e}")
    print(f"Composite (n=12): {approx_composite:.10f}, Error = {abs(approx_composite - exact2_scaled):.2e}")
    print(f"Adaptive:        {approx_adaptive:.10f}, Error = {abs(approx_adaptive - exact2_scaled):.2e}, n = {n_adaptive}")
    print(f"Richardson:      {approx_richardson:.10f}, Error = {abs(approx_richardson - exact2_scaled):.2e}")
    
    # Example 3: Data points
    print(f"\nExample 3: Integration from discrete data points")
    
    # Generate equally spaced data points from f(x) = x² on [0, 3]
    n_data = 9  # This gives 10 points (intervals = 9, multiple of 3)
    x_data = [i * 3.0 / n_data for i in range(n_data + 1)]
    y_data = [x**2 for x in x_data]
    exact3 = 9  # ∫₀³ x² dx = 9
    
    approx_data = simpson_38_rule_data(x_data, y_data)
    error_data = abs(approx_data - exact3)
    
    print(f"Data points: {list(zip([f'{x:.2f}' for x in x_data], [f'{y:.2f}' for y in y_data]))}")
    print(f"∫₀³ x² dx ≈ {approx_data:.8f}")
    print(f"Exact value = {exact3:.8f}")
    print(f"Error = {error_data:.2e}")
    
    # Example 4: Error estimation
    print(f"\nExample 4: Error estimation")
    
    def f4(x):
        return x**5
    
    a4, b4, n4 = 0, 1, 6
    approx4 = simpson_38_rule(f4, a4, b4, n4)
    exact4 = 1/6  # ∫₀¹ x⁵ dx = 1/6
    actual_error = abs(approx4 - exact4)
    
    # For f(x) = x⁵, f⁽⁴⁾(x) = 120x, so max|f⁽⁴⁾(x)| on [0,1] is 120
    estimated_error = simpson_38_error_estimate(f4, a4, b4, n4, fourth_derivative_max=120)
    
    print(f"∫₀¹ x⁵ dx with n = {n4}")
    print(f"Approximation: {approx4:.10f}")
    print(f"Exact value:   {exact4:.10f}")
    print(f"Actual error:  {actual_error:.2e}")
    print(f"Error bound:   {estimated_error:.2e}")
    
    # Example 5: Higher order polynomial
    print(f"\nExample 5: Higher order polynomial where 3/8 rule should be exact")
    
    def f5(x):
        return x**2 + 2*x + 1  # Quadratic polynomial
    
    a5, b5, n5 = 0, 3, 3
    approx5 = simpson_38_rule(f5, a5, b5, n5)
    # ∫₀³ (x² + 2x + 1) dx = [x³/3 + x² + x]₀³ = 9 + 9 + 3 = 21
    exact5 = 21
    error5 = abs(approx5 - exact5)
    
    print(f"∫₀³ (x² + 2x + 1) dx with n = {n5}")
    print(f"Approximation: {approx5:.10f}")
    print(f"Exact value:   {exact5:.10f}")
    print(f"Error:         {error5:.2e} (should be near machine precision)")


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1: x³ function
    def f1(x):
        return x**3
    plot_simpson_38_approximation(f1, 0, 3, 9, "Simpson's 3/8 Rule: ∫₀³ x³ dx")
    
    # Convergence analysis
    def f2(x):
        return math.exp(x)
    exact2 = math.exp(1) - 1
    convergence_analysis_simpson_38(f2, 0, 1, exact2, max_n=81)