"""
Newton's Method for Polynomial Interpolation

Newton's interpolation method uses divided differences to construct an 
interpolating polynomial. The Newton interpolating polynomial is given by:

P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ... + 
       f[x₀,x₁,...,xₙ](x-x₀)(x-x₁)...(x-xₙ₋₁)

where f[x₀,x₁,...,xₖ] are the divided differences.

There are two main forms:
1. Newton's Forward Difference Formula (for equally spaced points)
2. Newton's Divided Difference Formula (for unequally spaced points)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def divided_differences(x_points: List[float], y_points: List[float]) -> List[List[float]]:
    """
    Calculate the divided difference table for Newton's interpolation.
    
    Args:
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        
    Returns:
        2D list representing the divided difference table
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    n = len(x_points)
    # Initialize the divided difference table
    dd_table = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill the first column with y values (0th order divided differences)
    for i in range(n):
        dd_table[i][0] = y_points[i]
    
    # Calculate higher order divided differences
    for j in range(1, n):  # column
        for i in range(n - j):  # row
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    return dd_table


def newton_interpolation(x: Union[float, np.ndarray], 
                        x_points: List[float], 
                        y_points: List[float]) -> Union[float, np.ndarray]:
    """
    Perform Newton interpolation to find P(x) for given data points.
    
    Args:
        x: Point(s) at which to evaluate the interpolating polynomial
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        
    Returns:
        Interpolated value(s) P(x)
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    # Calculate divided differences
    dd_table = divided_differences(x_points, y_points)
    n = len(x_points)
    
    # Check if x is a scalar or array
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array([x])
    else:
        x = np.array(x)
    
    result = np.full_like(x, dd_table[0][0], dtype=float)  # Start with f[x₀]
    
    # Add each term: f[x₀,x₁,...,xₖ] * ∏(x - xᵢ)
    for i in range(1, n):
        term = dd_table[0][i]  # f[x₀,x₁,...,xᵢ]
        
        # Multiply by ∏(x - xⱼ) for j = 0 to i-1
        for j in range(i):
            term *= (x - x_points[j])
        
        result += term
    
    return result[0] if is_scalar else result


def newton_forward_difference(x: Union[float, np.ndarray], 
                            x_points: List[float], 
                            y_points: List[float], 
                            h: float = None) -> Union[float, np.ndarray]:
    """
    Newton's forward difference formula for equally spaced points.
    
    Args:
        x: Point(s) at which to evaluate the interpolating polynomial
        x_points: List of x-coordinates of equally spaced data points
        y_points: List of y-coordinates of data points
        h: Step size (spacing between x points). If None, calculated automatically
        
    Returns:
        Interpolated value(s) P(x)
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    n = len(x_points)
    
    # Calculate step size if not provided
    if h is None:
        h = x_points[1] - x_points[0]
        # Verify equally spaced points
        for i in range(1, n - 1):
            if abs((x_points[i + 1] - x_points[i]) - h) > 1e-10:
                raise ValueError("Points are not equally spaced. Use newton_interpolation instead.")
    
    # Create forward difference table
    diff_table = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill first column with y values
    for i in range(n):
        diff_table[i][0] = y_points[i]
    
    # Calculate forward differences
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    
    # Check if x is a scalar or array
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array([x])
    else:
        x = np.array(x)
    
    # Calculate u = (x - x₀) / h
    u = (x - x_points[0]) / h
    
    result = np.full_like(x, diff_table[0][0], dtype=float)  # Start with y₀
    
    # Add each term: (u choose k) * Δᵏy₀
    for k in range(1, n):
        # Calculate binomial coefficient C(u, k) = u(u-1)...(u-k+1) / k!
        binomial_coeff = np.ones_like(u)
        for i in range(k):
            binomial_coeff *= (u - i) / (i + 1)
        
        result += binomial_coeff * diff_table[0][k]
    
    return result[0] if is_scalar else result


def newton_backward_difference(x: Union[float, np.ndarray], 
                             x_points: List[float], 
                             y_points: List[float], 
                             h: float = None) -> Union[float, np.ndarray]:
    """
    Newton's backward difference formula for equally spaced points.
    
    Args:
        x: Point(s) at which to evaluate the interpolating polynomial
        x_points: List of x-coordinates of equally spaced data points
        y_points: List of y-coordinates of data points
        h: Step size (spacing between x points). If None, calculated automatically
        
    Returns:
        Interpolated value(s) P(x)
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    n = len(x_points)
    
    # Calculate step size if not provided
    if h is None:
        h = x_points[1] - x_points[0]
        # Verify equally spaced points
        for i in range(1, n - 1):
            if abs((x_points[i + 1] - x_points[i]) - h) > 1e-10:
                raise ValueError("Points are not equally spaced. Use newton_interpolation instead.")
    
    # Create backward difference table
    diff_table = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill first column with y values
    for i in range(n):
        diff_table[i][0] = y_points[i]
    
    # Calculate backward differences
    for j in range(1, n):
        for i in range(j, n):
            diff_table[i][j] = diff_table[i][j - 1] - diff_table[i - 1][j - 1]
    
    # Check if x is a scalar or array
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array([x])
    else:
        x = np.array(x)
    
    # Calculate u = (x - xₙ) / h (from the last point)
    u = (x - x_points[-1]) / h
    
    result = np.full_like(x, diff_table[-1][0], dtype=float)  # Start with yₙ
    
    # Add each term: (u+k-1 choose k) * ∇ᵏyₙ
    for k in range(1, n):
        # Calculate binomial coefficient C(u+k-1, k) = (u+k-1)(u+k-2)...(u) / k!
        binomial_coeff = np.ones_like(u)
        for i in range(k):
            binomial_coeff *= (u + i) / (i + 1)
        
        result += binomial_coeff * diff_table[-1][k]
    
    return result[0] if is_scalar else result


def print_divided_difference_table(x_points: List[float], y_points: List[float]) -> None:
    """
    Print the divided difference table in a formatted way.
    
    Args:
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
    """
    dd_table = divided_differences(x_points, y_points)
    n = len(x_points)
    
    print("Divided Difference Table:")
    print("=" * 80)
    
    # Header
    header = "x\t\tf[x]\t\t"
    for i in range(1, n):
        if i == 1:
            header += "f[x,x]\t\t"
        else:
            header += f"f[x,...,x]({i})\t"
    print(header)
    print("-" * 80)
    
    # Table content
    for i in range(n):
        row = f"{x_points[i]:.4f}\t\t{dd_table[i][0]:.6f}\t\t"
        for j in range(1, n - i):
            row += f"{dd_table[i][j]:.6f}\t\t"
        print(row)
    
    print("=" * 80)


def plot_newton_interpolation(x_points: List[float], 
                            y_points: List[float], 
                            x_range: Tuple[float, float] = None,
                            num_points: int = 100,
                            method: str = "divided",
                            title: str = "Newton Interpolation") -> None:
    """
    Plot the Newton interpolating polynomial along with the data points.
    
    Args:
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        x_range: Tuple of (min, max) for plotting range. If None, uses data range
        num_points: Number of points for smooth curve plotting
        method: "divided", "forward", or "backward"
        title: Title for the plot
    """
    if x_range is None:
        x_min, x_max = min(x_points), max(x_points)
        margin = 0.2 * (x_max - x_min)
        x_range = (x_min - margin, x_max + margin)
    
    # Generate points for smooth curve
    x_plot = np.linspace(x_range[0], x_range[1], num_points)
    
    if method == "divided":
        y_plot = newton_interpolation(x_plot, x_points, y_points)
    elif method == "forward":
        y_plot = newton_forward_difference(x_plot, x_points, y_points)
    elif method == "backward":
        y_plot = newton_backward_difference(x_plot, x_points, y_points)
    else:
        raise ValueError("Method must be 'divided', 'forward', or 'backward'")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'Newton {method.title()} Interpolation')
    plt.plot(x_points, y_points, 'ro', markersize=8, label='Data Points')
    
    # Add labels and grid
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate data points
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def example_usage():
    """
    Demonstrate Newton's interpolation methods with examples.
    """
    print("=== Newton's Interpolation Methods ===\n")
    
    # Example 1: Divided differences with unequally spaced points
    print("Example 1: Newton's Divided Difference Method")
    x_points = [1, 1.3, 1.6, 1.9, 2.2]
    y_points = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    
    print(f"Data points: {list(zip(x_points, y_points))}")
    print_divided_difference_table(x_points, y_points)
    
    # Test interpolation
    test_x = [1.5, 2.0]
    print(f"\nInterpolated values:")
    for x_val in test_x:
        y_val = newton_interpolation(x_val, x_points, y_points)
        print(f"P({x_val}) = {y_val:.6f}")
    
    # Example 2: Forward differences with equally spaced points
    print("\n" + "="*60)
    print("Example 2: Newton's Forward Difference Method")
    x_points2 = [0, 1, 2, 3, 4]
    y_points2 = [1, 4, 9, 16, 25]  # y = x^2 + 3x + 1 doesn't fit, let's use y = (x+1)^2
    
    print(f"Data points (equally spaced): {list(zip(x_points2, y_points2))}")
    
    # Test forward difference interpolation
    test_x2 = [0.5, 1.5, 2.5, 3.5]
    print(f"\nForward difference interpolation:")
    for x_val in test_x2:
        y_val = newton_forward_difference(x_val, x_points2, y_points2)
        print(f"P({x_val}) = {y_val:.6f}")
    
    # Example 3: Backward differences
    print("\n" + "="*60)
    print("Example 3: Newton's Backward Difference Method")
    
    # Test backward difference interpolation
    print(f"Backward difference interpolation (same data):")
    for x_val in test_x2:
        y_val = newton_backward_difference(x_val, x_points2, y_points2)
        print(f"P({x_val}) = {y_val:.6f}")
    
    # Example 4: Comparison with known function
    print("\n" + "="*60)
    print("Example 4: Error Analysis")
    
    import math
    # Use points from e^x
    x_exp = [0, 0.25, 0.5, 0.75, 1.0]
    y_exp = [math.exp(x) for x in x_exp]
    
    print(f"Interpolating e^x through {len(x_exp)} points")
    
    # Test at intermediate points
    test_points = [0.1, 0.3, 0.6, 0.8]
    print("Point\t\tActual e^x\tInterpolated\tError")
    print("-" * 55)
    
    for x_val in test_points:
        actual = math.exp(x_val)
        interpolated = newton_interpolation(x_val, x_exp, y_exp)
        error = abs(actual - interpolated)
        print(f"{x_val:.2f}\t\t{actual:.6f}\t{interpolated:.6f}\t{error:.6e}")


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1: Divided differences
    x_points = [1, 1.3, 1.6, 1.9, 2.2]
    y_points = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    plot_newton_interpolation(x_points, y_points, method="divided",
                             title="Newton Divided Difference Interpolation")
    
    # Plot Example 2: Forward differences
    x_points2 = [0, 1, 2, 3, 4]
    y_points2 = [1, 4, 9, 16, 25]
    plot_newton_interpolation(x_points2, y_points2, method="forward",
                             title="Newton Forward Difference Interpolation")