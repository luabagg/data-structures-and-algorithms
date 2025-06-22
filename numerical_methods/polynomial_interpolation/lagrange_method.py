"""
Lagrange Method for Polynomial Interpolation

The Lagrange interpolation method constructs a polynomial of degree n-1 
that passes through n given data points (x_i, y_i).

The Lagrange interpolating polynomial is given by:
P(x) = Σ(i=0 to n-1) y_i * L_i(x)

where L_i(x) are the Lagrange basis polynomials:
L_i(x) = Π(j=0 to n-1, j≠i) (x - x_j) / (x_i - x_j)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def lagrange_basis(x: float, x_points: List[float], i: int) -> float:
    """
    Calculate the i-th Lagrange basis polynomial L_i(x) at point x.
    
    Args:
        x: Point at which to evaluate the basis polynomial
        x_points: List of x-coordinates of data points
        i: Index of the basis polynomial
        
    Returns:
        Value of L_i(x)
    """
    n = len(x_points)
    result = 1.0
    
    for j in range(n):
        if j != i:
            result *= (x - x_points[j]) / (x_points[i] - x_points[j])
    
    return result


def lagrange_interpolation(x: Union[float, np.ndarray], 
                         x_points: List[float], 
                         y_points: List[float]) -> Union[float, np.ndarray]:
    """
    Perform Lagrange interpolation to find P(x) for given data points.
    
    Args:
        x: Point(s) at which to evaluate the interpolating polynomial
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        
    Returns:
        Interpolated value(s) P(x)
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    n = len(x_points)
    
    # Check if x is a scalar or array
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array([x])
    else:
        x = np.array(x)
    
    result = np.zeros_like(x, dtype=float)
    
    # Calculate P(x) = Σ y_i * L_i(x)
    for i in range(n):
        basis_values = np.ones_like(x, dtype=float)
        
        for j in range(n):
            if j != i:
                basis_values *= (x - x_points[j]) / (x_points[i] - x_points[j])
        
        result += y_points[i] * basis_values
    
    return result[0] if is_scalar else result


def lagrange_coefficients(x_points: List[float], y_points: List[float]) -> List[float]:
    """
    Calculate the coefficients of the Lagrange interpolating polynomial
    in standard form (descending powers of x).
    
    Args:
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        
    Returns:
        List of coefficients [a_n-1, a_n-2, ..., a_1, a_0] where
        P(x) = a_n-1 * x^(n-1) + ... + a_1 * x + a_0
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    n = len(x_points)
    coeffs = np.zeros(n)
    
    for i in range(n):
        # Calculate coefficients for y_i * L_i(x)
        basis_coeffs = np.array([1.0])
        
        for j in range(n):
            if j != i:
                # Multiply by (x - x_j) / (x_i - x_j)
                factor = 1.0 / (x_points[i] - x_points[j])
                # Polynomial multiplication: multiply by (x - x_j)
                new_coeffs = np.zeros(len(basis_coeffs) + 1)
                new_coeffs[:-1] += basis_coeffs  # x term
                new_coeffs[1:] -= x_points[j] * basis_coeffs  # constant term
                basis_coeffs = new_coeffs * factor
        
        # Add contribution to final polynomial
        # Pad with zeros if necessary
        if len(basis_coeffs) < n:
            basis_coeffs = np.pad(basis_coeffs, (n - len(basis_coeffs), 0), 'constant')
        
        coeffs += y_points[i] * basis_coeffs
    
    return coeffs.tolist()


def plot_lagrange_interpolation(x_points: List[float], 
                               y_points: List[float], 
                               x_range: Tuple[float, float] = None,
                               num_points: int = 100,
                               title: str = "Lagrange Interpolation") -> None:
    """
    Plot the Lagrange interpolating polynomial along with the data points.
    
    Args:
        x_points: List of x-coordinates of data points
        y_points: List of y-coordinates of data points
        x_range: Tuple of (min, max) for plotting range. If None, uses data range
        num_points: Number of points for smooth curve plotting
        title: Title for the plot
    """
    if x_range is None:
        x_min, x_max = min(x_points), max(x_points)
        margin = 0.2 * (x_max - x_min)
        x_range = (x_min - margin, x_max + margin)
    
    # Generate points for smooth curve
    x_plot = np.linspace(x_range[0], x_range[1], num_points)
    y_plot = lagrange_interpolation(x_plot, x_points, y_points)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Lagrange Polynomial')
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
    Demonstrate the Lagrange interpolation method with examples.
    """
    print("=== Lagrange Interpolation Method ===\n")
    
    # Example 1: Simple quadratic
    print("Example 1: Quadratic function through 3 points")
    x_points = [0, 1, 2]
    y_points = [1, 4, 9]  # y = x^2 + 3x + 1
    
    print(f"Data points: {list(zip(x_points, y_points))}")
    
    # Interpolate at specific points
    test_x = [0.5, 1.5, 2.5]
    for x_val in test_x:
        y_val = lagrange_interpolation(x_val, x_points, y_points)
        print(f"P({x_val}) = {y_val:.6f}")
    
    # Get polynomial coefficients
    coeffs = lagrange_coefficients(x_points, y_points)
    print(f"Polynomial coefficients (descending powers): {[f'{c:.6f}' for c in coeffs]}")
    
    # Example 2: More complex example
    print("\nExample 2: Interpolation through 5 points")
    x_points2 = [-2, -1, 0, 1, 2]
    y_points2 = [16, 1, 0, 1, 16]  # y = x^4
    
    print(f"Data points: {list(zip(x_points2, y_points2))}")
    
    # Test interpolation
    test_x2 = [-1.5, -0.5, 0.5, 1.5]
    print("Interpolated values:")
    for x_val in test_x2:
        y_val = lagrange_interpolation(x_val, x_points2, y_points2)
        print(f"P({x_val}) = {y_val:.6f}")
    
    # Example 3: Error analysis
    print("\nExample 3: Error analysis with known function")
    
    # Use points from sin(x) and compare interpolation with actual values
    import math
    x_sin = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
    y_sin = [math.sin(x) for x in x_sin]
    
    print(f"Interpolating sin(x) through {len(x_sin)} points")
    
    # Test at intermediate points
    test_points = [math.pi/8, 3*math.pi/8, 5*math.pi/8, 7*math.pi/8]
    print("Point\t\tActual sin(x)\tInterpolated\tError")
    print("-" * 55)
    
    for x_val in test_points:
        actual = math.sin(x_val)
        interpolated = lagrange_interpolation(x_val, x_sin, y_sin)
        error = abs(actual - interpolated)
        print(f"{x_val:.6f}\t{actual:.6f}\t{interpolated:.6f}\t{error:.6e}")


if __name__ == "__main__":
    example_usage()
    
    print("\nGenerating plots...")
    
    # Plot Example 1
    x_points = [0, 1, 2]
    y_points = [1, 4, 9]
    plot_lagrange_interpolation(x_points, y_points, title="Quadratic Interpolation")
    
    # Plot Example 2
    x_points2 = [-2, -1, 0, 1, 2]
    y_points2 = [16, 1, 0, 1, 16]
    plot_lagrange_interpolation(x_points2, y_points2, title="Fourth-degree Polynomial")