"""
Neural Odyssey - Week 1 Linear Algebra Exercises
================================================

These exercises will build your understanding of linear algebra from the ground up.
You'll implement fundamental operations without NumPy to truly understand the mechanics.

Complete the functions below and run the test cases to verify your implementation.
Remember: The goal is understanding, not just getting the right answer!

Author: Neural Explorer
"""

import math
import random
from typing import List, Tuple, Union

# Type hints for clarity
Vector = List[float]
Matrix = List[List[float]]

# =============================================================================
# PART 1: VECTOR OPERATIONS
# =============================================================================

def vector_add(v1: Vector, v2: Vector) -> Vector:
    """
    Add two vectors element-wise.
    
    Args:
        v1: First vector [x1, y1, z1, ...]
        v2: Second vector [x2, y2, z2, ...]
    
    Returns:
        Resulting vector [x1+x2, y1+y2, z1+z2, ...]
    
    Example:
        >>> vector_add([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    # TODO: Implement vector addition
    # Hint: Use a list comprehension or loop through indices
    pass


def vector_scalar_multiply(v: Vector, scalar: float) -> Vector:
    """
    Multiply a vector by a scalar.
    
    Args:
        v: Vector to multiply
        scalar: Number to multiply each element by
    
    Returns:
        Scaled vector
    
    Example:
        >>> vector_scalar_multiply([1, 2, 3], 2.5)
        [2.5, 5.0, 7.5]
    """
    # TODO: Implement scalar multiplication
    pass


def vector_dot_product(v1: Vector, v2: Vector) -> float:
    """
    Compute the dot product of two vectors.
    
    The dot product measures how much two vectors point in the same direction.
    In ML, it's used to measure similarity between data points!
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Dot product (a single number)
    
    Example:
        >>> vector_dot_product([1, 2, 3], [4, 5, 6])
        32.0
    """
    # TODO: Implement dot product
    # Formula: v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n]*v2[n]
    pass


def vector_magnitude(v: Vector) -> float:
    """
    Compute the magnitude (length) of a vector.
    
    This is the distance from the origin to the point represented by the vector.
    In ML, this often represents the "size" or "importance" of a feature vector.
    
    Args:
        v: Vector to compute magnitude for
    
    Returns:
        Magnitude of the vector
    
    Example:
        >>> vector_magnitude([3, 4])
        5.0
    """
    # TODO: Implement magnitude calculation
    # Hint: Use the Pythagorean theorem in n dimensions
    # |v| = sqrt(v1² + v2² + ... + vn²)
    pass


def vector_normalize(v: Vector) -> Vector:
    """
    Normalize a vector to unit length.
    
    A normalized vector has the same direction but magnitude 1.
    This is crucial in ML for comparing directions without being influenced by scale.
    
    Args:
        v: Vector to normalize
    
    Returns:
        Unit vector in the same direction
    
    Example:
        >>> vector_normalize([3, 4])
        [0.6, 0.8]
    """
    # TODO: Implement vector normalization
    # Hint: Divide each component by the vector's magnitude
    pass


def vector_angle_between(v1: Vector, v2: Vector) -> float:
    """
    Compute the angle between two vectors in radians.
    
    This tells us how similar two vectors are in direction.
    Angle of 0 = same direction, π/2 = perpendicular, π = opposite direction.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Angle in radians
    
    Example:
        >>> vector_angle_between([1, 0], [0, 1])
        1.5707963267948966  # π/2 radians = 90 degrees
    """
    # TODO: Implement angle calculation
    # Hint: Use the formula: cos(θ) = (v1 · v2) / (|v1| * |v2|)
    # Then use math.acos() to get the angle
    pass


# =============================================================================
# PART 2: MATRIX OPERATIONS
# =============================================================================

def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """
    Add two matrices element-wise.
    
    Args:
        A: First matrix
        B: Second matrix
    
    Returns:
        Sum of the matrices
    
    Example:
        >>> matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6, 8], [10, 12]]
    """
    # TODO: Implement matrix addition
    pass


def matrix_scalar_multiply(A: Matrix, scalar: float) -> Matrix:
    """
    Multiply a matrix by a scalar.
    
    Args:
        A: Matrix to multiply
        scalar: Number to multiply each element by
    
    Returns:
        Scaled matrix
    """
    # TODO: Implement scalar multiplication for matrices
    pass


def matrix_transpose(A: Matrix) -> Matrix:
    """
    Transpose a matrix (swap rows and columns).
    
    This operation is fundamental in linear algebra and appears everywhere in ML.
    The transpose of A is often written as A^T or A'.
    
    Args:
        A: Matrix to transpose
    
    Returns:
        Transposed matrix
    
    Example:
        >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    # TODO: Implement matrix transpose
    # Hint: The element at A[i][j] becomes A_transposed[j][i]
    pass


def matrix_vector_multiply(A: Matrix, v: Vector) -> Vector:
    """
    Multiply a matrix by a vector.
    
    This is one of the most important operations in ML!
    Each element of the result is the dot product of a matrix row with the vector.
    
    Args:
        A: Matrix (m x n)
        v: Vector (n elements)
    
    Returns:
        Resulting vector (m elements)
    
    Example:
        >>> matrix_vector_multiply([[1, 2], [3, 4]], [5, 6])
        [17, 39]
    """
    # TODO: Implement matrix-vector multiplication
    # Hint: Each result element is the dot product of a matrix row with the vector
    pass


def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiply two matrices.
    
    This is the operation that powers neural networks!
    Each element C[i][j] = dot product of row i of A with column j of B.
    
    Args:
        A: First matrix (m x k)
        B: Second matrix (k x n)
    
    Returns:
        Product matrix (m x n)
    
    Example:
        >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
    """
    # TODO: Implement matrix multiplication
    # This is the most challenging function - take your time!
    # Remember: A[i][j] of result = sum(A[i][k] * B[k][j] for all k)
    pass


def matrix_determinant_2x2(A: Matrix) -> float:
    """
    Compute the determinant of a 2x2 matrix.
    
    The determinant tells us if a matrix is invertible and measures how much
    the matrix scales areas/volumes. Very important for understanding when
    linear systems have unique solutions!
    
    Args:
        A: 2x2 matrix
    
    Returns:
        Determinant value
    
    Example:
        >>> matrix_determinant_2x2([[1, 2], [3, 4]])
        -2.0
    """
    # TODO: Implement 2x2 determinant
    # Formula for 2x2: det([[a, b], [c, d]]) = a*d - b*c
    pass


def matrix_inverse_2x2(A: Matrix) -> Matrix:
    """
    Compute the inverse of a 2x2 matrix.
    
    The inverse A^(-1) has the property that A * A^(-1) = I (identity matrix).
    This is used to solve linear systems: Ax = b -> x = A^(-1)b
    
    Args:
        A: 2x2 matrix
    
    Returns:
        Inverse matrix
    
    Raises:
        ValueError: If matrix is not invertible (determinant is 0)
    
    Example:
        >>> matrix_inverse_2x2([[1, 2], [3, 4]])
        [[-2.0, 1.0], [1.5, -0.5]]
    """
    # TODO: Implement 2x2 matrix inverse
    # Formula: A^(-1) = (1/det(A)) * [[d, -b], [-c, a]] for [[a, b], [c, d]]
    pass


# =============================================================================
# PART 3: PRACTICAL APPLICATIONS
# =============================================================================

def linear_regression_normal_equation(X: Matrix, y: Vector) -> Vector:
    """
    Solve linear regression using the normal equation.
    
    This is pure linear algebra! The normal equation gives us the optimal
    parameters for linear regression: θ = (X^T X)^(-1) X^T y
    
    Args:
        X: Feature matrix (each row is a data point)
        y: Target values
    
    Returns:
        Optimal parameters θ
    
    Note:
        This implementation assumes X^T X is invertible (which it might not be
        for real data, but works for our exercises).
    """
    # TODO: Implement the normal equation
    # Steps:
    # 1. Compute X^T (transpose of X)
    # 2. Compute X^T * X
    # 3. Compute inverse of (X^T * X)
    # 4. Compute X^T * y
    # 5. Multiply: (X^T * X)^(-1) * (X^T * y)
    pass


def transform_2d_points(points: List[Tuple[float, float]], 
                       transformation_matrix: Matrix) -> List[Tuple[float, float]]:
    """
    Apply a 2D transformation to a list of points.
    
    This demonstrates how matrices transform geometric objects.
    Different matrices create different effects:
    - Rotation matrices rotate points
    - Scaling matrices stretch/shrink
    - Reflection matrices flip across axes
    
    Args:
        points: List of (x, y) coordinates
        transformation_matrix: 2x2 transformation matrix
    
    Returns:
        List of transformed (x, y) coordinates
    """
    # TODO: Implement 2D point transformation
    # Hint: Convert each point to a vector, multiply by matrix, convert back to tuple
    pass


def create_rotation_matrix(angle_radians: float) -> Matrix:
    """
    Create a 2D rotation matrix.
    
    Args:
        angle_radians: Angle to rotate (positive = counterclockwise)
    
    Returns:
        2x2 rotation matrix
    
    Formula:
        [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
    """
    # TODO: Implement rotation matrix creation
    pass


def create_scaling_matrix(scale_x: float, scale_y: float) -> Matrix:
    """
    Create a 2D scaling matrix.
    
    Args:
        scale_x: Scaling factor for x-axis
        scale_y: Scaling factor for y-axis
    
    Returns:
        2x2 scaling matrix
    
    Formula:
        [[scale_x,    0    ],
         [   0   , scale_y ]]
    """
    # TODO: Implement scaling matrix creation
    pass


# =============================================================================
# PART 4: DATA ANALYSIS WITH LINEAR ALGEBRA
# =============================================================================

def compute_data_center(data_matrix: Matrix) -> Vector:
    """
    Compute the center (mean) of a dataset.
    
    This is the first step in many ML algorithms like PCA.
    
    Args:
        data_matrix: Each row is a data point, each column is a feature
    
    Returns:
        Vector representing the center point of the data
    """
    # TODO: Implement mean calculation
    # Hint: Average each column separately
    pass


def center_data(data_matrix: Matrix) -> Matrix:
    """
    Center the data by subtracting the mean from each data point.
    
    This is crucial for PCA and many other ML algorithms.
    Centered data has mean = [0, 0, ..., 0].
    
    Args:
        data_matrix: Original data
    
    Returns:
        Centered data matrix
    """
    # TODO: Implement data centering
    # Steps:
    # 1. Compute the center (mean) of the data
    # 2. Subtract the center from each data point
    pass


def compute_covariance_matrix(centered_data: Matrix) -> Matrix:
    """
    Compute the covariance matrix of centered data.
    
    The covariance matrix tells us how features relate to each other.
    This is the foundation of Principal Component Analysis (PCA)!
    
    Args:
        centered_data: Data that has been centered (mean subtracted)
    
    Returns:
        Covariance matrix
    
    Formula:
        Cov = (1/n) * X^T * X  (where X is centered data)
    """
    # TODO: Implement covariance matrix calculation
    # Steps:
    # 1. Transpose the centered data
    # 2. Multiply transposed data by original centered data
    # 3. Scale by 1/n (where n is number of data points)
    pass


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def run_tests():
    """Run all test cases to verify implementations."""
    
    print("🧠 Neural Odyssey - Linear Algebra Tests")
    print("=" * 50)
    
    # Test vector operations
    print("\n📐 Testing Vector Operations...")
    
    try:
        # Test vector addition
        result = vector_add([1, 2, 3], [4, 5, 6])
        expected = [5, 7, 9]
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ Vector addition: PASSED")
        
        # Test scalar multiplication
        result = vector_scalar_multiply([1, 2, 3], 2)
        expected = [2, 4, 6]
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ Vector scalar multiplication: PASSED")
        
        # Test dot product
        result = vector_dot_product([1, 2, 3], [4, 5, 6])
        expected = 32
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        print("✅ Vector dot product: PASSED")
        
        # Test magnitude
        result = vector_magnitude([3, 4])
        expected = 5.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        print("✅ Vector magnitude: PASSED")
        
    except (AssertionError, Exception) as e:
        print(f"❌ Vector operations test failed: {e}")
    
    # Test matrix operations
    print("\n🔢 Testing Matrix Operations...")
    
    try:
        # Test matrix multiplication
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = matrix_multiply(A, B)
        expected = [[19, 22], [43, 50]]
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ Matrix multiplication: PASSED")
        
        # Test transpose
        result = matrix_transpose([[1, 2, 3], [4, 5, 6]])
        expected = [[1, 4], [2, 5], [3, 6]]
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ Matrix transpose: PASSED")
        
        # Test determinant
        result = matrix_determinant_2x2([[1, 2], [3, 4]])
        expected = -2.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        print("✅ Matrix determinant: PASSED")
        
    except (AssertionError, Exception) as e:
        print(f"❌ Matrix operations test failed: {e}")
    
    print("\n🎯 All basic tests completed!")
    print("\nNext steps:")
    print("1. Implement any missing functions")
    print("2. Run the practical examples below")
    print("3. Experiment with your own data!")


def demonstrate_linear_regression():
    """Demonstrate linear regression using our linear algebra functions."""
    
    print("\n🔍 Linear Regression Demonstration")
    print("=" * 40)
    
    # Create simple dataset: y = 2x + 1 + noise
    X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]  # First column is bias term
    y = [3.1, 5.2, 6.9, 9.1, 11.0]  # y ≈ 2x + 1
    
    print(f"Data: X = {X}")
    print(f"      y = {y}")
    
    try:
        # Solve using normal equation
        theta = linear_regression_normal_equation(X, y)
        print(f"\nSolved parameters: θ = {theta}")
        print(f"Interpretation: y = {theta[1]:.2f}x + {theta[0]:.2f}")
        
        # Make predictions
        predictions = [matrix_vector_multiply([row], theta)[0] for row in X]
        print(f"Predictions: {[f'{p:.2f}' for p in predictions]}")
        
    except Exception as e:
        print(f"❌ Linear regression failed: {e}")
        print("Make sure you've implemented all required matrix functions!")


def demonstrate_2d_transformations():
    """Demonstrate 2D geometric transformations."""
    
    print("\n🎨 2D Transformations Demonstration")
    print("=" * 40)
    
    # Create a simple square
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    print(f"Original square: {square}")
    
    try:
        # Rotation by 45 degrees
        rotation_45 = create_rotation_matrix(math.pi / 4)  # 45 degrees
        rotated_square = transform_2d_points(square, rotation_45)
        print(f"After 45° rotation: {[(f'{x:.2f}', f'{y:.2f}') for x, y in rotated_square]}")
        
        # Scaling by 2x in both directions
        scaling_2x = create_scaling_matrix(2, 2)
        scaled_square = transform_2d_points(square, scaling_2x)
        print(f"After 2x scaling: {scaled_square}")
        
    except Exception as e:
        print(f"❌ Transformations failed: {e}")
        print("Make sure you've implemented the transformation functions!")


def demonstrate_data_analysis():
    """Demonstrate data analysis using linear algebra."""
    
    print("\n📊 Data Analysis Demonstration")
    print("=" * 40)
    
    # Create sample dataset (height and weight data)
    data = [
        [170, 65],  # height, weight
        [175, 70],
        [165, 60],
        [180, 75],
        [160, 55],
        [185, 80]
    ]
    
    print(f"Sample data (height, weight): {data}")
    
    try:
        # Compute center
        center = compute_data_center(data)
        print(f"Data center (mean): {[f'{x:.2f}' for x in center]}")
        
        # Center the data
        centered = center_data(data)
        print(f"Centered data: {[[f'{x:.2f}' for x in row] for row in centered]}")
        
        # Compute covariance matrix
        cov_matrix = compute_covariance_matrix(centered)
        print(f"Covariance matrix: {[[f'{x:.2f}' for x in row] for row in cov_matrix]}")
        print("This tells us how height and weight are correlated!")
        
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        print("Make sure you've implemented the data analysis functions!")


if __name__ == "__main__":
    """
    Run this file to test your implementations and see linear algebra in action!
    
    Complete the TODO functions above, then run:
    python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey Linear Algebra!")
    print("Complete the TODO functions and run tests to verify your implementation.")
    print("\nTo get started:")
    print("1. Implement the vector operations first")
    print("2. Move on to matrix operations") 
    print("3. Tackle the practical applications")
    print("4. Run tests to verify everything works!")
    
    # Uncomment these lines after implementing the functions:
    # run_tests()
    # demonstrate_linear_regression()
    # demonstrate_2d_transformations()
    # demonstrate_data_analysis()
    
    print("\n💡 Pro tip: Start with vector_add() and work your way through!")
    print("Remember: Understanding is more important than just getting the right answer!")