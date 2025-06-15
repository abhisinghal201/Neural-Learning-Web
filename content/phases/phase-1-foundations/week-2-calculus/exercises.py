"""
Neural Odyssey - Week 2 Calculus Exercises
==========================================

These exercises will build your understanding of calculus as the engine of machine learning.
You'll implement gradient descent, backpropagation, and optimization algorithms from scratch.

The goal is to understand how calculus powers every learning algorithm!

Author: Neural Explorer
"""

import math
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
import numpy as np  # For visualization only - implement core logic yourself!

# Type hints for clarity
Function = Callable[[float], float]
MultiVarFunction = Callable[[List[float]], float]
Vector = List[float]

# =============================================================================
# PART 1: DERIVATIVES AND NUMERICAL DIFFERENTIATION
# =============================================================================

def numerical_derivative(f: Function, x: float, h: float = 1e-7) -> float:
    """
    Compute the derivative of a function at point x using numerical differentiation.
    
    This is how computers approximate derivatives when we don't have analytical formulas.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute derivative
        h: Small step size (smaller = more accurate, but beware of floating point errors)
    
    Returns:
        Approximate derivative f'(x)
    
    Example:
        >>> def f(x): return x**2
        >>> numerical_derivative(f, 3.0)  # Should be close to 6.0
        6.000000000139778
    """
    # TODO: Implement numerical differentiation using the formula:
    # f'(x) ≈ (f(x + h) - f(x - h)) / (2h)  [central difference method]
    # This is more accurate than forward difference: (f(x + h) - f(x)) / h
    pass


def numerical_gradient(f: MultiVarFunction, x: Vector, h: float = 1e-7) -> Vector:
    """
    Compute the gradient of a multivariable function using numerical differentiation.
    
    The gradient is a vector of partial derivatives - it tells us how the function
    changes with respect to each input variable.
    
    Args:
        f: Multivariable function
        x: Point at which to compute gradient
        h: Small step size
    
    Returns:
        Gradient vector [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    Example:
        >>> def f(x): return x[0]**2 + x[1]**2  # f(x,y) = x² + y²
        >>> numerical_gradient(f, [1.0, 2.0])   # Should be [2.0, 4.0]
        [2.0000000000002, 4.000000000000398]
    """
    # TODO: Implement gradient computation
    # For each variable i, compute ∂f/∂xᵢ by varying only xᵢ while keeping others constant
    pass


def plot_function_and_derivative(f: Function, f_prime: Function, x_min: float, x_max: float):
    """
    Plot a function and its derivative to visualize their relationship.
    
    Args:
        f: Original function
        f_prime: Derivative function
        x_min: Start of plotting range
        x_max: End of plotting range
    """
    x_values = np.linspace(x_min, x_max, 1000)
    y_values = [f(x) for x in x_values]
    y_prime_values = [f_prime(x) for x in x_values]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, 'b-', label='f(x)', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Original Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_values, y_prime_values, 'r-', label="f'(x)", linewidth=2)
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# PART 2: GRADIENT DESCENT OPTIMIZATION
# =============================================================================

def gradient_descent_1d(f: Function, f_prime: Function, x_start: float, 
                       learning_rate: float = 0.01, num_steps: int = 100,
                       tolerance: float = 1e-6) -> Tuple[float, List[float]]:
    """
    Perform gradient descent on a 1D function.
    
    This is the fundamental optimization algorithm behind machine learning!
    
    Args:
        f: Function to minimize
        f_prime: Derivative of the function
        x_start: Starting point
        learning_rate: Step size (how big steps to take)
        num_steps: Maximum number of iterations
        tolerance: Stop when change is smaller than this
    
    Returns:
        Tuple of (final_x, x_history) where x_history tracks the optimization path
    
    Example:
        >>> def f(x): return (x - 2)**2 + 1  # Minimum at x = 2
        >>> def f_prime(x): return 2 * (x - 2)
        >>> final_x, history = gradient_descent_1d(f, f_prime, 0.0)
        >>> abs(final_x - 2.0) < 0.01  # Should converge near x = 2
        True
    """
    # TODO: Implement 1D gradient descent
    # Algorithm:
    # 1. Start at x_start
    # 2. For each step:
    #    - Compute gradient (slope) at current point
    #    - Take step in negative gradient direction: x_new = x - learning_rate * gradient
    #    - Check for convergence
    # 3. Return final x and history for visualization
    pass


def gradient_descent_multi(f: MultiVarFunction, x_start: Vector,
                          learning_rate: float = 0.01, num_steps: int = 1000,
                          tolerance: float = 1e-6) -> Tuple[Vector, List[Vector]]:
    """
    Perform gradient descent on a multivariable function.
    
    This is what happens inside neural network training!
    
    Args:
        f: Function to minimize
        x_start: Starting point (vector)
        learning_rate: Step size
        num_steps: Maximum iterations
        tolerance: Convergence threshold
    
    Returns:
        Tuple of (final_x, x_history)
    
    Example:
        >>> def f(x): return x[0]**2 + x[1]**2  # Bowl-shaped function, minimum at [0,0]
        >>> final_x, _ = gradient_descent_multi(f, [5.0, 3.0])
        >>> abs(final_x[0]) < 0.01 and abs(final_x[1]) < 0.01  # Should converge to [0,0]
        True
    """
    # TODO: Implement multivariable gradient descent
    # Same algorithm as 1D, but now:
    # - x is a vector
    # - gradient is a vector
    # - update: x_new = x - learning_rate * gradient_vector
    pass


def plot_gradient_descent_1d(f: Function, f_prime: Function, x_start: float,
                            learning_rate: float = 0.01, num_steps: int = 50):
    """
    Visualize gradient descent optimization on a 1D function.
    
    This helps you understand how the algorithm navigates the function landscape.
    """
    # Get optimization path
    final_x, x_history = gradient_descent_1d(f, f_prime, x_start, learning_rate, num_steps)
    
    # Create plot
    x_range = np.linspace(min(x_history) - 1, max(x_history) + 1, 1000)
    y_range = [f(x) for x in x_range]
    
    plt.figure(figsize=(12, 5))
    
    # Plot function and optimization path
    plt.subplot(1, 2, 1)
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
    
    # Plot optimization steps
    for i, x in enumerate(x_history[::5]):  # Show every 5th step to avoid clutter
        y = f(x)
        plt.plot(x, y, 'ro', markersize=8, alpha=0.7)
        if i == 0:
            plt.annotate('Start', (x, y), xytext=(5, 5), textcoords='offset points')
        elif i == len(x_history[::5]) - 1:
            plt.annotate('End', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent Path (lr={learning_rate})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot convergence
    plt.subplot(1, 2, 2)
    f_history = [f(x) for x in x_history]
    plt.plot(f_history, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Convergence Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Started at x = {x_start:.3f}, f(x) = {f(x_start):.3f}")
    print(f"Ended at x = {final_x:.3f}, f(x) = {f(final_x):.3f}")
    print(f"Improvement: {f(x_start) - f(final_x):.3f}")


def plot_gradient_descent_2d(f: MultiVarFunction, x_start: Vector,
                            learning_rate: float = 0.01, num_steps: int = 100):
    """
    Visualize gradient descent on a 2D function using contour plots.
    
    This shows how the algorithm navigates the "loss landscape".
    """
    # Get optimization path
    final_x, x_history = gradient_descent_multi(f, x_start, learning_rate, num_steps)
    
    # Create meshgrid for contour plot
    x_coords = [point[0] for point in x_history]
    y_coords = [point[1] for point in x_history]
    
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[f([x, y]) for x in x_range] for y in y_range])
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=20, alpha=0.7)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization path
    x_path = [point[0] for point in x_history]
    y_path = [point[1] for point in x_history]
    
    plt.plot(x_path, y_path, 'r-', linewidth=2, alpha=0.8, label='Optimization Path')
    plt.plot(x_path[0], y_path[0], 'go', markersize=10, label='Start')
    plt.plot(x_path[-1], y_path[-1], 'ro', markersize=10, label='End')
    
    # Add arrows to show direction
    for i in range(0, len(x_path) - 1, max(1, len(x_path) // 10)):
        dx = x_path[i + 1] - x_path[i]
        dy = y_path[i + 1] - y_path[i]
        plt.arrow(x_path[i], y_path[i], dx, dy, head_width=0.1, head_length=0.1, 
                 fc='red', ec='red', alpha=0.7)
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'Gradient Descent on 2D Function (lr={learning_rate})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    print(f"Started at {x_start}, f(x) = {f(x_start):.3f}")
    print(f"Ended at [{final_x[0]:.3f}, {final_x[1]:.3f}], f(x) = {f(final_x):.3f}")


# =============================================================================
# PART 3: LEARNING RATE EFFECTS AND OPTIMIZATION CHALLENGES
# =============================================================================

def compare_learning_rates(f: Function, f_prime: Function, x_start: float,
                          learning_rates: List[float], num_steps: int = 50):
    """
    Compare how different learning rates affect gradient descent convergence.
    
    This demonstrates one of the most important hyperparameters in ML!
    
    Args:
        f: Function to minimize
        f_prime: Derivative
        x_start: Starting point
        learning_rates: List of learning rates to compare
        num_steps: Number of optimization steps
    """
    plt.figure(figsize=(15, 5))
    
    # Plot the function
    x_range = np.linspace(x_start - 2, x_start + 2, 1000)
    y_range = [f(x) for x in x_range]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(1, len(learning_rates), i + 1)
        plt.plot(x_range, y_range, 'k-', alpha=0.3, linewidth=1)
        
        try:
            final_x, x_history = gradient_descent_1d(f, f_prime, x_start, lr, num_steps)
            
            # Plot optimization path
            for j, x in enumerate(x_history[::max(1, len(x_history)//20)]):
                y = f(x)
                alpha = 0.3 + 0.7 * (j / len(x_history[::max(1, len(x_history)//20)]))
                plt.plot(x, y, 'o', color=colors[i % len(colors)], alpha=alpha, markersize=4)
            
            plt.title(f'LR = {lr}\nSteps: {len(x_history)}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.grid(True, alpha=0.3)
            
            # Analyze convergence
            if len(x_history) < num_steps:
                plt.text(0.02, 0.98, 'Converged!', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                        verticalalignment='top')
            else:
                final_change = abs(x_history[-1] - x_history[-2]) if len(x_history) > 1 else float('inf')
                if final_change > 1.0:
                    plt.text(0.02, 0.98, 'Diverging!', transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                            verticalalignment='top')
                else:
                    plt.text(0.02, 0.98, 'Slow convergence', transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                            verticalalignment='top')
                    
        except Exception as e:
            plt.title(f'LR = {lr}\nFailed!')
            plt.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', transform=plt.gca().transAxes,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def momentum_gradient_descent(f: MultiVarFunction, x_start: Vector,
                             learning_rate: float = 0.01, momentum: float = 0.9,
                             num_steps: int = 1000, tolerance: float = 1e-6) -> Tuple[Vector, List[Vector]]:
    """
    Implement gradient descent with momentum.
    
    Momentum helps accelerate convergence and escape local minima by
    accumulating gradients from previous steps.
    
    Args:
        f: Function to minimize
        x_start: Starting point
        learning_rate: Step size
        momentum: Momentum coefficient (0 = no momentum, 0.9 = high momentum)
        num_steps: Maximum iterations
        tolerance: Convergence threshold
    
    Returns:
        Tuple of (final_x, x_history)
    """
    # TODO: Implement momentum gradient descent
    # Algorithm:
    # 1. Initialize velocity vector v = 0
    # 2. For each step:
    #    - Compute gradient g
    #    - Update velocity: v = momentum * v - learning_rate * g
    #    - Update position: x = x + v
    # 3. The momentum term helps the algorithm "remember" previous gradients
    pass


# =============================================================================
# PART 4: CHAIN RULE AND BACKPROPAGATION
# =============================================================================

class SimpleNeuron:
    """
    A simple neuron that demonstrates the chain rule in action.
    
    This is the building block of neural networks!
    """
    
    def __init__(self, weight: float, bias: float):
        self.weight = weight
        self.bias = bias
        # Cache for backpropagation
        self.last_input = None
        self.last_z = None
        self.last_output = None
    
    def forward(self, x: float) -> float:
        """
        Forward pass: compute output.
        
        Args:
            x: Input value
        
        Returns:
            Output after applying weight, bias, and activation
        """
        # TODO: Implement forward pass
        # 1. Compute z = weight * x + bias
        # 2. Apply activation function (use sigmoid)
        # 3. Cache intermediate values for backpropagation
        # 4. Return final output
        pass
    
    def backward(self, d_output: float) -> float:
        """
        Backward pass: compute gradients using chain rule.
        
        Args:
            d_output: Gradient of loss with respect to this neuron's output
        
        Returns:
            Gradient of loss with respect to this neuron's input
        """
        # TODO: Implement backward pass using chain rule
        # Chain rule: d_loss/d_input = d_loss/d_output * d_output/d_z * d_z/d_input
        # 
        # 1. d_output/d_z = sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        # 2. d_z/d_weight = input
        # 3. d_z/d_bias = 1
        # 4. d_z/d_input = weight
        #
        # Update self.weight and self.bias using the computed gradients
        # Return d_loss/d_input for the previous layer
        pass
    
    def sigmoid(self, z: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-z))
    
    def sigmoid_derivative(self, z: float) -> float:
        """Derivative of sigmoid function."""
        s = self.sigmoid(z)
        return s * (1 - s)


class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network to demonstrate backpropagation.
    
    Architecture: input → hidden neuron → output neuron
    """
    
    def __init__(self, hidden_weight: float, hidden_bias: float,
                 output_weight: float, output_bias: float):
        self.hidden = SimpleNeuron(hidden_weight, hidden_bias)
        self.output = SimpleNeuron(output_weight, output_bias)
        self.learning_rate = 0.1
    
    def forward(self, x: float) -> float:
        """Forward pass through the network."""
        hidden_output = self.hidden.forward(x)
        final_output = self.output.forward(hidden_output)
        return final_output
    
    def backward(self, x: float, target: float) -> float:
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            x: Input value
            target: Target output
        
        Returns:
            Loss value
        """
        # TODO: Implement backpropagation
        # 1. Compute forward pass to get prediction
        # 2. Compute loss = (prediction - target)²
        # 3. Compute gradient of loss w.r.t. output: d_loss/d_output = 2 * (prediction - target)
        # 4. Backpropagate through output neuron
        # 5. Backpropagate through hidden neuron
        # 6. Return loss for monitoring
        pass
    
    def train(self, training_data: List[Tuple[float, float]], epochs: int = 100) -> List[float]:
        """
        Train the network on data.
        
        Args:
            training_data: List of (input, target) pairs
            epochs: Number of training epochs
        
        Returns:
            List of loss values for each epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for x, target in training_data:
                loss = self.backward(x, target)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return loss_history


# =============================================================================
# PART 5: PRACTICAL ML APPLICATIONS
# =============================================================================

def linear_regression_gradient_descent(X: List[List[float]], y: List[float],
                                     learning_rate: float = 0.01, num_steps: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Implement linear regression using gradient descent.
    
    This shows how calculus powers one of the most fundamental ML algorithms!
    
    Args:
        X: Feature matrix (each row is a data point)
        y: Target values
        learning_rate: Step size
        num_steps: Number of gradient descent steps
    
    Returns:
        Tuple of (final_weights, loss_history)
    
    Model: y = w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ
    Loss: MSE = (1/m) * Σ(prediction - actual)²
    """
    # TODO: Implement linear regression with gradient descent
    # 
    # Algorithm:
    # 1. Initialize weights randomly
    # 2. For each step:
    #    a. Compute predictions: pred = X @ weights
    #    b. Compute loss: MSE = mean((pred - y)²)
    #    c. Compute gradients: d_loss/d_weights
    #    d. Update weights: weights -= learning_rate * gradients
    # 3. Return final weights and loss history
    #
    # Gradient formulas:
    # d_loss/d_w₀ = (2/m) * Σ(pred - y)           [bias term]
    # d_loss/d_wᵢ = (2/m) * Σ(pred - y) * xᵢ     [feature weights]
    pass


def logistic_regression_gradient_descent(X: List[List[float]], y: List[int],
                                        learning_rate: float = 0.01, num_steps: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Implement logistic regression using gradient descent.
    
    This demonstrates how calculus enables classification algorithms!
    
    Args:
        X: Feature matrix
        y: Binary labels (0 or 1)
        learning_rate: Step size
        num_steps: Number of steps
    
    Returns:
        Tuple of (final_weights, loss_history)
    
    Model: p = sigmoid(w₀ + w₁*x₁ + ... + wₙ*xₙ)
    Loss: Cross-entropy = -Σ[y*log(p) + (1-y)*log(1-p)]
    """
    # TODO: Implement logistic regression with gradient descent
    #
    # Algorithm:
    # 1. Initialize weights
    # 2. For each step:
    #    a. Compute logits: z = X @ weights
    #    b. Compute probabilities: p = sigmoid(z)
    #    c. Compute cross-entropy loss
    #    d. Compute gradients: d_loss/d_weights = X.T @ (p - y) / m
    #    e. Update weights
    # 3. Return weights and loss history
    pass


# =============================================================================
# PART 6: TESTING AND DEMONSTRATIONS
# =============================================================================

def test_derivatives():
    """Test numerical differentiation implementations."""
    print("🧮 Testing Numerical Derivatives")
    print("=" * 40)
    
    # Test cases with known analytical derivatives
    test_cases = [
        (lambda x: x**2, lambda x: 2*x, 3.0, "f(x) = x²"),
        (lambda x: x**3, lambda x: 3*x**2, 2.0, "f(x) = x³"),
        (lambda x: math.sin(x), lambda x: math.cos(x), math.pi/4, "f(x) = sin(x)"),
        (lambda x: math.exp(x), lambda x: math.exp(x), 1.0, "f(x) = eˣ"),
    ]
    
    for f, f_analytical, test_point, description in test_cases:
        numerical = numerical_derivative(f, test_point)
        analytical = f_analytical(test_point)
        error = abs(numerical - analytical)
        
        print(f"{description} at x={test_point}")
        print(f"  Numerical: {numerical:.6f}")
        print(f"  Analytical: {analytical:.6f}")
        print(f"  Error: {error:.6e}")
        print(f"  {'✅ PASS' if error < 1e-5 else '❌ FAIL'}")
        print()


def test_gradient_descent():
    """Test gradient descent implementations."""
    print("🎯 Testing Gradient Descent")
    print("=" * 40)
    
    # Test 1D optimization
    def quadratic(x):
        return (x - 3)**2 + 1  # Minimum at x = 3
    
    def quadratic_prime(x):
        return 2 * (x - 3)
    
    final_x, _ = gradient_descent_1d(quadratic, quadratic_prime, 0.0, 0.1, 100)
    error = abs(final_x - 3.0)
    print(f"1D Quadratic optimization:")
    print(f"  Target: x = 3.0")
    print(f"  Found: x = {final_x:.4f}")
    print(f"  Error: {error:.6f}")
    print(f"  {'✅ PASS' if error < 0.01 else '❌ FAIL'}")
    print()
    
    # Test 2D optimization
    def paraboloid(x):
        return x[0]**2 + x[1]**2  # Minimum at [0, 0]
    
    final_x, _ = gradient_descent_multi(paraboloid, [5.0, -3.0], 0.1, 1000)
    error = math.sqrt(final_x[0]**2 + final_x[1]**2)
    print(f"2D Paraboloid optimization:")
    print(f"  Target: [0.0, 0.0]")
    print(f"  Found: [{final_x[0]:.4f}, {final_x[1]:.4f}]")
    print(f"  Error: {error:.6f}")
    print(f"  {'✅ PASS' if error < 0.01 else '❌ FAIL'}")


def demonstrate_learning_rates():
    """Demonstrate the effect of different learning rates."""
    print("📈 Learning Rate Effects Demonstration")
    print("=" * 50)
    
    def steep_quadratic(x):
        return 5 * (x - 2)**2 + 1
    
    def steep_quadratic_prime(x):
        return 10 * (x - 2)
    
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3]
    compare_learning_rates(steep_quadratic, steep_quadratic_prime, 0.0, learning_rates)


def demonstrate_neural_network():
    """Demonstrate a simple neural network with backpropagation."""
    print("🧠 Neural Network Demonstration")
    print("=" * 40)
    
    # Create a simple network
    network = SimpleNeuralNetwork(0.5, -0.2, 0.3, 0.1)
    
    # Create XOR-like training data (linearly separable version)
    training_data = [
        (0.0, 0.1),
        (0.5, 0.4),
        (1.0, 0.9),
        (1.5, 0.6)
    ]
    
    print("Training simple neural network...")
    print("Data: Simple function approximation")
    
    # Train the network
    loss_history = network.train(training_data, epochs=100)
    
    # Test the trained network
    print("\nTesting trained network:")
    for x, target in training_data:
        prediction = network.forward(x)
        print(f"  Input: {x:.1f}, Target: {target:.1f}, Prediction: {prediction:.3f}, Error: {abs(prediction - target):.3f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot function approximation
    plt.subplot(1, 2, 2)
    x_test = np.linspace(-0.5, 2.0, 100)
    y_pred = [network.forward(x) for x in x_test]
    
    plt.plot(x_test, y_pred, 'r-', linewidth=2, label='Neural Network')
    x_train = [point[0] for point in training_data]
    y_train = [point[1] for point in training_data]
    plt.plot(x_train, y_train, 'bo', markersize=8, label='Training Data')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Function Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_linear_regression():
    """Demonstrate linear regression with gradient descent."""
    print("📊 Linear Regression with Gradient Descent")
    print("=" * 50)
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = [[1, x] for x in np.linspace(0, 10, 50)]  # [bias, feature]
    true_weights = [1, 2]  # [bias, slope]
    noise = np.random.normal(0, 0.5, 50)
    y = [true_weights[0] + true_weights[1] * x[1] + noise[i] for i, x in enumerate(X)]
    
    print(f"True relationship: y = {true_weights[1]}x + {true_weights[0]} + noise")
    print("Learning this relationship using gradient descent...")
    
    # Train using gradient descent
    weights, loss_history = linear_regression_gradient_descent(X, y, learning_rate=0.01, num_steps=1000)
    
    print(f"\nLearned weights: [{weights[0]:.3f}, {weights[1]:.3f}]")
    print(f"True weights: [{true_weights[0]:.3f}, {true_weights[1]:.3f}]")
    print(f"Error: [{abs(weights[0] - true_weights[0]):.3f}, {abs(weights[1] - true_weights[1]):.3f}]")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot data and learned line
    plt.subplot(1, 3, 1)
    x_vals = [point[1] for point in X]
    plt.scatter(x_vals, y, alpha=0.6, label='Data')
    
    x_line = np.linspace(0, 10, 100)
    y_line = [weights[0] + weights[1] * x for x in x_line]
    y_true = [true_weights[0] + true_weights[1] * x for x in x_line]
    
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Learned: y = {weights[1]:.2f}x + {weights[0]:.2f}')
    plt.plot(x_line, y_true, 'g--', linewidth=2, label=f'True: y = {true_weights[1]}x + {true_weights[0]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss convergence
    plt.subplot(1, 3, 2)
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Loss Convergence')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot residuals
    plt.subplot(1, 3, 3)
    predictions = [weights[0] + weights[1] * x[1] for x in X]
    residuals = [pred - actual for pred, actual in zip(predictions, y)]
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_optimization_landscape():
    """Demonstrate different optimization challenges."""
    print("🗻 Optimization Landscape Exploration")
    print("=" * 50)
    
    # Define different challenging functions
    functions = {
        "Simple Bowl": (
            lambda x: x[0]**2 + x[1]**2,
            "Easy convex function - should converge quickly"
        ),
        "Elongated Bowl": (
            lambda x: 10*x[0]**2 + x[1]**2,
            "Different scales - shows importance of learning rate"
        ),
        "Saddle Point": (
            lambda x: x[0]**2 - x[1]**2,
            "Has a saddle point at origin - challenging for optimization"
        ),
        "Rosenbrock": (
            lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
            "The classic optimization challenge - banana-shaped valley"
        )
    }
    
    start_point = [2.0, 2.0]
    learning_rate = 0.01
    
    plt.figure(figsize=(16, 12))
    
    for i, (name, (func, description)) in enumerate(functions.items()):
        plt.subplot(2, 2, i + 1)
        
        print(f"\nOptimizing {name}: {description}")
        
        try:
            final_point, history = gradient_descent_multi(func, start_point, learning_rate, 200)
            
            # Create contour plot
            x_coords = [p[0] for p in history]
            y_coords = [p[1] for p in history]
            
            x_min, x_max = min(x_coords + [start_point[0]]) - 1, max(x_coords + [start_point[0]]) + 1
            y_min, y_max = min(y_coords + [start_point[1]]) - 1, max(y_coords + [start_point[1]]) + 1
            
            x_range = np.linspace(x_min, x_max, 50)
            y_range = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.array([[func([x, y]) for x in x_range] for y in y_range])
            
            contour = plt.contour(X, Y, Z, levels=15, alpha=0.6)
            plt.clabel(contour, inline=True, fontsize=8)
            
            # Plot optimization path
            plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)
            plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
            plt.plot(final_point[0], final_point[1], 'ro', markersize=10, label='End')
            
            plt.title(f'{name}\nSteps: {len(history)}, Final: [{final_point[0]:.2f}, {final_point[1]:.2f}]')
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            print(f"  Converged to: [{final_point[0]:.3f}, {final_point[1]:.3f}] in {len(history)} steps")
            print(f"  Final function value: {func(final_point):.6f}")
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Optimization failed:\n{str(e)}', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            plt.title(f'{name} - Failed')
            print(f"  Optimization failed: {e}")
    
    plt.tight_layout()
    plt.show()


def comprehensive_calculus_demo():
    """Run a comprehensive demonstration of all calculus concepts."""
    print("🎓 NEURAL ODYSSEY - CALCULUS COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases how calculus powers machine learning!")
    print()
    
    # 1. Basic derivatives
    print("1️⃣  DERIVATIVES - The Foundation of Change")
    test_derivatives()
    
    # 2. Gradient descent basics
    print("\n2️⃣  GRADIENT DESCENT - The Learning Algorithm")
    test_gradient_descent()
    
    # 3. Learning rate effects
    print("\n3️⃣  LEARNING RATES - The Goldilocks Problem")
    demonstrate_learning_rates()
    
    # 4. Neural network training
    print("\n4️⃣  NEURAL NETWORKS - Backpropagation in Action")
    demonstrate_neural_network()
    
    # 5. Linear regression
    print("\n5️⃣  LINEAR REGRESSION - Statistics Meets Calculus")
    demonstrate_linear_regression()
    
    # 6. Optimization challenges
    print("\n6️⃣  OPTIMIZATION LANDSCAPES - The Real World")
    demonstrate_optimization_landscape()
    
    print("\n🎉 Calculus Demo Complete!")
    print("\nKey Takeaways:")
    print("• Derivatives measure how functions change")
    print("• Gradients point in the direction of steepest increase")
    print("• Gradient descent finds minima by following negative gradients")
    print("• Learning rate controls optimization speed and stability")
    print("• Chain rule enables backpropagation in neural networks")
    print("• Real optimization landscapes have many challenges")
    print("\n💡 Remember: Every time you call model.fit(), calculus is working behind the scenes!")


if __name__ == "__main__":
    """
    Run this file to explore calculus concepts and implementations!
    
    Complete the TODO functions above, then run:
    python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey Calculus!")
    print("Complete the TODO functions to unlock the power of calculus in ML.")
    print("\nTo get started:")
    print("1. Implement numerical_derivative() first")
    print("2. Move on to gradient_descent_1d()")
    print("3. Tackle multivariable functions")
    print("4. Build the neural network with backpropagation")
    print("5. Run the comprehensive demo!")
    
    # Uncomment these lines after implementing the functions:
    # comprehensive_calculus_demo()
    
    print("\n💡 Pro tip: Start with simple functions and visualize everything!")
    print("Understanding the geometry behind calculus will make ML intuitive!")
    
    # Example of what to expect when functions are implemented:
    print("\n🎯 Example Usage (implement functions first):")
    print(">>> numerical_derivative(lambda x: x**2, 3.0)  # Should return ~6.0")
    print(">>> gradient_descent_1d(lambda x: (x-2)**2, lambda x: 2*(x-2), 0.0)")
    print(">>> # Should find minimum at x ≈ 2.0")
    
    print("\n🎨 Visualization examples:")
    print("• plot_function_and_derivative() - See functions and their slopes")
    print("• plot_gradient_descent_1d() - Watch optimization in action")
    print("• plot_gradient_descent_2d() - Navigate 2D loss landscapes")
    print("• compare_learning_rates() - See why hyperparameters matter")