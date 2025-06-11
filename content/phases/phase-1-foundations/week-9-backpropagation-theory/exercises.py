def higher_order_derivatives(func, x, order=2):
    """
    TODO: Compute higher-order derivatives using autodiff
    
    Shows how to compute second derivatives, Hessians, etc.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute derivatives
        order: Order of derivative to compute
        
    Returns:
        Higher-order derivative information
    """
    # TODO: Use autodiff to compute higher-order derivatives
    # This requires differentiating the gradient function
    pass

def hessian_computation(func, x):
    """
    TODO: Compute Hessian matrix using automatic differentiation
    
    The Hessian contains all second-order partial derivatives.
    Important for second-order optimization methods.
    
    Args:
        func: Scalar function
        x: Input vector
        
    Returns:
        Hessian matrix
    """
    # TODO: Compute Hessian using forward-over-reverse or reverse-over-forward
    pass

def jacobian_vector_products(func, x, v):
    """
    TODO: Compute Jacobian-vector products efficiently
    
    JVP: J(x) * v using forward mode
    VJP: v^T * J(x) using reverse mode
    
    More efficient than computing full Jacobian when only products are needed.
    
    Args:
        func: Vector-valued function
        x: Input vector
        v: Vector to multiply with Jacobian
        
    Returns:
        JVP and VJP results
    """
    # TODO: Implement efficient Jacobian-vector products
    pass

def automatic_differentiation_optimizations():
    """
    TODO: Demonstrate autodiff optimizations
    
    Shows techniques used in production autodiff systems:
    - Operation fusion
    - Memory optimization
    - Sparse gradients
    
    Returns:
        Comparison of optimized vs naive implementations
    """
    # TODO: Implement and compare optimized autodiff techniques
    pass

def symbolic_vs_automatic_differentiation():
    """
    TODO: Compare symbolic and automatic differentiation
    
    Shows trade-offs between different differentiation approaches.
    
    Returns:
        Comparison analysis
    """
    # TODO: Compare symbolic differentiation (using sympy) with autodiff
    # Show accuracy, efficiency, and ease of use differences
    pass

# ==============================================
# PART 8: REAL-WORLD APPLICATIONS
# ==============================================

def physics_simulation_with_autodiff():
    """
    TODO: Use autodiff for physics simulation
    
    Shows how autodiff enables differentiable physics simulators.
    Example: Simple pendulum with learnable parameters.
    
    Returns:
        Differentiable physics simulation
    """
    # TODO: Implement simple physics simulation where parameters can be learned
    # Use autodiff to compute gradients w.r.t. physical parameters
    pass

def neural_ode_example():
    """
    TODO: Implement Neural Ordinary Differential Equations
    
    Shows how autodiff enables continuous-time neural networks.
    
    Returns:
        Neural ODE implementation and training
    """
    # TODO: Implement simple Neural ODE using autodiff
    # Show how to backpropagate through ODE solver
    pass

def meta_learning_with_autodiff():
    """
    TODO: Demonstrate meta-learning using higher-order gradients
    
    Shows how autodiff enables gradients through gradient descent steps.
    
    Returns:
        Simple meta-learning example
    """
    # TODO: Implement MAML-style meta-learning using higher-order gradients
    pass

def probabilistic_programming_autodiff():
    """
    TODO: Use autodiff for probabilistic programming
    
    Shows how autodiff enables gradient-based inference in probabilistic models.
    
    Returns:
        Probabilistic model with gradient-based inference
    """
    # TODO: Implement simple Bayesian model with variational inference using autodiff
    pass

# ==============================================
# PART 9: COMPREHENSIVE TESTING AND VALIDATION
# ==============================================

class AutoDiffTester:
    """
    TODO: Build comprehensive testing suite for autodiff systems
    
    Tests correctness, efficiency, and numerical stability.
    """
    
    def __init__(self):
        self.test_functions = []
        self.test_results = {}
    
    def add_test_function(self, name, func, analytical_grad):
        """
        TODO: Add test function with known analytical gradient
        """
        pass
    
    def test_correctness(self, autodiff_system, tolerance=1e-6):
        """
        TODO: Test autodiff correctness against analytical gradients
        """
        pass
    
    def test_efficiency(self, autodiff_system, problem_sizes):
        """
        TODO: Test computational efficiency for different problem sizes
        """
        pass
    
    def test_numerical_stability(self, autodiff_system, difficult_cases):
        """
        TODO: Test numerical stability on challenging cases
        """
        pass
    
    def generate_test_report(self):
        """
        TODO: Generate comprehensive test report
        """
        pass

def stress_test_autodiff():
    """
    TODO: Stress test autodiff system with challenging cases
    
    Tests edge cases, numerical stability, and performance limits.
    
    Returns:
        Stress test results
    """
    # TODO: Test autodiff on challenging functions:
    # - Nearly singular functions
    # - Functions with discontinuous derivatives
    # - Very high-dimensional problems
    pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_computational_graphs():
    """Demonstrate computational graph concepts."""
    
    print("🔗 Computational Graphs Demonstration")
    print("=" * 50)
    
    try:
        # Manual chain rule example
        result = manual_chain_rule_example()
        print(f"Manual chain rule computation completed")
        
        # Simple computation graph
        def example_function(x):
            return np.sin(x**2 + 1)
        
        def example_gradient(x):
            return np.cos(x**2 + 1) * 2 * x
        
        x_test = 2.0
        graph_root = build_computation_graph(example_function, {'x': x_test})
        
        if graph_root:
            print(f"✅ Computation graph built for f(x) = sin(x² + 1)")
            
            # Visualize graph
            visualize_computation_graph(graph_root)
            print("✅ Computation graph visualization created")
            
            # Compare with analytical
            comparison = compare_analytical_vs_computational(
                example_function, example_gradient, [1.0, 2.0, 3.0]
            )
            print("✅ Analytical vs computational comparison completed")
        else:
            print("⚠️  Implement build_computation_graph function")
            
    except Exception as e:
        print(f"❌ Computational graphs demo failed: {e}")
        print("Implement the computational graph functions!")

def demonstrate_autodiff_tensors():
    """Demonstrate automatic differentiation tensors."""
    
    print("\n🔢 AutoDiff Tensors Demonstration")
    print("=" * 50)
    
    try:
        # Create autodiff tensors
        x = AutoDiffTensor(2.0, requires_grad=True)
        y = AutoDiffTensor(3.0, requires_grad=True)
        
        print(f"Created tensors: x={x.data}, y={y.data}")
        
        # Test operations
        z = x + y
        w = x * y
        u = x ** 2
        
        if hasattr(z, 'data'):
            print(f"x + y = {z.data}")
            print(f"x * y = {w.data}")
            print(f"x² = {u.data}")
            
            # Test backward pass
            if hasattr(z, 'backward'):
                z.backward()
                print(f"✅ Backward pass completed")
                
                if x.grad is not None:
                    print(f"∂z/∂x = {x.grad}")
                    print(f"∂z/∂y = {y.grad}")
            else:
                print("⚠️  Implement backward method")
        else:
            print("⚠️  Implement AutoDiffTensor operations")
            
        # Test all operations
        test_result = test_autodiff_operations()
        if test_result:
            print("✅ All autodiff operations working correctly")
            
    except Exception as e:
        print(f"❌ AutoDiff tensors demo failed: {e}")
        print("Implement the AutoDiffTensor class!")

def demonstrate_forward_mode():
    """Demonstrate forward-mode automatic differentiation."""
    
    print("\n➡️ Forward-Mode AutoDiff Demonstration")
    print("=" * 50)
    
    try:
        # Test dual numbers
        x = DualNumber(2.0, 1.0)  # f(2) with derivative seed 1
        
        if hasattr(x, 'real') and hasattr(x, 'dual'):
            print(f"Dual number: {x.real} + {x.dual}ε")
            
            # Test dual number operations
            y = x + DualNumber(1.0, 0.0)
            z = x * x
            
            print(f"(x + 1): {y.real} + {y.dual}ε")
            print(f"x²: {z.real} + {z.dual}ε")
            
            # Test forward mode on function
            def test_func(x):
                return x**3 + 2*x**2 + x + 1
            
            result = forward_mode_autodiff(test_func, 2.0)
            if result:
                print(f"✅ Forward mode result: f(2) = {result[0]}, f'(2) = {result[1]}")
                
                # Compare with analytical
                analytical_value = 8 + 8 + 2 + 1  # 19
                analytical_deriv = 12 + 8 + 1  # 21
                print(f"Analytical: f(2) = {analytical_value}, f'(2) = {analytical_deriv}")
            
            # Test Jacobian computation
            def vector_func(x):
                return np.array([x[0]**2 + x[1], x[0] * x[1]])
            
            jacobian = compute_jacobian_forward_mode(vector_func, np.array([2.0, 3.0]))
            if jacobian is not None:
                print(f"✅ Jacobian computation completed")
                print(f"Jacobian shape: {jacobian.shape}")
        else:
            print("⚠️  Implement DualNumber class")
            
    except Exception as e:
        print(f"❌ Forward mode demo failed: {e}")
        print("Implement the forward-mode autodiff functions!")

def demonstrate_reverse_mode():
    """Demonstrate reverse-mode automatic differentiation."""
    
    print("\n⬅️ Reverse-Mode AutoDiff Demonstration")
    print("=" * 50)
    
    try:
        # Test reverse mode on simple function
        def test_func(variables):
            x, y = variables['x'], variables['y']
            return x**2 + x*y + y**3
        
        variables = {'x': 2.0, 'y': 1.0}
        result = reverse_mode_autodiff(test_func, variables)
        
        if result:
            func_value, gradients = result
            print(f"Function value: {func_value}")
            print(f"Gradients: {gradients}")
            
            # Analytical verification
            # f = x² + xy + y³
            # ∂f/∂x = 2x + y = 2(2) + 1 = 5
            # ∂f/∂y = x + 3y² = 2 + 3(1)² = 5
            print(f"Analytical gradients: ∂f/∂x = 5, ∂f/∂y = 5")
            
        # Test efficiency analysis
        input_dims = [10, 100, 1000]
        output_dims = [1, 10, 100]
        
        for in_dim in input_dims:
            for out_dim in output_dims:
                analysis = reverse_mode_efficiency_analysis(None, in_dim, out_dim)
                if analysis:
                    print(f"✅ Efficiency analysis: {in_dim}→{out_dim}")
                    
    except Exception as e:
        print(f"❌ Reverse mode demo failed: {e}")
        print("Implement the reverse-mode autodiff functions!")

def demonstrate_neural_network_backprop():
    """Demonstrate backpropagation in neural networks."""
    
    print("\n🧠 Neural Network Backpropagation")
    print("=" * 50)
    
    try:
        # Create simple neural network
        np.random.seed(42)
        X = np.random.randn(5, 3)
        y = np.random.randn(5, 1)
        
        # Simple 2-layer network
        weights = [np.random.randn(3, 4) * 0.1, np.random.randn(4, 1) * 0.1]
        biases = [np.random.randn(4) * 0.1, np.random.randn(1) * 0.1]
        activations = ['relu', 'linear']
        
        # Run detailed backpropagation
        result = neural_network_backprop_detailed(X, y, weights, biases, activations)
        
        if result:
            print("✅ Neural network backpropagation completed")
            forward_results, gradients = result
            print(f"Forward pass completed, {len(gradients)} gradient sets computed")
            
        # Analyze activation gradients
        activations_to_test = ['sigmoid', 'relu', 'tanh']
        input_range = np.linspace(-5, 5, 100)
        
        grad_analysis = activation_gradients_analysis(activations_to_test, input_range)
        if grad_analysis:
            print("✅ Activation gradients analysis completed")
            
        # Demonstrate gradient problems
        vanishing_demo = vanishing_gradient_demonstration()
        exploding_demo = exploding_gradient_demonstration()
        
        if vanishing_demo and exploding_demo:
            print("✅ Gradient flow problems demonstrated")
            
    except Exception as e:
        print(f"❌ Neural network backprop demo failed: {e}")
        print("Implement the neural network backpropagation functions!")

def comprehensive_autodiff_demo():
    """Comprehensive automatic differentiation demonstration."""
    
    print("\n🚀 Comprehensive AutoDiff System Demo")
    print("=" * 50)
    
    try:
        # Build complete autodiff system
        autodiff_system = build_simple_autograd()
        
        if autodiff_system:
            print("✅ AutoDiff system built successfully")
            
            # Test system with various functions
            x = autodiff_system.create_variable(2.0)
            y = autodiff_system.create_variable(3.0)
            
            """
Neural Odyssey - Week 9: Backpropagation Theory and Automatic Differentiation
Exercises for mastering the mathematical engine that enables neural network training

This module implements the core concepts behind automatic differentiation:
- Computational graphs and the chain rule
- Forward and reverse mode automatic differentiation
- Backpropagation as reverse-mode autodiff
- Building automatic differentiation systems from scratch
- Advanced differentiation techniques and optimizations

Complete the TODO functions to build your autodiff toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: COMPUTATIONAL GRAPHS AND CHAIN RULE
# ==============================================

class ComputationNode:
    """
    TODO: Implement a computational graph node
    
    Represents a single operation in a computational graph.
    Stores the operation, inputs, output, and gradient information.
    """
    
    def __init__(self, operation, inputs, output_value):
        """
        TODO: Initialize computation node
        
        Args:
            operation: String describing the operation
            inputs: List of input nodes
            output_value: The computed output value
        """
        # TODO: Store operation details and initialize gradient
        self.operation = operation
        self.inputs = inputs
        self.output_value = output_value
        self.gradient = 0.0  # Will accumulate gradients during backprop
        self.local_gradients = {}  # Gradients w.r.t. each input
        
    def backward(self, upstream_gradient):
        """
        TODO: Implement backward pass for this node
        
        Args:
            upstream_gradient: Gradient flowing from nodes above
        """
        # TODO: Accumulate gradient and propagate to inputs
        pass

def build_computation_graph(expression_func, variables):
    """
    TODO: Build computational graph for a mathematical expression
    
    Takes a function and variables, builds the computational graph
    by tracing through the operations.
    
    Args:
        expression_func: Function to trace
        variables: Dictionary of variable names to values
        
    Returns:
        Root node of computational graph
    """
    # TODO: Trace through function execution and build graph
    # This is a simplified version - real autodiff is more complex
    pass

def visualize_computation_graph(root_node):
    """
    TODO: Visualize the computational graph
    
    Creates a visual representation of the computation graph
    showing operations and data flow.
    
    Args:
        root_node: Root of the computational graph
    """
    # TODO: Create graph visualization using matplotlib or graphviz
    # Show nodes (operations) and edges (data flow)
    pass

def manual_chain_rule_example():
    """
    TODO: Demonstrate chain rule computation manually
    
    Shows step-by-step application of chain rule for a complex function.
    Example: f(x) = sin(x^2 + 1) where x = 2
    
    Returns:
        Manual computation of derivative
    """
    # TODO: Compute derivative manually using chain rule
    # f(x) = sin(u) where u = x^2 + 1
    # df/dx = df/du * du/dx = cos(u) * 2x
    
    pass

def compare_analytical_vs_computational(func, grad_func, x_values):
    """
    TODO: Compare analytical gradients with computational graph gradients
    
    Verifies that computational graph produces correct gradients.
    
    Args:
        func: Function to differentiate
        grad_func: Analytical gradient function
        x_values: Points to test
        
    Returns:
        Comparison results
    """
    # TODO: Compute gradients both ways and compare
    pass

# ==============================================
# PART 2: AUTOMATIC DIFFERENTIATION PRIMITIVES
# ==============================================

class AutoDiffTensor:
    """
    TODO: Implement automatic differentiation tensor
    
    A tensor that tracks operations for automatic differentiation.
    Similar to PyTorch tensors with requires_grad=True.
    """
    
    def __init__(self, data, requires_grad=False, operation=None, inputs=None):
        """
        TODO: Initialize autodiff tensor
        
        Args:
            data: Numerical data (scalar or array)
            requires_grad: Whether to track gradients
            operation: Operation that created this tensor
            inputs: Input tensors for this operation
        """
        # TODO: Initialize tensor with gradient tracking
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.operation = operation
        self.inputs = inputs or []
        
        # Initialize gradient if needed
        if requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self, upstream_grad=None):
        """
        TODO: Implement backward pass for tensor
        
        Args:
            upstream_grad: Gradient from upstream operations
        """
        # TODO: Accumulate gradients and propagate to inputs
        pass
    
    def __add__(self, other):
        """TODO: Implement addition with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass
    
    def __mul__(self, other):
        """TODO: Implement multiplication with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass
    
    def __pow__(self, exponent):
        """TODO: Implement power operation with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass
    
    def sin(self):
        """TODO: Implement sine function with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass
    
    def exp(self):
        """TODO: Implement exponential function with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass
    
    def log(self):
        """TODO: Implement natural logarithm with gradient tracking"""
        # TODO: Implement forward pass and set up backward pass
        pass

def test_autodiff_operations():
    """
    TODO: Test all autodiff operations
    
    Verifies that each operation correctly computes gradients.
    """
    # TODO: Test each operation against analytical derivatives
    pass

# ==============================================
# PART 3: FORWARD MODE AUTOMATIC DIFFERENTIATION
# ==============================================

class DualNumber:
    """
    TODO: Implement dual numbers for forward-mode autodiff
    
    Dual numbers are numbers of the form a + b*ε where ε² = 0.
    They naturally compute derivatives through the algebra.
    """
    
    def __init__(self, real, dual=0.0):
        """
        TODO: Initialize dual number
        
        Args:
            real: Real part (function value)
            dual: Dual part (derivative value)
        """
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        """TODO: Implement dual number addition"""
        # (a + b*ε) + (c + d*ε) = (a + c) + (b + d)*ε
        pass
    
    def __mul__(self, other):
        """TODO: Implement dual number multiplication"""
        # (a + b*ε) * (c + d*ε) = ac + (ad + bc)*ε
        pass
    
    def __pow__(self, n):
        """TODO: Implement dual number power"""
        # (a + b*ε)^n = a^n + n*a^(n-1)*b*ε
        pass
    
    def sin(self):
        """TODO: Implement sine for dual numbers"""
        # sin(a + b*ε) = sin(a) + cos(a)*b*ε
        pass
    
    def exp(self):
        """TODO: Implement exponential for dual numbers"""
        # exp(a + b*ε) = exp(a) + exp(a)*b*ε
        pass

def forward_mode_autodiff(func, x, direction=1.0):
    """
    TODO: Implement forward-mode automatic differentiation
    
    Computes the directional derivative of func at x in the given direction.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute derivative
        direction: Direction vector for directional derivative
        
    Returns:
        Function value and directional derivative
    """
    # TODO: Use dual numbers to compute derivative
    # Replace input with dual number and evaluate function
    pass

def compute_jacobian_forward_mode(func, x):
    """
    TODO: Compute Jacobian matrix using forward mode
    
    For vector-valued functions, computes the full Jacobian matrix.
    
    Args:
        func: Vector-valued function
        x: Input vector
        
    Returns:
        Jacobian matrix
    """
    # TODO: Compute Jacobian by running forward mode for each input dimension
    pass

def forward_mode_efficiency_analysis(func, input_dims, output_dims):
    """
    TODO: Analyze efficiency of forward mode autodiff
    
    Forward mode is efficient when input dimension << output dimension.
    
    Args:
        func: Function to analyze
        input_dims: Number of input dimensions
        output_dims: Number of output dimensions
        
    Returns:
        Efficiency analysis
    """
    # TODO: Compare computational cost of forward mode vs finite differences
    pass

# ==============================================
# PART 4: REVERSE MODE AUTOMATIC DIFFERENTIATION
# ==============================================

class ReverseNode:
    """
    TODO: Implement node for reverse-mode autodiff
    
    Stores operation and gradient computation for reverse pass.
    """
    
    def __init__(self, value, local_gradients=None):
        """
        TODO: Initialize reverse mode node
        
        Args:
            value: Forward pass value
            local_gradients: Function to compute local gradients
        """
        self.value = value
        self.gradient = 0.0
        self.local_gradients = local_gradients or (lambda: {})
        self.children = []
    
    def add_child(self, child, local_grad_func):
        """
        TODO: Add child node with local gradient function
        """
        pass
    
    def backward(self):
        """
        TODO: Implement reverse pass
        """
        pass

def reverse_mode_autodiff(func, variables):
    """
    TODO: Implement reverse-mode automatic differentiation
    
    This is the algorithm used by modern deep learning frameworks.
    Efficient when input dimension >> output dimension.
    
    Args:
        func: Function to differentiate
        variables: Dictionary of variable names to values
        
    Returns:
        Function value and gradients w.r.t. all variables
    """
    # TODO: Build computation graph and run reverse pass
    pass

def compute_gradient_reverse_mode(func, x):
    """
    TODO: Compute gradient using reverse mode
    
    Args:
        func: Scalar-valued function
        x: Input vector
        
    Returns:
        Gradient vector
    """
    # TODO: Use reverse mode to compute gradient efficiently
    pass

def reverse_mode_efficiency_analysis(func, input_dims, output_dims):
    """
    TODO: Analyze efficiency of reverse mode autodiff
    
    Reverse mode is efficient when output dimension << input dimension.
    
    Args:
        func: Function to analyze
        input_dims: Number of input dimensions
        output_dims: Number of output dimensions
        
    Returns:
        Efficiency analysis
    """
    # TODO: Compare computational cost vs forward mode and finite differences
    pass

# ==============================================
# PART 5: BACKPROPAGATION IN NEURAL NETWORKS
# ==============================================

def neural_network_backprop_detailed(X, y, weights, biases, activations):
    """
    TODO: Implement detailed backpropagation for neural networks
    
    Shows step-by-step application of chain rule through network layers.
    
    Args:
        X: Input data
        y: Target values
        weights: List of weight matrices
        biases: List of bias vectors
        activations: List of activation functions
        
    Returns:
        Forward pass results and gradients
    """
    # TODO: Implement detailed forward and backward passes
    # Show how chain rule propagates through each layer
    pass

def activation_gradients_analysis(activation_functions, input_range):
    """
    TODO: Analyze gradients of different activation functions
    
    Shows how different activations affect gradient flow.
    
    Args:
        activation_functions: List of activation functions to analyze
        input_range: Range of inputs to test
        
    Returns:
        Analysis of gradient behavior
    """
    # TODO: Compute and visualize gradients for different activations
    # Show vanishing/exploding gradient problems
    pass

def gradient_flow_visualization(network_architecture, input_data):
    """
    TODO: Visualize gradient flow through neural network
    
    Shows how gradients propagate from output back to input.
    
    Args:
        network_architecture: Network structure
        input_data: Sample input
        
    Returns:
        Visualization of gradient magnitudes through layers
    """
    # TODO: Create visualization showing gradient flow
    pass

def vanishing_gradient_demonstration():
    """
    TODO: Demonstrate vanishing gradient problem
    
    Shows how gradients can vanish in deep networks with certain activations.
    
    Returns:
        Analysis of gradient vanishing in deep networks
    """
    # TODO: Create deep network and show gradient magnitudes at different depths
    pass

def exploding_gradient_demonstration():
    """
    TODO: Demonstrate exploding gradient problem
    
    Shows how gradients can explode with poor initialization or activations.
    
    Returns:
        Analysis of gradient explosion
    """
    # TODO: Create network prone to exploding gradients and demonstrate the problem
    pass

# ==============================================
# PART 6: AUTOMATIC DIFFERENTIATION SYSTEM
# ==============================================

class AutoDiffSystem:
    """
    TODO: Build complete automatic differentiation system
    
    A minimal but complete autodiff system similar to PyTorch/TensorFlow.
    Supports both forward and reverse mode differentiation.
    """
    
    def __init__(self):
        """TODO: Initialize autodiff system"""
        self.operations = {}
        self.gradient_functions = {}
        
    def register_operation(self, name, forward_func, backward_func):
        """
        TODO: Register new operation with the system
        
        Args:
            name: Operation name
            forward_func: Forward computation function
            backward_func: Backward gradient computation function
        """
        pass
    
    def create_variable(self, value, requires_grad=True):
        """
        TODO: Create a differentiable variable
        
        Args:
            value: Initial value
            requires_grad: Whether to track gradients
            
        Returns:
            Variable tensor
        """
        pass
    
    def compute_gradients(self, loss, variables):
        """
        TODO: Compute gradients of loss w.r.t. variables
        
        Args:
            loss: Loss tensor (scalar)
            variables: List of variables to compute gradients for
            
        Returns:
            Dictionary of variable -> gradient
        """
        pass
    
    def gradient_check(self, func, variables, epsilon=1e-5):
        """
        TODO: Verify gradients using finite differences
        
        Args:
            func: Function to check
            variables: Variables to check gradients for
            epsilon: Finite difference step size
            
        Returns:
            Boolean indicating if gradients are correct
        """
        pass

def build_simple_autograd():
    """
    TODO: Build simple autograd system from scratch
    
    Demonstrates the core concepts behind PyTorch's autograd.
    
    Returns:
        Working autograd system
    """
    # TODO: Create basic autograd with essential operations
    pass

def compare_autodiff_modes(func, input_sizes, output_sizes):
    """
    TODO: Compare forward vs reverse mode efficiency
    
    Shows when to use each mode based on problem characteristics.
    
    Args:
        func: Function to differentiate
        input_sizes: Different input dimensionalities to test
        output_sizes: Different output dimensionalities to test
        
    Returns:
        Efficiency comparison
    """
    # TODO: Benchmark both modes for different problem sizes
    pass

# ==============================================
# PART 7: ADVANCED DIFFERENTIATION TECHNIQUES
# ==============================================

def higher_order_derivatives(func, x, order=2):
    """
    TODO: Compute higher-order derivatives using autodiff