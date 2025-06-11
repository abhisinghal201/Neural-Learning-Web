"""
Neural Odyssey - Week 5: Optimization Theory and Gradient Methods
Exercises for mastering the mathematical engine that powers all machine learning

This module implements core concepts that enable machines to learn:
- Optimization fundamentals: gradients, convexity, and convergence
- Gradient descent variants: batch, stochastic, mini-batch
- Advanced optimizers: momentum, adaptive methods, second-order
- Non-convex optimization and neural network training

Complete the TODO functions to build your optimization toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification, load_boston
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: OPTIMIZATION FUNDAMENTALS
# ==============================================

def compute_gradient_numerical(f, x, h=1e-8):
    """
    TODO: Compute gradient using finite differences
    
    This is the fundamental way to compute gradients numerically.
    Understanding this helps you debug gradient implementations.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute gradient
        h: Small step size for finite differences
        
    Returns:
        Gradient vector at point x
    """
    # TODO: Implement numerical gradient computation
    # Use central differences: (f(x+h) - f(x-h)) / (2h)
    # Handle scalar and vector inputs appropriately
    
    pass

def check_gradient(f, grad_f, x, tolerance=1e-5):
    """
    TODO: Verify analytical gradient against numerical gradient
    
    Essential for debugging gradient implementations.
    Every ML practitioner should know how to do this!
    
    Args:
        f: Function
        grad_f: Analytical gradient function
        x: Point to check
        tolerance: Acceptable difference
        
    Returns:
        Boolean indicating if gradients match within tolerance
    """
    # TODO: Compare analytical vs numerical gradients
    # Compute relative error: |analytical - numerical| / max(|analytical|, |numerical|)
    
    pass

def visualize_function_landscape(f, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    """
    TODO: Visualize optimization landscape for 2D functions
    
    Helps build intuition about optimization challenges.
    
    Args:
        f: Function to visualize (should take 2D input)
        x_range, y_range: Ranges for visualization
        resolution: Number of points per dimension
    """
    # TODO: Create 3D surface plot and 2D contour plot
    # Show function values and gradient directions
    
    pass

def analyze_convexity(f, grad_f, hess_f, x_test_points):
    """
    TODO: Analyze function convexity properties
    
    Understanding convexity is crucial for optimization theory.
    
    Args:
        f: Function
        grad_f: Gradient function
        hess_f: Hessian function  
        x_test_points: Points to test convexity
        
    Returns:
        Dictionary with convexity analysis results
    """
    # TODO: Check convexity conditions:
    # 1. Hessian positive semidefinite everywhere
    # 2. f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for test points
    # 3. First-order condition: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
    
    pass

# ==============================================
# PART 2: GRADIENT DESCENT IMPLEMENTATIONS
# ==============================================

def gradient_descent_basic(f, grad_f, x0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    TODO: Implement basic gradient descent algorithm
    
    This is the foundation of all machine learning optimization.
    
    Args:
        f: Objective function to minimize
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Step size
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary with optimization results and trajectory
    """
    # TODO: Implement gradient descent:
    # 1. x_{t+1} = x_t - α * ∇f(x_t)
    # 2. Track function values and gradients
    # 3. Check convergence based on gradient norm
    # 4. Return trajectory for analysis
    
    pass

def stochastic_gradient_descent(grad_f_single, x0, data, learning_rate=0.01, max_epochs=100, batch_size=1):
    """
    TODO: Implement stochastic gradient descent
    
    SGD is the workhorse of modern machine learning.
    Handles large datasets by using mini-batches.
    
    Args:
        grad_f_single: Function that computes gradient for single sample
        x0: Initial parameters
        data: Dataset (list of samples)
        learning_rate: Step size
        max_epochs: Number of passes through data
        batch_size: Size of mini-batches
        
    Returns:
        Dictionary with optimization results
    """
    # TODO: Implement SGD:
    # 1. Shuffle data each epoch
    # 2. Process in mini-batches
    # 3. Average gradients within each batch
    # 4. Update parameters: x_{t+1} = x_t - α * (average gradient)
    
    pass

def gradient_descent_with_line_search(f, grad_f, x0, max_iterations=1000):
    """
    TODO: Implement gradient descent with line search
    
    Automatically finds good step sizes instead of using fixed learning rate.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results with adaptive step sizes
    """
    # TODO: Implement with Armijo line search:
    # 1. Start with step size α = 1
    # 2. Check Armijo condition: f(x - α∇f) ≤ f(x) - c₁α‖∇f‖²
    # 3. Backtrack (α = βα) until condition satisfied
    # 4. Use c₁ = 1e-4, β = 0.5 as standard values
    
    pass

def compare_learning_rates(f, grad_f, x0, learning_rates=[0.001, 0.01, 0.1, 1.0]):
    """
    TODO: Compare gradient descent with different learning rates
    
    Shows the crucial importance of learning rate selection.
    
    Args:
        f, grad_f: Function and gradient
        x0: Starting point
        learning_rates: List of learning rates to test
        
    Returns:
        Dictionary with results for each learning rate
    """
    # TODO: Run gradient descent with each learning rate
    # Compare convergence speed, final values, and stability
    # Create visualization showing different trajectories
    
    pass

# ==============================================
# PART 3: ADVANCED OPTIMIZERS
# ==============================================

def momentum_optimizer(f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iterations=1000):
    """
    TODO: Implement momentum-based gradient descent
    
    Momentum accelerates convergence and helps escape poor local minima.
    
    Args:
        f, grad_f: Function and gradient
        x0: Initial point
        learning_rate: Learning rate
        momentum: Momentum coefficient (typically 0.9)
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results with momentum
    """
    # TODO: Implement momentum:
    # v_t = γ * v_{t-1} + α * ∇f(x_t)
    # x_{t+1} = x_t - v_t
    # Track velocity and show acceleration effects
    
    pass

def nesterov_momentum(f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iterations=1000):
    """
    TODO: Implement Nesterov accelerated gradient
    
    Nesterov momentum looks ahead before computing gradients.
    Often converges faster than standard momentum.
    
    Args:
        f, grad_f: Function and gradient
        x0: Initial point
        learning_rate: Learning rate
        momentum: Momentum coefficient
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results with Nesterov acceleration
    """
    # TODO: Implement Nesterov momentum:
    # v_t = γ * v_{t-1} + α * ∇f(x_t - γ * v_{t-1})
    # x_{t+1} = x_t - v_t
    # The key difference: gradient computed at "look-ahead" point
    
    pass

def adagrad_optimizer(grad_f_stochastic, x0, data, learning_rate=0.01, epsilon=1e-8, max_epochs=100):
    """
    TODO: Implement AdaGrad optimizer
    
    AdaGrad adapts learning rate based on historical gradients.
    Good for sparse features but can stop learning too early.
    
    Args:
        grad_f_stochastic: Stochastic gradient function
        x0: Initial parameters
        data: Training data
        learning_rate: Base learning rate
        epsilon: Small constant for numerical stability
        max_epochs: Number of epochs
        
    Returns:
        Optimization results with adaptive learning rates
    """
    # TODO: Implement AdaGrad:
    # G_t = G_{t-1} + g_t ⊙ g_t  (accumulate squared gradients)
    # x_{t+1} = x_t - (α / √(G_t + ε)) ⊙ g_t
    # Track how learning rates adapt for each parameter
    
    pass

def rmsprop_optimizer(grad_f_stochastic, x0, data, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, max_epochs=100):
    """
    TODO: Implement RMSprop optimizer
    
    RMSprop fixes AdaGrad's diminishing learning rate problem
    by using exponential moving average of squared gradients.
    
    Args:
        grad_f_stochastic: Stochastic gradient function
        x0: Initial parameters
        data: Training data
        learning_rate: Base learning rate
        decay_rate: Exponential decay rate (typically 0.9)
        epsilon: Small constant for stability
        max_epochs: Number of epochs
        
    Returns:
        Optimization results
    """
    # TODO: Implement RMSprop:
    # v_t = γ * v_{t-1} + (1-γ) * g_t²
    # x_{t+1} = x_t - (α / √(v_t + ε)) * g_t
    # Compare with AdaGrad to show improvement
    
    pass

def adam_optimizer(grad_f_stochastic, x0, data, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=100):
    """
    TODO: Implement Adam optimizer
    
    Adam combines momentum with adaptive learning rates.
    Currently the most popular optimizer for deep learning.
    
    Args:
        grad_f_stochastic: Stochastic gradient function
        x0: Initial parameters
        data: Training data
        learning_rate: Base learning rate
        beta1: Exponential decay for first moment (momentum)
        beta2: Exponential decay for second moment (RMSprop)
        epsilon: Small constant for stability
        max_epochs: Number of epochs
        
    Returns:
        Optimization results
    """
    # TODO: Implement Adam:
    # m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    # v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
    # m̂_t = m_t / (1 - β₁ᵗ)  (bias correction)
    # v̂_t = v_t / (1 - β₂ᵗ)  (bias correction)
    # x_{t+1} = x_t - α * m̂_t / (√v̂_t + ε)
    
    pass

def optimizer_comparison(f, grad_f, x0, optimizers_to_test=None):
    """
    TODO: Compare different optimizers on the same problem
    
    Shows strengths and weaknesses of different optimization methods.
    
    Args:
        f, grad_f: Function and gradient
        x0: Starting point
        optimizers_to_test: List of optimizer functions to compare
        
    Returns:
        Comparison results and visualizations
    """
    # TODO: Run multiple optimizers and compare:
    # 1. Convergence speed
    # 2. Final objective value
    # 3. Stability and robustness
    # 4. Create side-by-side trajectory plots
    
    pass

# ==============================================
# PART 4: SECOND-ORDER METHODS
# ==============================================

def newton_method(f, grad_f, hess_f, x0, max_iterations=100, tolerance=1e-6):
    """
    TODO: Implement Newton's method for optimization
    
    Uses second-order information (Hessian) for faster convergence.
    Quadratic convergence near the optimum but expensive per iteration.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        hess_f: Hessian function
        x0: Initial point
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Optimization results
    """
    # TODO: Implement Newton's method:
    # x_{t+1} = x_t - H⁻¹(x_t) * ∇f(x_t)
    # where H is the Hessian matrix
    # Handle case where Hessian is not positive definite
    
    pass

def quasi_newton_bfgs(f, grad_f, x0, max_iterations=100, tolerance=1e-6):
    """
    TODO: Implement BFGS quasi-Newton method
    
    Approximates Hessian using gradient information.
    Faster than Newton (no Hessian computation) but still superlinear convergence.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Optimization results
    """
    # TODO: Implement BFGS:
    # 1. Maintain approximate inverse Hessian B
    # 2. Update B using BFGS formula after each step
    # 3. Use B to compute Newton-like steps
    # This is quite complex - focus on understanding the algorithm
    
    pass

def compare_first_vs_second_order(f, grad_f, hess_f, x0):
    """
    TODO: Compare first-order vs second-order methods
    
    Shows trade-offs between computation per iteration vs convergence speed.
    
    Args:
        f, grad_f, hess_f: Function, gradient, and Hessian
        x0: Starting point
        
    Returns:
        Comparison of different method classes
    """
    # TODO: Compare:
    # 1. Gradient descent (first-order)
    # 2. Newton's method (second-order)
    # 3. BFGS (quasi-second-order)
    # Analyze convergence rate vs computational cost
    
    pass

# ==============================================
# PART 5: MACHINE LEARNING APPLICATIONS
# ==============================================

def linear_regression_optimization(X, y, method='gradient_descent', learning_rate=0.01, max_iterations=1000):
    """
    TODO: Solve linear regression using various optimization methods
    
    Linear regression is convex, so we can compare optimizers on a well-understood problem.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Optimization method to use
        learning_rate: Learning rate (if applicable)
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results and learned parameters
    """
    # TODO: Implement linear regression optimization:
    # 1. Define loss function: L(w) = (1/2n) * ||Xw - y||²
    # 2. Compute gradient: ∇L(w) = (1/n) * Xᵀ(Xw - y)
    # 3. Apply chosen optimization method
    # 4. Compare with analytical solution: w* = (XᵀX)⁻¹Xᵀy
    
    pass

def logistic_regression_optimization(X, y, method='gradient_descent', learning_rate=0.01, max_iterations=1000):
    """
    TODO: Solve logistic regression using optimization
    
    Logistic regression is convex but has no closed-form solution,
    making it perfect for comparing iterative optimizers.
    
    Args:
        X: Feature matrix
        y: Binary target vector
        method: Optimization method
        learning_rate: Learning rate
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results and learned parameters
    """
    # TODO: Implement logistic regression optimization:
    # 1. Define sigmoid: σ(z) = 1/(1 + exp(-z))
    # 2. Define loss: L(w) = -(1/n) * Σ[y*log(σ(Xw)) + (1-y)*log(1-σ(Xw))]
    # 3. Compute gradient: ∇L(w) = (1/n) * Xᵀ(σ(Xw) - y)
    # 4. Apply optimization method
    
    pass

def neural_network_optimization(X, y, hidden_size=10, method='adam', learning_rate=0.001, max_epochs=100):
    """
    TODO: Train a simple neural network with different optimizers
    
    Neural networks are non-convex, showing the challenge of modern ML optimization.
    
    Args:
        X: Input features
        y: Targets
        hidden_size: Number of hidden units
        method: Optimization method
        learning_rate: Learning rate
        max_epochs: Number of training epochs
        
    Returns:
        Training results and learned network
    """
    # TODO: Implement neural network training:
    # 1. Initialize weights randomly
    # 2. Define forward pass with sigmoid activation
    # 3. Implement backpropagation for gradient computation
    # 4. Apply chosen optimizer
    # 5. Track training loss and convergence
    
    pass

def hyperparameter_optimization_demo(X, y, param_ranges):
    """
    TODO: Demonstrate hyperparameter optimization
    
    Shows how optimization principles apply to tuning ML algorithms.
    
    Args:
        X, y: Dataset
        param_ranges: Dictionary of parameter ranges to search
        
    Returns:
        Best hyperparameters and optimization trajectory
    """
    # TODO: Implement hyperparameter optimization:
    # 1. Define objective function (validation error)
    # 2. Use grid search, random search, or Bayesian optimization
    # 3. Apply cross-validation for robust evaluation
    # 4. Track and visualize the search process
    
    pass

# ==============================================
# PART 6: OPTIMIZATION DIAGNOSTICS
# ==============================================

class OptimizationDiagnostics:
    """
    TODO: Build comprehensive optimization analysis toolkit
    
    This class should provide all tools needed for diagnosing
    optimization problems in ML:
    - Convergence analysis
    - Learning rate selection
    - Optimizer comparison
    - Gradient checking
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_convergence(self, optimization_results):
        """
        TODO: Analyze convergence properties
        
        Returns:
            Dictionary with convergence analysis
        """
        # TODO: Analyze:
        # 1. Convergence rate (linear, superlinear, quadratic)
        # 2. Final gradient norm
        # 3. Function value improvement
        # 4. Stability of optimization
        
        pass
    
    def learning_rate_analysis(self, f, grad_f, x0, lr_range=(1e-4, 1e0), num_points=20):
        """
        TODO: Analyze learning rate sensitivity
        
        Helps choose appropriate learning rates.
        """
        # TODO: Test range of learning rates and analyze:
        # 1. Convergence speed vs learning rate
        # 2. Stability vs learning rate
        # 3. Final objective value vs learning rate
        
        pass
    
    def gradient_norm_analysis(self, optimization_trajectory):
        """
        TODO: Analyze gradient norms during optimization
        
        Helps diagnose optimization problems.
        """
        # TODO: Analyze gradient norm evolution:
        # 1. Gradient explosion or vanishing
        # 2. Convergence indicators
        # 3. Optimization phase identification
        
        pass
    
    def loss_landscape_analysis(self, f, grad_f, optimum, radius=1.0):
        """
        TODO: Analyze local loss landscape around optimum
        
        Reveals optimization challenges and curvature.
        """
        # TODO: Analyze landscape properties:
        # 1. Curvature and conditioning
        # 2. Local minima vs saddle points
        # 3. Optimization difficulty indicators
        
        pass
    
    def generate_optimization_report(self, optimization_results):
        """
        TODO: Generate comprehensive optimization analysis report
        """
        # TODO: Create detailed analysis including:
        # 1. Convergence assessment
        # 2. Efficiency analysis
        # 3. Robustness evaluation
        # 4. Recommendations for improvement
        
        pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_optimization_fundamentals():
    """Demonstrate fundamental optimization concepts."""
    
    print("🎯 Optimization Fundamentals Demonstration")
    print("=" * 50)
    
    # Define test functions
    def quadratic_2d(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
    
    def quadratic_grad(x):
        return np.array([2*x[0] + x[1], 4*x[1] + x[0]])
    
    try:
        # Test gradient computation
        x_test = np.array([1.0, 2.0])
        analytical_grad = quadratic_grad(x_test)
        numerical_grad = compute_gradient_numerical(quadratic_2d, x_test)
        
        grad_check = check_gradient(quadratic_2d, quadratic_grad, x_test)
        print(f"Gradient check passed: {grad_check}")
        print(f"Analytical gradient: {analytical_grad}")
        print(f"Numerical gradient: {numerical_grad}")
        
        # Visualize function landscape
        visualize_function_landscape(quadratic_2d)
        print("✅ Function landscape visualization created")
        
    except Exception as e:
        print(f"❌ Fundamentals demo failed: {e}")
        print("Implement the gradient computation functions!")

def demonstrate_gradient_descent():
    """Demonstrate gradient descent variants."""
    
    print("\n⬇️ Gradient Descent Demonstration")
    print("=" * 50)
    
    # Rosenbrock function - classic optimization test case
    def rosenbrock(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def rosenbrock_grad(x):
        dfdx0 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
        dfdx1 = 200*(x[1] - x[0]**2)
        return np.array([dfdx0, dfdx1])
    
    try:
        x0 = np.array([-1.0, 1.0])
        
        # Test basic gradient descent
        result = gradient_descent_basic(rosenbrock, rosenbrock_grad, x0, learning_rate=0.001)
        print(f"Basic GD converged to: {result['x_final']}")
        print(f"Iterations: {result['iterations']}")
        
        # Compare learning rates
        lr_comparison = compare_learning_rates(rosenbrock, rosenbrock_grad, x0)
        print("✅ Learning rate comparison completed")
        
        # Test line search
        ls_result = gradient_descent_with_line_search(rosenbrock, rosenbrock_grad, x0)
        print(f"Line search GD converged to: {ls_result['x_final']}")
        
    except Exception as e:
        print(f"❌ Gradient descent demo failed: {e}")
        print("Implement the gradient descent functions!")

def demonstrate_advanced_optimizers():
    """Demonstrate advanced optimization methods."""
    
    print("\n🚀 Advanced Optimizers Demonstration")
    print("=" * 50)
    
    # Create synthetic dataset for stochastic optimization
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    true_weights = np.array([1, -2, 0.5, 3, -1])
    y = X @ true_weights + 0.1 * np.random.randn(1000)
    data = list(zip(X, y))
    
    def linear_grad_single(w, sample):
        x_i, y_i = sample
        pred = np.dot(w, x_i)
        return x_i * (pred - y_i)
    
    try:
        w0 = np.random.randn(5)
        
        # Test momentum
        momentum_result = momentum_optimizer(
            lambda w: 0.5 * np.mean([(np.dot(w, x) - y)**2 for x, y in data]),
            lambda w: np.mean([linear_grad_single(w, sample) for sample in data]),
            w0
        )
        print(f"Momentum converged to: {momentum_result['x_final']}")
        
        # Test Adam
        adam_result = adam_optimizer(linear_grad_single, w0, data)
        print(f"Adam converged to: {adam_result['x_final']}")
        print(f"True weights: {true_weights}")
        
        # Compare optimizers
        comparison = optimizer_comparison(
            lambda w: 0.5 * np.mean([(np.dot(w, x) - y)**2 for x, y in data]),
            lambda w: np.mean([linear_grad_single(w, sample) for sample in data]),
            w0
        )
        print("✅ Optimizer comparison completed")
        
    except Exception as e:
        print(f"❌ Advanced optimizers demo failed: {e}")
        print("Implement the advanced optimizer functions!")

def demonstrate_ml_applications():
    """Demonstrate optimization in ML applications."""
    
    print("\n🤖 Machine Learning Applications")
    print("=" * 50)
    
    # Generate classification dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0, 
                             n_informative=5, random_state=42)
    
    try:
        # Linear regression optimization
        lr_result = linear_regression_optimization(X, y, method='gradient_descent')
        print(f"Linear regression optimization completed")
        print(f"Final weights: {lr_result['weights']}")
        
        # Logistic regression optimization
        y_binary = (y > 0).astype(int)
        logistic_result = logistic_regression_optimization(X, y_binary, method='adam')
        print(f"Logistic regression optimization completed")
        
        # Neural network optimization
        nn_result = neural_network_optimization(X, y_binary, method='adam')
        print(f"Neural network training completed")
        print(f"Final training loss: {nn_result['final_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ ML applications demo failed: {e}")
        print("Implement the ML optimization functions!")

def comprehensive_optimization_analysis():
    """Run comprehensive optimization analysis."""
    
    print("\n🔍 Comprehensive Optimization Analysis")
    print("=" * 50)
    
    try:
        # Create diagnostics toolkit
        diagnostics = OptimizationDiagnostics()
        
        # Analyze different optimization scenarios
        print("✅ Optimization diagnostics toolkit created")
        
        # Run comprehensive analysis
        analysis_report = diagnostics.generate_optimization_report({})
        print("✅ Optimization analysis report generated")
        
        print("\n🎉 Congratulations! You've built a complete optimization toolkit!")
        print("You now understand the mathematical engine that powers all machine learning.")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("Implement the OptimizationDiagnostics class!")

if __name__ == "__main__":
    """
    Run this file to explore optimization theory and gradient methods!
    
    Complete the TODO functions above, then run:
    python week5_exercises.py
    """
    
    print("⚡ Welcome to Neural Odyssey Week 5: Optimization Theory!")
    print("Complete the TODO functions to master the engine of machine learning.")
    print("\nTo get started:")
    print("1. Implement gradient computation and verification tools")
    print("2. Build gradient descent from scratch")
    print("3. Create advanced optimizers (momentum, Adam, etc.)")
    print("4. Apply to real ML problems (regression, neural networks)")
    print("5. Build comprehensive optimization diagnostics")
    
    # Uncomment these lines after implementing the functions:
    # demonstrate_optimization_fundamentals()
    # demonstrate_gradient_descent()
    # demonstrate_advanced_optimizers()
    # demonstrate_ml_applications()
    # comprehensive_optimization_analysis()
    
    print("\n💡 Pro tip: Optimization is the bridge between theory and practice!")
    print("Every ML algorithm you'll ever use relies on optimization principles.")
    
    print("\n🎯 Success metrics:")
    print("• Can you implement gradient descent and understand convergence?")
    print("• Can you explain why Adam works better than SGD for many problems?")
    print("• Can you debug optimization problems using diagnostic tools?")
    print("• Can you choose appropriate optimizers for different ML tasks?")
    
    print("\n🏆 Master this week and you'll understand why machines can learn!")
    print("Optimization theory is the mathematical foundation of all AI progress!")