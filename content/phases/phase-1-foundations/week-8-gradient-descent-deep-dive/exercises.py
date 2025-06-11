"""
Neural Odyssey - Week 8: Gradient Descent Deep Dive and Advanced Optimization
Exercises for mastering the algorithms that train neural networks

This module implements advanced optimization concepts for deep learning:
- Gradient descent variants and their trade-offs
- Momentum methods and acceleration techniques
- Adaptive learning rate optimizers (AdaGrad, RMSprop, Adam)
- Advanced optimization techniques for deep networks
- Practical optimization strategies and debugging

Complete the TODO functions to build your advanced optimization toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: GRADIENT DESCENT VARIANTS
# ==============================================

def batch_gradient_descent(X, y, weights, bias, learning_rate, loss_func, grad_func, max_epochs=100):
    """
    TODO: Implement batch gradient descent
    
    Uses entire dataset for each gradient computation.
    Most stable but slowest for large datasets.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training targets (n_samples,)
        weights: Initial weights (n_features,)
        bias: Initial bias (scalar)
        learning_rate: Step size
        loss_func: Function to compute loss
        grad_func: Function to compute gradients
        max_epochs: Maximum number of epochs
        
    Returns:
        Dictionary with optimization history
    """
    # TODO: Implement batch gradient descent
    # 1. For each epoch, compute loss and gradients on entire dataset
    # 2. Update weights: w = w - lr * dw
    # 3. Update bias: b = b - lr * db
    # 4. Track loss, weights, and convergence metrics
    
    pass

def stochastic_gradient_descent(X, y, weights, bias, learning_rate, loss_func, grad_func, 
                               max_epochs=100, random_state=42):
    """
    TODO: Implement stochastic gradient descent (SGD)
    
    Uses one random sample for each gradient computation.
    Fast updates but noisy convergence.
    
    Args:
        X: Training features
        y: Training targets
        weights: Initial weights
        bias: Initial bias
        learning_rate: Step size
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        random_state: For reproducible randomness
        
    Returns:
        Optimization history
    """
    # TODO: Implement SGD
    # 1. For each epoch, shuffle the data
    # 2. For each sample, compute gradients and update
    # 3. Track metrics (can evaluate loss periodically for efficiency)
    
    pass

def mini_batch_gradient_descent(X, y, weights, bias, learning_rate, loss_func, grad_func,
                               batch_size=32, max_epochs=100, random_state=42):
    """
    TODO: Implement mini-batch gradient descent
    
    Best of both worlds: stability of batch GD with speed of SGD.
    Industry standard for neural network training.
    
    Args:
        X: Training features
        y: Training targets
        weights: Initial weights
        bias: Initial bias
        learning_rate: Step size
        loss_func: Loss function
        grad_func: Gradient function
        batch_size: Size of mini-batches
        max_epochs: Maximum epochs
        random_state: For reproducibility
        
    Returns:
        Optimization history
    """
    # TODO: Implement mini-batch GD
    # 1. Split data into mini-batches
    # 2. For each batch, compute average gradients
    # 3. Update parameters once per batch
    # 4. Shuffle data between epochs
    
    pass

def compare_gradient_descent_variants(X, y, loss_func, grad_func, batch_sizes=[1, 32, None]):
    """
    TODO: Compare different gradient descent variants
    
    Shows trade-offs between convergence speed, stability, and computational cost.
    
    Args:
        X: Training data
        y: Training targets
        loss_func: Loss function
        grad_func: Gradient function
        batch_sizes: List of batch sizes (1=SGD, None=Batch GD)
        
    Returns:
        Comparison results and visualizations
    """
    # TODO: Run different GD variants and compare:
    # 1. Convergence speed (epochs to reach threshold)
    # 2. Final loss achieved
    # 3. Computational cost per epoch
    # 4. Stability of convergence
    
    pass

def learning_rate_analysis(X, y, loss_func, grad_func, learning_rates=[0.001, 0.01, 0.1, 1.0]):
    """
    TODO: Analyze effect of learning rate on convergence
    
    Critical hyperparameter that determines training success.
    
    Args:
        X, y: Training data
        loss_func: Loss function
        grad_func: Gradient function
        learning_rates: List of learning rates to test
        
    Returns:
        Analysis of learning rate effects
    """
    # TODO: Test different learning rates and analyze:
    # 1. Convergence speed vs learning rate
    # 2. Stability vs learning rate
    # 3. Final performance vs learning rate
    # 4. Identify optimal learning rate range
    
    pass

# ==============================================
# PART 2: MOMENTUM METHODS
# ==============================================

def momentum_gradient_descent(X, y, weights, bias, learning_rate=0.01, momentum=0.9,
                             loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement momentum gradient descent
    
    Momentum accelerates convergence by accumulating velocity.
    Helps overcome local minima and reduces oscillations.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Step size
        momentum: Momentum coefficient (typically 0.9)
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history with momentum tracking
    """
    # TODO: Implement momentum GD
    # v_w = momentum * v_w + learning_rate * dw
    # v_b = momentum * v_b + learning_rate * db
    # weights = weights - v_w
    # bias = bias - v_b
    
    pass

def nesterov_accelerated_gradient(X, y, weights, bias, learning_rate=0.01, momentum=0.9,
                                 loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement Nesterov Accelerated Gradient (NAG)
    
    "Look ahead" momentum that computes gradients at the anticipated position.
    Often converges faster than standard momentum.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Step size
        momentum: Momentum coefficient
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history
    """
    # TODO: Implement Nesterov momentum
    # 1. Compute look-ahead position: w_lookahead = w - momentum * v_w
    # 2. Compute gradients at look-ahead position
    # 3. Update velocity and parameters
    
    pass

def momentum_visualization(objective_func, grad_func, start_point=(-1, 1), momentum_values=[0, 0.5, 0.9]):
    """
    TODO: Visualize momentum effects on optimization trajectory
    
    Shows how momentum helps escape valleys and reduces oscillations.
    
    Args:
        objective_func: 2D function to optimize
        grad_func: Gradient function
        start_point: Starting position
        momentum_values: Different momentum coefficients to compare
        
    Returns:
        Visualization of optimization trajectories
    """
    # TODO: Create visualization showing:
    # 1. Contour plot of objective function
    # 2. Optimization trajectories for different momentum values
    # 3. Speed and stability comparisons
    
    pass

def momentum_parameter_sensitivity(X, y, momentum_range=np.linspace(0, 0.99, 20)):
    """
    TODO: Analyze sensitivity to momentum parameter
    
    Shows optimal momentum values for different problems.
    
    Args:
        X, y: Training data
        momentum_range: Range of momentum values to test
        
    Returns:
        Analysis of momentum parameter effects
    """
    # TODO: Test different momentum values and measure:
    # 1. Convergence speed
    # 2. Final loss achieved
    # 3. Training stability
    
    pass

# ==============================================
# PART 3: ADAPTIVE LEARNING RATE METHODS
# ==============================================

def adagrad_optimizer(X, y, weights, bias, learning_rate=0.01, epsilon=1e-8,
                     loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement AdaGrad optimizer
    
    Adapts learning rate for each parameter based on historical gradients.
    Good for sparse features but can stop learning too early.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Base learning rate
        epsilon: Small constant for numerical stability
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history with adaptive learning rates
    """
    # TODO: Implement AdaGrad
    # G_w += dw * dw  (accumulate squared gradients)
    # G_b += db * db
    # weights -= (learning_rate / sqrt(G_w + epsilon)) * dw
    # bias -= (learning_rate / sqrt(G_b + epsilon)) * db
    
    pass

def rmsprop_optimizer(X, y, weights, bias, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8,
                     loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement RMSprop optimizer
    
    Fixes AdaGrad's diminishing learning rate problem using exponential moving average.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Base learning rate
        decay_rate: Exponential decay rate for squared gradients
        epsilon: Numerical stability constant
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history
    """
    # TODO: Implement RMSprop
    # v_w = decay_rate * v_w + (1 - decay_rate) * dw^2
    # v_b = decay_rate * v_b + (1 - decay_rate) * db^2
    # weights -= (learning_rate / sqrt(v_w + epsilon)) * dw
    # bias -= (learning_rate / sqrt(v_b + epsilon)) * db
    
    pass

def adam_optimizer(X, y, weights, bias, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                  loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement Adam optimizer
    
    Combines momentum with adaptive learning rates.
    Currently the most popular optimizer for deep learning.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Base learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        epsilon: Numerical stability constant
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history
    """
    # TODO: Implement Adam
    # m_w = beta1 * m_w + (1 - beta1) * dw     (first moment)
    # v_w = beta2 * v_w + (1 - beta2) * dw^2   (second moment)
    # m_w_hat = m_w / (1 - beta1^t)            (bias correction)
    # v_w_hat = v_w / (1 - beta2^t)            (bias correction)
    # weights -= learning_rate * m_w_hat / (sqrt(v_w_hat) + epsilon)
    
    pass

def adamw_optimizer(X, y, weights, bias, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                   epsilon=1e-8, weight_decay=0.01, loss_func=None, grad_func=None, max_epochs=100):
    """
    TODO: Implement AdamW optimizer
    
    Adam with decoupled weight decay regularization.
    Often works better than Adam for deep learning.
    
    Args:
        X, y: Training data
        weights, bias: Initial parameters
        learning_rate: Base learning rate
        beta1, beta2: Moment decay rates
        epsilon: Numerical stability
        weight_decay: L2 regularization strength
        loss_func: Loss function
        grad_func: Gradient function
        max_epochs: Maximum epochs
        
    Returns:
        Optimization history
    """
    # TODO: Implement AdamW
    # Same as Adam but with decoupled weight decay:
    # weights -= learning_rate * (m_w_hat / (sqrt(v_w_hat) + epsilon) + weight_decay * weights)
    
    pass

def compare_adaptive_optimizers(X, y, loss_func, grad_func):
    """
    TODO: Compare different adaptive optimizers
    
    Shows strengths and weaknesses of each adaptive method.
    
    Args:
        X, y: Training data
        loss_func: Loss function
        grad_func: Gradient function
        
    Returns:
        Comprehensive comparison of adaptive optimizers
    """
    # TODO: Compare AdaGrad, RMSprop, Adam, AdamW on:
    # 1. Convergence speed
    # 2. Final performance
    # 3. Hyperparameter sensitivity
    # 4. Memory requirements
    
    pass

# ==============================================
# PART 4: LEARNING RATE SCHEDULES
# ==============================================

def step_decay_schedule(initial_lr, drop_rate=0.5, epochs_drop=10):
    """
    TODO: Implement step decay learning rate schedule
    
    Reduces learning rate by a factor every few epochs.
    
    Args:
        initial_lr: Starting learning rate
        drop_rate: Factor to multiply learning rate by
        epochs_drop: How often to drop the learning rate
        
    Returns:
        Function that returns learning rate for given epoch
    """
    # TODO: Return function that computes lr = initial_lr * drop_rate^(epoch // epochs_drop)
    
    pass

def exponential_decay_schedule(initial_lr, decay_rate=0.96):
    """
    TODO: Implement exponential decay schedule
    
    Smoothly decreases learning rate exponentially.
    
    Args:
        initial_lr: Starting learning rate
        decay_rate: Exponential decay factor
        
    Returns:
        Function that returns learning rate for given epoch
    """
    # TODO: Return function that computes lr = initial_lr * decay_rate^epoch
    
    pass

def cosine_annealing_schedule(initial_lr, T_max, eta_min=0):
    """
    TODO: Implement cosine annealing schedule
    
    Learning rate follows cosine curve, popular in deep learning.
    
    Args:
        initial_lr: Maximum learning rate
        T_max: Maximum number of epochs
        eta_min: Minimum learning rate
        
    Returns:
        Function that returns learning rate for given epoch
    """
    # TODO: Return function that computes cosine annealing:
    # lr = eta_min + (initial_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    
    pass

def warm_restart_schedule(initial_lr, T_0=10, T_mult=2, eta_min=0):
    """
    TODO: Implement cosine annealing with warm restarts
    
    Periodically restarts learning rate to escape local minima.
    
    Args:
        initial_lr: Maximum learning rate
        T_0: Initial restart period
        T_mult: Factor to increase restart period
        eta_min: Minimum learning rate
        
    Returns:
        Function that returns learning rate for given epoch
    """
    # TODO: Implement SGDR (Stochastic Gradient Descent with Warm Restarts)
    
    pass

def compare_learning_schedules(X, y, loss_func, grad_func, schedules_to_test):
    """
    TODO: Compare different learning rate schedules
    
    Shows impact of learning rate scheduling on convergence.
    
    Args:
        X, y: Training data
        loss_func: Loss function
        grad_func: Gradient function
        schedules_to_test: List of schedule functions
        
    Returns:
        Comparison of different schedules
    """
    # TODO: Test each schedule and compare:
    # 1. Convergence curves
    # 2. Final performance
    # 3. Training stability
    
    pass

# ==============================================
# PART 5: ADVANCED OPTIMIZATION TECHNIQUES
# ==============================================

def gradient_clipping(gradients, max_norm=1.0, norm_type=2):
    """
    TODO: Implement gradient clipping
    
    Prevents exploding gradients by limiting gradient norm.
    Essential for training RNNs and some deep networks.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm (1, 2, or inf)
        
    Returns:
        Clipped gradients
    """
    # TODO: Implement gradient clipping
    # 1. Compute total norm of all gradients
    # 2. If norm > max_norm, scale all gradients by max_norm/norm
    
    pass

def gradient_accumulation(X, y, weights, bias, optimizer_func, accumulation_steps=4):
    """
    TODO: Implement gradient accumulation
    
    Simulates large batch training with limited memory.
    Accumulates gradients over multiple mini-batches before updating.
    
    Args:
        X, y: Training data
        weights, bias: Parameters
        optimizer_func: Optimizer function to use
        accumulation_steps: Number of steps to accumulate
        
    Returns:
        Training results with gradient accumulation
    """
    # TODO: Implement gradient accumulation
    # 1. Forward pass and compute gradients for mini-batch
    # 2. Accumulate gradients (don't update parameters yet)
    # 3. After accumulation_steps, apply accumulated gradients
    # 4. Reset accumulated gradients
    
    pass

def lookahead_optimizer(base_optimizer, k=5, alpha=0.5):
    """
    TODO: Implement Lookahead optimizer wrapper
    
    Wraps any optimizer to make it more stable and less sensitive to hyperparameters.
    
    Args:
        base_optimizer: Base optimizer to wrap
        k: Number of fast weight updates before slow weight update
        alpha: Interpolation factor for slow weights
        
    Returns:
        Lookahead-wrapped optimizer
    """
    # TODO: Implement Lookahead
    # 1. Maintain fast weights (updated by base optimizer)
    # 2. Maintain slow weights (updated every k steps)
    # 3. Every k steps: slow_weights = slow_weights + alpha * (fast_weights - slow_weights)
    
    pass

def learning_rate_finder(X, y, model, loss_func, initial_lr=1e-8, final_lr=10, beta=0.98):
    """
    TODO: Implement learning rate finder
    
    Systematically finds good learning rate range by testing exponentially increasing rates.
    
    Args:
        X, y: Training data
        model: Model to train
        loss_func: Loss function
        initial_lr: Starting learning rate
        final_lr: Final learning rate
        beta: Smoothing factor for loss
        
    Returns:
        Learning rates and corresponding losses
    """
    # TODO: Implement LR finder
    # 1. Train for one epoch with exponentially increasing learning rate
    # 2. Track smoothed loss at each learning rate
    # 3. Stop when loss explodes
    # 4. Return lr vs loss curve for analysis
    
    pass

# ==============================================
# PART 6: NEURAL NETWORK OPTIMIZATION
# ==============================================

def optimize_neural_network(X, y, architecture, optimizer='adam', learning_rate=0.001,
                           epochs=100, batch_size=32, validation_split=0.2):
    """
    TODO: Complete neural network optimization with advanced techniques
    
    Integrates all optimization techniques for neural network training.
    
    Args:
        X, y: Training data
        architecture: Network architecture specification
        optimizer: Optimizer to use
        learning_rate: Initial learning rate
        epochs: Number of training epochs
        batch_size: Mini-batch size
        validation_split: Fraction of data for validation
        
    Returns:
        Trained model and training history
    """
    # TODO: Implement complete neural network optimization
    # 1. Split data into train/validation
    # 2. Initialize network with proper weight initialization
    # 3. Apply selected optimizer with learning rate schedule
    # 4. Track training and validation metrics
    # 5. Apply early stopping if validation loss increases
    
    pass

def hyperparameter_optimization_demo(X, y, param_grid):
    """
    TODO: Demonstrate hyperparameter optimization for optimizers
    
    Shows how to systematically tune optimizer hyperparameters.
    
    Args:
        X, y: Training data
        param_grid: Grid of hyperparameters to search
        
    Returns:
        Best hyperparameters and performance comparison
    """
    # TODO: Implement hyperparameter search
    # 1. Grid search or random search over parameter space
    # 2. Cross-validation for robust evaluation
    # 3. Track best parameters and performance
    
    pass

def optimization_diagnostics(training_history):
    """
    TODO: Analyze optimization behavior and diagnose problems
    
    Helps identify and fix common optimization issues.
    
    Args:
        training_history: Dictionary with loss/metric history
        
    Returns:
        Diagnostic analysis and recommendations
    """
    # TODO: Analyze training curves and diagnose:
    # 1. Learning rate too high/low
    # 2. Optimizer not suitable for problem
    # 3. Gradient vanishing/exploding
    # 4. Overfitting/underfitting patterns
    
    pass

# ==============================================
# PART 7: OPTIMIZATION BENCHMARKING
# ==============================================

class OptimizationBenchmark:
    """
    TODO: Build comprehensive optimization benchmarking suite
    
    Tests optimizers on various problems to understand their strengths/weaknesses.
    """
    
    def __init__(self):
        self.results = {}
        self.test_functions = {}
        
    def add_test_function(self, name, func, grad_func, optimal_value=None):
        """
        TODO: Add test function to benchmark suite
        """
        # TODO: Store test function and its properties
        
        pass
    
    def benchmark_optimizer(self, optimizer_func, test_name, **optimizer_kwargs):
        """
        TODO: Benchmark optimizer on specific test function
        """
        # TODO: Run optimizer on test function and collect metrics:
        # 1. Convergence speed
        # 2. Final objective value
        # 3. Stability across runs
        # 4. Computational cost
        
        pass
    
    def compare_optimizers(self, optimizers_to_test, test_functions=None):
        """
        TODO: Compare multiple optimizers across test functions
        """
        # TODO: Run all optimizers on all test functions
        # Create comprehensive comparison report
        
        pass
    
    def generate_benchmark_report(self):
        """
        TODO: Generate comprehensive benchmark report
        """
        # TODO: Create detailed analysis including:
        # 1. Optimizer rankings by problem type
        # 2. Hyperparameter sensitivity analysis
        # 3. Computational efficiency comparison
        # 4. Recommendations for different scenarios
        
        pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_gradient_descent_variants():
    """Demonstrate different gradient descent variants."""
    
    print("📈 Gradient Descent Variants Demonstration")
    print("=" * 50)
    
    try:
        # Create synthetic regression problem
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        true_weights = np.array([1, -2, 0.5, 3, -1])
        y = X @ true_weights + 0.1 * np.random.randn(1000)
        
        # Define loss and gradient functions
        def mse_loss(X, y, weights, bias):
            pred = X @ weights + bias
            return 0.5 * np.mean((pred - y) ** 2)
        
        def mse_gradients(X, y, weights, bias):
            pred = X @ weights + bias
            error = pred - y
            dw = X.T @ error / len(X)
            db = np.mean(error)
            return dw, db
        
        # Initialize parameters
        weights = np.random.randn(5) * 0.1
        bias = 0.0
        
        print("Testing gradient descent variants...")
        
        # Test batch GD
        batch_result = batch_gradient_descent(X, y, weights.copy(), bias, 0.01, 
                                            mse_loss, mse_gradients, max_epochs=50)
        print(f"Batch GD final loss: {batch_result['final_loss']:.6f}")
        
        # Test SGD
        sgd_result = stochastic_gradient_descent(X, y, weights.copy(), bias, 0.01,
                                               mse_loss, mse_gradients, max_epochs=50)
        print(f"SGD final loss: {sgd_result['final_loss']:.6f}")
        
        # Test mini-batch GD
        mb_result = mini_batch_gradient_descent(X, y, weights.copy(), bias, 0.01,
                                              mse_loss, mse_gradients, batch_size=32, max_epochs=50)
        print(f"Mini-batch GD final loss: {mb_result['final_loss']:.6f}")
        
        # Compare variants
        comparison = compare_gradient_descent_variants(X, y, mse_loss, mse_gradients)
        print("✅ Gradient descent variants comparison completed")
        
    except Exception as e:
        print(f"❌ GD variants demo failed: {e}")
        print("Implement the gradient descent variant functions!")

def demonstrate_momentum_methods():
    """Demonstrate momentum-based optimization."""
    
    print("\n🚀 Momentum Methods Demonstration")
    print("=" * 50)
    
    try:
        # Create challenging optimization problem (Rosenbrock function)
        def rosenbrock_2d(x, y):
            return 100 * (y - x**2)**2 + (1 - x)**2
        
        def rosenbrock_grad(x, y):
            dx = -400 * x * (y - x**2) - 2 * (1 - x)
            dy = 200 * (y - x**2)
            return np.array([dx, dy])
        
        # Test momentum methods
        start_point = np.array([-1.0, 1.0])
        
        print("Testing momentum methods on Rosenbrock function...")
        
        # Standard momentum
        momentum_result = momentum_gradient_descent(
            None, None, start_point, 0, learning_rate=0.001, momentum=0.9,
            max_epochs=1000
        )
        print(f"Momentum GD converged in {momentum_result['epochs']} epochs")
        
        # Nesterov momentum
        nesterov_result = nesterov_accelerated_gradient(
            None, None, start_point, 0, learning_rate=0.001, momentum=0.9,
            max_epochs=1000
        )
        print(f"Nesterov GD converged in {nesterov_result['epochs']} epochs")
        
        # Visualize momentum effects
        momentum_visualization(rosenbrock_2d, rosenbrock_grad)
        print("✅ Momentum visualization created")
        
    except Exception as e:
        print(f"❌ Momentum demo failed: {e}")
        print("Implement the momentum method functions!")

def demonstrate_adaptive_optimizers():
    """Demonstrate adaptive learning rate optimizers."""
    
    print("\n🎯 Adaptive Optimizers Demonstration")
    print("=" * 50)
    
    try:
        # Create classification dataset
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Define logistic regression loss and gradients
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
        def logistic_loss(X, y, weights, bias):
            z = X @ weights + bias
            pred = sigmoid(z)
            return -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
        
        def logistic_gradients(X, y, weights, bias):
            z = X @ weights + bias
            pred = sigmoid(z)
            error = pred - y
            dw = X.T @ error / len(X)
            db = np.mean(error)
            return dw, db
        
        # Initialize parameters
        weights = np.random.randn(10) * 0.1
        bias = 0.0
        
        print("Testing adaptive optimizers...")
        
        # Test AdaGrad
        adagrad_result = adagrad_optimizer(X, y, weights.copy(), bias, 0.1,
                                         loss_func=logistic_loss, grad_func=logistic_gradients)
        print(f"AdaGrad final loss: {adagrad_result['final_loss']:.6f}")
        
        # Test RMSprop
        rmsprop_result = rmsprop_optimizer(X, y, weights.copy(), bias, 0.01,
                                         loss_func=logistic_loss, grad_func=logistic_gradients)
        print(f"RMSprop final loss: {rmsprop_result['final_loss']:.6f}")
        
        # Test Adam
        adam_result = adam_optimizer(X, y, weights.copy(), bias, 0.01,
                                   loss_func=logistic_loss, grad_func=logistic_gradients)
        print(f"Adam final loss: {adam_result['final_loss']:.6f}")
        
        # Compare adaptive optimizers
        comparison = compare_adaptive_optimizers(X, y, logistic_loss, logistic_gradients)
        print("✅ Adaptive optimizers comparison completed")
        
    except Exception as e:
        print(f"❌ Adaptive optimizers demo failed: {e}")
        print("Implement the adaptive optimizer functions!")

def demonstrate_learning_rate_schedules():
    """Demonstrate learning rate scheduling."""
    
    print("\n📅 Learning Rate Schedules Demonstration")
    print("=" * 50)
    
    try:
        # Create different learning rate schedules
        initial_lr = 0.1
        epochs = 100
        
        step_schedule = step_decay_schedule(initial_lr, drop_rate=0.5, epochs_drop=20)
        exp_schedule = exponential_decay_schedule(initial_lr, decay_rate=0.95)
        cosine_schedule = cosine_annealing_schedule(initial_lr, T_max=epochs)
        restart_schedule = warm_restart_schedule(initial_lr, T_0=25, T_mult=2)
        
        print("Learning rate schedules created:")
        print(f"Step decay: lr={step_schedule(0):.4f} -> {step_schedule(epochs-1):.6f}")
        print(f"Exponential: lr={exp_schedule(0):.4f} -> {exp_schedule(epochs-1):.6f}")
        print(f"Cosine: lr={cosine_schedule(0):.4f} -> {cosine_schedule(epochs-1):.6f}")
        
        # Visualize schedules
        epoch_range = np.arange(epochs)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epoch_range, [step_schedule(e) for e in epoch_range], label='Step Decay')
        plt.plot(epoch_range, [exp_schedule(e) for e in epoch_range], label='Exponential')
        plt.plot(epoch_range, [cosine_schedule(e) for e in epoch_range], label='Cosine')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Learning Rate Schedules')
        
        plt.subplot(1, 2, 2)
        plt.plot(epoch_range, [restart_schedule(e) for e in epoch_range], label='Warm Restart')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Cosine Annealing with Warm Restarts')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Learning rate schedules visualization created")
        
    except Exception as e:
        print(f"❌ Learning rate schedules demo failed: {e}")
        print("Implement the learning rate schedule functions!")

def demonstrate_advanced_techniques():
    """Demonstrate advanced optimization techniques."""
    
    print("\n🔬 Advanced Optimization Techniques")
    print("=" * 50)
    
    try:
        # Generate sample gradients for testing
        gradients = [
            np.random.randn(10) * 5,  # Large gradients
            np.random.randn(5, 5) * 3,
            np.random.randn(3) * 10
        ]
        
        print("Testing gradient clipping...")
        original_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
        clipped_gradients = gradient_clipping(gradients, max_norm=1.0)
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped_gradients))
        
        print(f"Original gradient norm: {original_norm:.4f}")
        print(f"Clipped gradient norm: {clipped_norm:.4f}")
        
        # Test learning rate finder
        print("\nTesting learning rate finder...")
        # Create simple dataset for LR finder
        X_lr = np.random.randn(100, 5)
        y_lr = np.random.randn(100)
        
        # Mock model for LR finder demo
        class SimpleModel:
            def __init__(self):
                self.weights = np.random.randn(5)
                self.bias = 0.0
            
            def forward(self, X):
                return X @ self.weights + self.bias
            
            def loss(self, X, y):
                pred = self.forward(X)
                return 0.5 * np.mean((pred - y)**2)
        
        model = SimpleModel()
        lr_finder_result = learning_rate_finder(X_lr, y_lr, model, model.loss)
        print(f"✅ Learning rate finder completed, tested {len(lr_finder_result['lrs'])} learning rates")
        
    except Exception as e:
        print(f"❌ Advanced techniques demo failed: {e}")
        print("Implement the advanced optimization functions!")

def comprehensive_optimization_demo():
    """Comprehensive optimization demonstration."""
    
    print("\n🚀 Comprehensive Optimization Demonstration")
    print("=" * 50)
    
    try:
        # Load real dataset for comprehensive demo
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Convert to binary classification for simplicity
        binary_mask = (y == 0) | (y == 1)
        X_binary = X[binary_mask]
        y_binary = y[binary_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_binary, y_binary, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
        
        # Define neural network architecture
        architecture = {
            'layers': [64, 32, 16, 1],  # Hidden layers + output
            'activations': ['relu', 'relu', 'relu', 'sigmoid']
        }
        
        # Test comprehensive optimization
        result = optimize_neural_network(
            X_train, y_train, architecture, 
            optimizer='adam', learning_rate=0.001,
            epochs=50, batch_size=32, validation_split=0.2
        )
        
        print(f"✅ Neural network optimization completed")
        print(f"Final training accuracy: {result['train_accuracy']:.3f}")
        print(f"Final validation accuracy: {result['val_accuracy']:.3f}")
        
        # Hyperparameter optimization demo
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'optimizer': ['sgd', 'adam', 'rmsprop'],
            'batch_size': [16, 32, 64]
        }
        
        hp_result = hyperparameter_optimization_demo(X_train, y_train, param_grid)
        print(f"✅ Hyperparameter optimization completed")
        print(f"Best parameters: {hp_result['best_params']}")
        
        # Optimization diagnostics
        diagnostics = optimization_diagnostics(result['history'])
        print("✅ Optimization diagnostics completed")
        
        # Benchmarking suite
        benchmark = OptimizationBenchmark()
        
        # Add test functions
        benchmark.add_test_function('quadratic', lambda x: np.sum(x**2), lambda x: 2*x, 0)
        benchmark.add_test_function('rosenbrock', 
                                  lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2,
                                  lambda x: np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 
                                                     200*(x[1]-x[0]**2)]), 0)
        
        # Benchmark optimizers
        optimizers = ['sgd', 'momentum', 'adam', 'rmsprop']
        benchmark_result = benchmark.compare_optimizers(optimizers)
        
        print("✅ Optimization benchmarking completed")
        
        # Generate comprehensive report
        report = benchmark.generate_benchmark_report()
        print("✅ Benchmark report generated")
        
        print("\n🎉 Congratulations! You've mastered advanced optimization!")
        print("You now understand the algorithms that train all modern AI systems.")
        
    except Exception as e:
        print(f"❌ Comprehensive demo failed: {e}")
        print("Implement the comprehensive optimization functions!")

if __name__ == "__main__":
    """
    Run this file to explore advanced gradient descent and optimization!
    
    Complete the TODO functions above, then run:
    python week8_exercises.py
    """
    
    print("⚡ Welcome to Neural Odyssey Week 8: Gradient Descent Deep Dive!")
    print("Complete the TODO functions to master advanced optimization algorithms.")
    print("\nTo get started:")
    print("1. Implement gradient descent variants (batch, SGD, mini-batch)")
    print("2. Build momentum methods for acceleration")
    print("3. Create adaptive optimizers (AdaGrad, RMSprop, Adam)")
    print("4. Add learning rate schedules and advanced techniques")
    print("5. Build comprehensive optimization benchmarking suite")
    
    # Uncomment these lines after implementing the functions:
    # demonstrate_gradient_descent_variants()
    # demonstrate_momentum_methods()
    # demonstrate_adaptive_optimizers()
    # demonstrate_learning_rate_schedules()
    # demonstrate_advanced_techniques()
    # comprehensive_optimization_demo()
    
    print("\n💡 Pro tip: The right optimizer can make or break your model!")
    print("Understanding optimization deeply will make you a master ML practitioner.")
    
    print("\n🎯 Success metrics:")
    print("• Can you explain why Adam works better than SGD for many problems?")
    print("• Can you implement momentum and understand its acceleration effects?")
    print("• Can you design learning rate schedules for different scenarios?")
    print("• Can you debug optimization problems using diagnostic tools?")
    
    print("\n🏆 Master this week and you'll understand the engines that power AI!")
    print("From GPT to image recognition, it all comes down to optimization!")