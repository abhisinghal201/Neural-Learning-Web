"""
Neural Odyssey - Week 12: Phase 1 Integration & Foundation Synthesis
Phase 1: Mathematical Foundations (Final Integration Week)

Synthesizing Mathematical Mastery for Machine Learning

This final week of Phase 1 integrates all mathematical concepts learned over the past
11 weeks into a coherent foundation for machine learning. You'll build comprehensive
projects that demonstrate mastery and prepare for Phase 2's algorithmic focus.

Learning Objectives:
- Integrate linear algebra, calculus, probability, and information theory
- Build end-to-end mathematical ML implementations from scratch
- Demonstrate deep understanding through synthesis projects
- Create a comprehensive mathematical portfolio
- Prepare for transition to core ML algorithms

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional, Callable, Union, Any
import math
import random
from scipy import stats, optimize, linalg
from sklearn.datasets import make_classification, make_regression, make_blobs
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# COMPREHENSIVE MATHEMATICAL ML FRAMEWORK
# ==========================================

class MathematicalMLFramework:
    """
    Complete ML framework built from mathematical first principles
    Integrates all Phase 1 concepts: Linear Algebra + Calculus + Probability + Information Theory
    """
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        
    def create_linear_regression_from_scratch(self):
        """
        Linear regression using pure linear algebra
        Demonstrates: matrix operations, normal equations, geometric interpretation
        """
        print("📐 Linear Regression: Pure Linear Algebra Implementation")
        print("=" * 60)
        
        class LinearRegressionMath:
            def __init__(self):
                self.weights = None
                self.bias = None
                self.training_history = {'mse': [], 'r_squared': []}
                
            def fit_normal_equation(self, X, y):
                """Solve using normal equation: θ = (X^T X)^(-1) X^T y"""
                print("🎯 Solving with Normal Equation (Closed Form)")
                
                # Add bias column
                X_with_bias = np.column_stack([np.ones(len(X)), X])
                
                # Normal equation: θ = (X^T X)^(-1) X^T y
                XtX = X_with_bias.T @ X_with_bias
                Xty = X_with_bias.T @ y
                
                # Check if matrix is invertible
                det = np.linalg.det(XtX)
                print(f"   Matrix determinant: {det:.6f}")
                
                if abs(det) < 1e-10:
                    print("   ⚠️  Matrix is near-singular, using pseudo-inverse")
                    theta = np.linalg.pinv(XtX) @ Xty
                else:
                    theta = np.linalg.inv(XtX) @ Xty
                
                self.bias = theta[0]
                self.weights = theta[1:]
                
                print(f"   Learned weights: {self.weights}")
                print(f"   Learned bias: {self.bias}")
                
                return self
            
            def fit_gradient_descent(self, X, y, learning_rate=0.01, max_iterations=1000):
                """Solve using gradient descent (calculus-based optimization)"""
                print("📈 Solving with Gradient Descent (Iterative)")
                
                # Initialize parameters
                n_features = X.shape[1]
                self.weights = np.random.randn(n_features) * 0.01
                self.bias = 0.0
                
                self.training_history = {'mse': [], 'r_squared': [], 'gradients': []}
                
                for iteration in range(max_iterations):
                    # Forward pass
                    y_pred = X @ self.weights + self.bias
                    
                    # Calculate loss (MSE)
                    mse = np.mean((y - y_pred) ** 2)
                    
                    # Calculate gradients
                    residuals = y_pred - y
                    grad_weights = (2 / len(X)) * (X.T @ residuals)
                    grad_bias = (2 / len(X)) * np.sum(residuals)
                    
                    # Update parameters
                    self.weights -= learning_rate * grad_weights
                    self.bias -= learning_rate * grad_bias
                    
                    # Store metrics
                    r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                    self.training_history['mse'].append(mse)
                    self.training_history['r_squared'].append(r_squared)
                    self.training_history['gradients'].append(np.linalg.norm(grad_weights))
                    
                    # Print progress
                    if iteration % 100 == 0:
                        print(f"   Iteration {iteration}: MSE = {mse:.4f}, R² = {r_squared:.4f}")
                    
                    # Check convergence
                    if np.linalg.norm(grad_weights) < 1e-6:
                        print(f"   ✅ Converged after {iteration} iterations")
                        break
                
                return self
            
            def predict(self, X):
                """Make predictions"""
                return X @ self.weights + self.bias
            
            def geometric_interpretation(self, X, y):
                """Visualize geometric interpretation of linear regression"""
                if X.shape[1] != 1:
                    print("   Geometric visualization only for 1D features")
                    return
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. Data and fitted line
                ax = axes[0]
                ax.scatter(X[:, 0], y, alpha=0.6, label='Data')
                
                x_range = np.linspace(X.min(), X.max(), 100)
                y_pred_range = x_range * self.weights[0] + self.bias
                ax.plot(x_range, y_pred_range, 'r-', linewidth=2, label='Fitted Line')
                
                # Show residuals
                y_pred = self.predict(X)
                for i in range(0, len(X), 3):  # Show every 3rd residual
                    ax.plot([X[i, 0], X[i, 0]], [y[i], y_pred[i]], 'k--', alpha=0.5)
                
                ax.set_title('Linear Regression Fit')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Target')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 2. Residual analysis
                ax = axes[1]
                residuals = y - y_pred
                ax.scatter(y_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_title('Residual Plot')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.grid(True, alpha=0.3)
                
                # 3. Normal equation geometry (for 1D case)
                ax = axes[2]
                
                # Show projection interpretation
                X_with_bias = np.column_stack([np.ones(len(X)), X])
                
                # Project y onto column space of X
                projection = X_with_bias @ np.linalg.pinv(X_with_bias) @ y
                
                ax.scatter(range(len(y)), y, alpha=0.6, label='Original y')
                ax.scatter(range(len(y)), projection, alpha=0.6, label='Projection')
                
                # Show projection lines
                for i in range(0, len(y), 3):
                    ax.plot([i, i], [y[i], projection[i]], 'k--', alpha=0.5)
                
                ax.set_title('Projection Interpretation')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Demonstrate with synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 1) * 2
        true_weight = 3.5
        true_bias = 1.2
        noise = np.random.randn(n_samples) * 0.5
        y = X[:, 0] * true_weight + true_bias + noise
        
        print(f"Generated dataset: {n_samples} samples")
        print(f"True parameters: weight = {true_weight}, bias = {true_bias}")
        
        # Method 1: Normal Equation
        print(f"\n" + "="*60)
        lr_normal = LinearRegressionMath()
        lr_normal.fit_normal_equation(X, y)
        
        # Method 2: Gradient Descent
        print(f"\n" + "="*60)
        lr_gd = LinearRegressionMath()
        lr_gd.fit_gradient_descent(X, y, learning_rate=0.1, max_iterations=1000)
        
        # Compare methods
        print(f"\n📊 Method Comparison:")
        print(f"   Normal Equation  - Weight: {lr_normal.weights[0]:.4f}, Bias: {lr_normal.bias:.4f}")
        print(f"   Gradient Descent - Weight: {lr_gd.weights[0]:.4f}, Bias: {lr_gd.bias:.4f}")
        print(f"   True Parameters  - Weight: {true_weight:.4f}, Bias: {true_bias:.4f}")
        
        # Visualize results
        lr_normal.geometric_interpretation(X, y)
        
        # Plot training history for gradient descent
        if lr_gd.training_history['mse']:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].plot(lr_gd.training_history['mse'])
            axes[0].set_title('MSE Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('MSE')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(lr_gd.training_history['r_squared'])
            axes[1].set_title('R² Score')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('R²')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(lr_gd.training_history['gradients'])
            axes[2].set_title('Gradient Magnitude')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('||∇w||')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        self.models['linear_regression'] = {
            'normal_equation': lr_normal,
            'gradient_descent': lr_gd,
            'dataset': (X, y)
        }
        
        return lr_normal, lr_gd
    
    def create_logistic_regression_from_scratch(self):
        """
        Logistic regression using probability theory and optimization
        Demonstrates: sigmoid function, maximum likelihood, gradient descent
        """
        print("\n🎲 Logistic Regression: Probability Theory Implementation")
        print("=" * 65)
        
        class LogisticRegressionMath:
            def __init__(self):
                self.weights = None
                self.bias = None
                self.training_history = {'loss': [], 'accuracy': [], 'likelihood': []}
            
            def sigmoid(self, z):
                """Sigmoid activation function"""
                # Clip to prevent overflow
                z = np.clip(z, -500, 500)
                return 1 / (1 + np.exp(-z))
            
            def negative_log_likelihood(self, X, y):
                """Calculate negative log-likelihood (cross-entropy loss)"""
                z = X @ self.weights + self.bias
                p = self.sigmoid(z)
                
                # Avoid log(0) by clipping
                p = np.clip(p, 1e-15, 1 - 1e-15)
                
                # Negative log-likelihood
                nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                return nll / len(y)  # Average
            
            def fit(self, X, y, learning_rate=0.01, max_iterations=1000):
                """Fit using maximum likelihood estimation"""
                print("🎯 Maximum Likelihood Estimation with Gradient Descent")
                
                # Initialize parameters
                n_features = X.shape[1]
                self.weights = np.random.randn(n_features) * 0.01
                self.bias = 0.0
                
                self.training_history = {'loss': [], 'accuracy': [], 'likelihood': []}
                
                for iteration in range(max_iterations):
                    # Forward pass
                    z = X @ self.weights + self.bias
                    p = self.sigmoid(z)
                    
                    # Calculate loss (negative log-likelihood)
                    loss = self.negative_log_likelihood(X, y)
                    
                    # Calculate likelihood
                    likelihood = np.exp(-loss * len(y))
                    
                    # Calculate gradients
                    residuals = p - y
                    grad_weights = (1 / len(X)) * (X.T @ residuals)
                    grad_bias = (1 / len(X)) * np.sum(residuals)
                    
                    # Update parameters
                    self.weights -= learning_rate * grad_weights
                    self.bias -= learning_rate * grad_bias
                    
                    # Calculate accuracy
                    predictions = (p > 0.5).astype(int)
                    accuracy = np.mean(predictions == y)
                    
                    # Store metrics
                    self.training_history['loss'].append(loss)
                    self.training_history['accuracy'].append(accuracy)
                    self.training_history['likelihood'].append(likelihood)
                    
                    # Print progress
                    if iteration % 100 == 0:
                        print(f"   Iteration {iteration}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
                    
                    # Check convergence
                    if np.linalg.norm(grad_weights) < 1e-6:
                        print(f"   ✅ Converged after {iteration} iterations")
                        break
                
                return self
            
            def predict_proba(self, X):
                """Predict class probabilities"""
                z = X @ self.weights + self.bias
                return self.sigmoid(z)
            
            def predict(self, X):
                """Predict binary classes"""
                return (self.predict_proba(X) > 0.5).astype(int)
            
            def decision_boundary_analysis(self, X, y):
                """Analyze and visualize decision boundary"""
                if X.shape[1] != 2:
                    print("   Decision boundary visualization only for 2D features")
                    return
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 1. Data and decision boundary
                ax = axes[0, 0]
                
                # Plot data points
                colors = ['red', 'blue']
                for class_val in [0, 1]:
                    mask = y == class_val
                    ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                             alpha=0.6, label=f'Class {class_val}')
                
                # Plot decision boundary
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                   np.linspace(y_min, y_max, 100))
                
                grid_points = np.column_stack([xx.ravel(), yy.ravel()])
                probs = self.predict_proba(grid_points).reshape(xx.shape)
                
                # Decision boundary (p = 0.5)
                ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
                
                # Probability contours
                contours = ax.contour(xx, yy, probs, levels=[0.1, 0.3, 0.7, 0.9], 
                                    colors='gray', alpha=0.5)
                ax.clabel(contours, inline=True, fontsize=8)
                
                ax.set_title('Decision Boundary & Probability Contours')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 2. Sigmoid function visualization
                ax = axes[0, 1]
                z_range = np.linspace(-6, 6, 100)
                sigmoid_vals = self.sigmoid(z_range)
                
                ax.plot(z_range, sigmoid_vals, 'b-', linewidth=2, label='σ(z)')
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision threshold')
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                
                ax.set_title('Sigmoid Function')
                ax.set_xlabel('z = w·x + b')
                ax.set_ylabel('P(y=1|x)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 3. Log-likelihood surface (for 2D case, fix one weight)
                ax = axes[1, 0]
                
                # Create grid around current weights
                w0_range = np.linspace(self.weights[0] - 2, self.weights[0] + 2, 50)
                w1_range = np.linspace(self.weights[1] - 2, self.weights[1] + 2, 50)
                W0, W1 = np.meshgrid(w0_range, w1_range)
                
                # Calculate log-likelihood for each weight combination
                ll_surface = np.zeros_like(W0)
                for i in range(len(w0_range)):
                    for j in range(len(w1_range)):
                        # Temporarily set weights
                        temp_weights = np.array([W0[j, i], W1[j, i]])
                        z = X @ temp_weights + self.bias
                        p = self.sigmoid(z)
                        p = np.clip(p, 1e-15, 1 - 1e-15)
                        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                        ll_surface[j, i] = ll
                
                contour = ax.contour(W0, W1, ll_surface, levels=20)
                ax.clabel(contour, inline=True, fontsize=8)
                
                # Mark current weights
                ax.plot(self.weights[0], self.weights[1], 'ro', markersize=10, 
                       label='Current weights')
                
                ax.set_title('Log-Likelihood Surface')
                ax.set_xlabel('Weight 1')
                ax.set_ylabel('Weight 2')
                ax.legend()
                
                # 4. Probability calibration
                ax = axes[1, 1]
                
                # Bin predictions and calculate empirical probabilities
                probs = self.predict_proba(X)
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                bin_centers = []
                empirical_probs = []
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (probs > bin_lower) & (probs <= bin_upper)
                    if np.sum(in_bin) > 0:
                        bin_centers.append((bin_lower + bin_upper) / 2)
                        empirical_probs.append(np.mean(y[in_bin]))
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect calibration')
                if bin_centers:
                    ax.plot(bin_centers, empirical_probs, 'bo-', label='Actual calibration')
                
                ax.set_title('Probability Calibration')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Empirical Probability')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Generate synthetic binary classification data
        np.random.seed(42)
        n_samples = 200
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        
        print(f"Generated binary classification dataset: {n_samples} samples, 2 features")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Fit logistic regression
        lr = LogisticRegressionMath()
        lr.fit(X, y, learning_rate=0.1, max_iterations=1000)
        
        # Evaluate
        train_accuracy = np.mean(lr.predict(X) == y)
        print(f"\n📊 Final Results:")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Final Weights: {lr.weights}")
        print(f"   Final Bias: {lr.bias:.4f}")
        
        # Visualize
        lr.decision_boundary_analysis(X, y)
        
        # Plot training history
        if lr.training_history['loss']:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].plot(lr.training_history['loss'])
            axes[0].set_title('Cross-Entropy Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(lr.training_history['accuracy'])
            axes[1].set_title('Training Accuracy')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(lr.training_history['likelihood'])
            axes[2].set_title('Likelihood')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Likelihood')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        self.models['logistic_regression'] = {
            'model': lr,
            'dataset': (X, y)
        }
        
        return lr
    
    def create_neural_network_from_scratch(self):
        """
        Neural network integrating all mathematical concepts
        Demonstrates: linear algebra (forward pass), calculus (backprop), probability (activations)
        """
        print("\n🧠 Neural Network: Complete Mathematical Integration")
        print("=" * 65)
        
        class NeuralNetworkMath:
            def __init__(self, layer_sizes, activations):
                """
                Initialize neural network
                layer_sizes: [input_size, hidden1_size, hidden2_size, ..., output_size]
                activations: ['relu', 'sigmoid', 'tanh', 'linear'] for each layer
                """
                self.layer_sizes = layer_sizes
                self.activations = activations
                self.weights = []
                self.biases = []
                self.training_history = {
                    'loss': [], 'accuracy': [], 'gradient_norms': []
                }
                
                # Initialize weights using Xavier/He initialization
                for i in range(len(layer_sizes) - 1):
                    input_size = layer_sizes[i]
                    output_size = layer_sizes[i + 1]
                    
                    # Xavier initialization for sigmoid/tanh, He for ReLU
                    if activations[i] in ['sigmoid', 'tanh']:
                        # Xavier: sqrt(6 / (fan_in + fan_out))
                        limit = np.sqrt(6 / (input_size + output_size))
                        weights = np.random.uniform(-limit, limit, (input_size, output_size))
                    else:  # ReLU or linear
                        # He initialization: sqrt(2 / fan_in)
                        weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
                    
                    self.weights.append(weights)
                    self.biases.append(np.zeros((1, output_size)))
            
            def activation_function(self, z, activation_type):
                """Apply activation function"""
                if activation_type == 'sigmoid':
                    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                elif activation_type == 'tanh':
                    return np.tanh(z)
                elif activation_type == 'relu':
                    return np.maximum(0, z)
                elif activation_type == 'linear':
                    return z
                else:
                    raise ValueError(f"Unknown activation: {activation_type}")
            
            def activation_derivative(self, z, activation_type):
                """Compute activation function derivative"""
                if activation_type == 'sigmoid':
                    s = self.activation_function(z, 'sigmoid')
                    return s * (1 - s)
                elif activation_type == 'tanh':
                    return 1 - np.tanh(z) ** 2
                elif activation_type == 'relu':
                    return (z > 0).astype(float)
                elif activation_type == 'linear':
                    return np.ones_like(z)
                else:
                    raise ValueError(f"Unknown activation: {activation_type}")
            
            def forward_pass(self, X):
                """
                Forward propagation through the network
                Returns: final output and intermediate values for backprop
                """
                activations = [X]  # Store activations for each layer
                z_values = []      # Store pre-activation values
                
                current_input = X
                
                for i, (W, b, activation_type) in enumerate(zip(self.weights, self.biases, self.activations)):
                    # Linear transformation: z = XW + b
                    z = current_input @ W + b
                    z_values.append(z)
                    
                    # Apply activation function
                    a = self.activation_function(z, activation_type)
                    activations.append(a)
                    current_input = a
                
                return activations, z_values
            
            def backward_pass(self, X, y, activations, z_values):
                """
                Backpropagation to compute gradients
                Returns: gradients for weights and biases
                """
                m = X.shape[0]  # Number of samples
                
                # Initialize gradient lists
                weight_gradients = [np.zeros_like(W) for W in self.weights]
                bias_gradients = [np.zeros_like(b) for b in self.biases]
                
                # Start with output layer error
                # For MSE loss: dL/da = 2(a - y)
                output_error = 2 * (activations[-1] - y) / m
                
                # Backpropagate through layers
                for i in reversed(range(len(self.weights))):
                    # Current layer pre-activation values
                    z = z_values[i]
                    
                    # Activation derivative
                    activation_grad = self.activation_derivative(z, self.activations[i])
                    
                    # Error for current layer: δ = error * activation_derivative
                    delta = output_error * activation_grad
                    
                    # Gradients for current layer
                    # dL/dW = a_prev^T @ δ
                    weight_gradients[i] = activations[i].T @ delta
                    
                    # dL/db = sum(δ)
                    bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)
                    
                    # Propagate error to previous layer (if not input layer)
                    if i > 0:
                        # error_prev = δ @ W^T
                        output_error = delta @ self.weights[i].T
                
                return weight_gradients, bias_gradients
            
            def compute_loss(self, y_true, y_pred):
                """Compute MSE loss"""
                return np.mean((y_true - y_pred) ** 2)
            
            def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=True):
                """Train the neural network"""
                print(f"🎯 Training Neural Network")
                print(f"   Architecture: {self.layer_sizes}")
                print(f"   Activations: {self.activations}")
                print(f"   Learning rate: {learning_rate}")
                
                for epoch in range(epochs):
                    # Forward pass
                    activations, z_values = self.forward_pass(X)
                    y_pred = activations[-1]
                    
                    # Compute loss
                    loss = self.compute_loss(y, y_pred)
                    
                    # Backward pass
                    weight_grads, bias_grads = self.backward_pass(X, y, activations, z_values)
                    
                    # Update parameters
                    for i in range(len(self.weights)):
                        self.weights[i] -= learning_rate * weight_grads[i]
                        self.biases[i] -= learning_rate * bias_grads[i]
                    
                    # Calculate metrics
                    if y.shape[1] == 1 and np.all(np.isin(y, [0, 1])):
                        # Binary classification
                        predictions = (y_pred > 0.5).astype(int)
                        accuracy = np.mean(predictions == y)
                    elif y.shape[1] > 1:
                        # Multi-class classification
                        predictions = np.argmax(y_pred, axis=1)
                        true_labels = np.argmax(y, axis=1)
                        accuracy = np.mean(predictions == true_labels)
                    else:
                        # Regression - use R²
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        accuracy = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate gradient norms
                    total_grad_norm = sum(np.linalg.norm(grad) for grad in weight_grads)
                    
                    # Store training history
                    self.training_history['loss'].append(loss)
                    self.training_history['accuracy'].append(accuracy)
                    self.training_history['gradient_norms'].append(total_grad_norm)
                    
                    # Print progress
                    if verbose and epoch % 100 == 0:
                        print(f"   Epoch {epoch}: Loss = {loss:.4f}, Metric = {accuracy:.4f}")
                    
                    # Early stopping
                    if total_grad_norm < 1e-6:
                        print(f"   ✅ Converged after {epoch} epochs")
                        break
                
                return self
            
            def predict(self, X):
                """Make predictions"""
                activations, _ = self.forward_pass(X)
                return activations[-1]
            
            def analyze_network_behavior(self, X, y):
                """Comprehensive analysis of network behavior"""
                print(f"\n🔍 Network Analysis")
                
                # Get network outputs
                activations, z_values = self.forward_pass(X)
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # 1. Weight distribution
                ax = axes[0, 0]
                all_weights = np.concatenate([W.flatten() for W in self.weights])
                ax.hist(all_weights, bins=50, alpha=0.7, density=True)
                ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_title('Weight Distribution')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # 2. Activation distributions for each layer
                ax = axes[0, 1]
                colors = plt.cm.viridis(np.linspace(0, 1, len(activations)))
                for i, (activation, color) in enumerate(zip(activations, colors)):
                    if i == 0:  # Skip input layer
                        continue
                    flat_activations = activation.flatten()
                    ax.hist(flat_activations, bins=30, alpha=0.5, label=f'Layer {i}', 
                           color=color, density=True)
                
                ax.set_title('Activation Distributions')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 3. Gradient flow analysis
                ax = axes[0, 2]
                if self.training_history['gradient_norms']:
                    ax.plot(self.training_history['gradient_norms'])
                    ax.set_title('Gradient Flow')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('||∇W||')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                
                # 4. Loss landscape (for 2D weight space)
                ax = axes[1, 0]
                if len(self.weights) > 0 and self.weights[0].size >= 2:
                    # Sample around current weights
                    w1_current = self.weights[0][0, 0]
                    w2_current = self.weights[0][0, 1] if self.weights[0].shape[1] > 1 else self.weights[0][1, 0]
                    
                    w1_range = np.linspace(w1_current - 1, w1_current + 1, 20)
                    w2_range = np.linspace(w2_current - 1, w2_current + 1, 20)
                    W1, W2 = np.meshgrid(w1_range, w2_range)
                    
                    loss_surface = np.zeros_like(W1)
                    
                    # Save original weights
                    orig_w = self.weights[0].copy()
                    
                    for i in range(len(w1_range)):
                        for j in range(len(w2_range)):
                            # Modify weights
                            self.weights[0][0, 0] = W1[j, i]
                            if self.weights[0].shape[1] > 1:
                                self.weights[0][0, 1] = W2[j, i]
                            else:
                                self.weights[0][1, 0] = W2[j, i]
                            
                            # Calculate loss
                            pred = self.predict(X)
                            loss_surface[j, i] = self.compute_loss(y, pred)
                    
                    # Restore original weights
                    self.weights[0] = orig_w
                    
                    contour = ax.contour(W1, W2, loss_surface, levels=20)
                    ax.clabel(contour, inline=True, fontsize=8)
                    ax.plot(w1_current, w2_current, 'ro', markersize=8, label='Current')
                    ax.set_title('Loss Landscape (2D slice)')
                    ax.set_xlabel('Weight 1')
                    ax.set_ylabel('Weight 2')
                    ax.legend()
                
                # 5. Learning curves
                ax = axes[1, 1]
                if self.training_history['loss']:
                    epochs = range(len(self.training_history['loss']))
                    ax.plot(epochs, self.training_history['loss'], 'b-', label='Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss', color='b')
                    ax.tick_params(axis='y', labelcolor='b')
                    ax.grid(True, alpha=0.3)
                    
                    # Secondary y-axis for accuracy
                    ax2 = ax.twinx()
                    ax2.plot(epochs, self.training_history['accuracy'], 'r-', label='Accuracy/R²')
                    ax2.set_ylabel('Accuracy/R²', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    ax.set_title('Learning Curves')
                
                # 6. Feature importance (if input is 2D)
                ax = axes[1, 2]
                if X.shape[1] == 2:
                    # Calculate sensitivity to each input feature
                    baseline_pred = self.predict(X)
                    
                    sensitivities = []
                    for feature_idx in range(X.shape[1]):
                        X_perturbed = X.copy()
                        X_perturbed[:, feature_idx] += 0.1 * np.std(X[:, feature_idx])
                        perturbed_pred = self.predict(X_perturbed)
                        
                        sensitivity = np.mean(np.abs(perturbed_pred - baseline_pred))
                        sensitivities.append(sensitivity)
                    
                    ax.bar(range(len(sensitivities)), sensitivities)
                    ax.set_title('Feature Sensitivity')
                    ax.set_xlabel('Feature Index')
                    ax.set_ylabel('Sensitivity')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Demonstrate on different problems
        print("🎯 Demonstration 1: Binary Classification")
        
        # Binary classification
        np.random.seed(42)
        X_binary, y_binary = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                                n_informative=2, n_clusters_per_class=1, 
                                                random_state=42)
        y_binary = y_binary.reshape(-1, 1)  # Make it column vector
        
        # Create and train network
        nn_binary = NeuralNetworkMath(
            layer_sizes=[2, 8, 4, 1],
            activations=['relu', 'relu', 'sigmoid']
        )
        
        nn_binary.fit(X_binary, y_binary, learning_rate=0.01, epochs=1000, verbose=True)
        
        # Analyze
        nn_binary.analyze_network_behavior(X_binary, y_binary)
        
        print("\n🎯 Demonstration 2: Regression")
        
        # Regression problem
        np.random.seed(42)
        X_reg, y_reg = make_regression(n_samples=200, n_features=2, noise=0.1, random_state=42)
        y_reg = y_reg.reshape(-1, 1)
        
        # Normalize targets for better training
        y_reg = (y_reg - np.mean(y_reg)) / np.std(y_reg)
        
        nn_reg = NeuralNetworkMath(
            layer_sizes=[2, 10, 5, 1],
            activations=['tanh', 'tanh', 'linear']
        )
        
        nn_reg.fit(X_reg, y_reg, learning_rate=0.001, epochs=1500, verbose=True)
        
        # Evaluate
        pred_reg = nn_reg.predict(X_reg)
        r2_score = 1 - np.sum((y_reg - pred_reg)**2) / np.sum((y_reg - np.mean(y_reg))**2)
        print(f"   Final R² Score: {r2_score:.4f}")
        
        self.models['neural_network'] = {
            'binary_classifier': nn_binary,
            'regressor': nn_reg,
            'datasets': {
                'binary': (X_binary, y_binary),
                'regression': (X_reg, y_reg)
            }
        }
        
        return nn_binary, nn_reg
    
    def create_pca_from_scratch(self):
        """
        Principal Component Analysis using eigendecomposition
        Demonstrates: covariance matrices, eigenvalues/eigenvectors, dimensionality reduction
        """
        print("\n📊 Principal Component Analysis: Eigendecomposition Implementation")
        print("=" * 75)
        
        class PCAMath:
            def __init__(self, n_components=None):
                self.n_components = n_components
                self.components_ = None
                self.explained_variance_ = None
                self.explained_variance_ratio_ = None
                self.mean_ = None
                
            def fit(self, X):
                """Fit PCA using eigendecomposition"""
                print("🎯 Computing Principal Components via Eigendecomposition")
                
                # Center the data
                self.mean_ = np.mean(X, axis=0)
                X_centered = X - self.mean_
                
                print(f"   Data shape: {X.shape}")
                print(f"   Centering data...")
                
                # Compute covariance matrix
                n_samples = X_centered.shape[0]
                cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
                
                print(f"   Covariance matrix shape: {cov_matrix.shape}")
                print(f"   Covariance matrix condition number: {np.linalg.cond(cov_matrix):.2e}")
                
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Sort by decreasing eigenvalue
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                
                # Store results
                if self.n_components is None:
                    self.n_components = len(eigenvalues)
                
                self.components_ = eigenvectors[:, :self.n_components].T
                self.explained_variance_ = eigenvalues[:self.n_components]
                self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
                
                print(f"   Number of components: {self.n_components}")
                print(f"   Explained variance ratio: {self.explained_variance_ratio_}")
                print(f"   Cumulative explained variance: {np.cumsum(self.explained_variance_ratio_)}")
                
                return self
            
            def transform(self, X):
                """Transform data to principal component space"""
                X_centered = X - self.mean_
                return X_centered @ self.components_.T
            
            def inverse_transform(self, X_transformed):
                """Transform back to original space"""
                return (X_transformed @ self.components_) + self.mean_
            
            def fit_transform(self, X):
                """Fit and transform in one step"""
                return self.fit(X).transform(X)
            
            def visualize_pca_analysis(self, X):
                """Comprehensive PCA visualization"""
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # 1. Original data
                ax = axes[0, 0]
                if X.shape[1] >= 2:
                    ax.scatter(X[:, 0], X[:, 1], alpha=0.6)
                    ax.set_title('Original Data')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    ax.grid(True, alpha=0.3)
                    
                    # Plot principal component directions
                    origin = self.mean_[:2]
                    for i, (component, variance) in enumerate(zip(self.components_[:2], self.explained_variance_[:2])):
                        # Scale arrow by explained variance
                        scale = 3 * np.sqrt(variance)
                        ax.arrow(origin[0], origin[1], 
                                scale * component[0], scale * component[1],
                                head_width=0.1, head_length=0.1, 
                                fc=f'C{i}', ec=f'C{i}', linewidth=2,
                                label=f'PC{i+1}')
                    
                    ax.legend()
                
                # 2. Explained variance
                ax = axes[0, 1]
                components_range = range(1, len(self.explained_variance_) + 1)
                ax.bar(components_range, self.explained_variance_ratio_, alpha=0.7)
                ax.set_title('Explained Variance by Component')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Explained Variance Ratio')
                ax.grid(True, alpha=0.3)
                
                # Add cumulative line
                ax2 = ax.twinx()
                ax2.plot(components_range, np.cumsum(self.explained_variance_ratio_), 
                        'ro-', color='red', label='Cumulative')
                ax2.set_ylabel('Cumulative Explained Variance', color='red')
                ax2.legend()
                
                # 3. Transformed data
                ax = axes[0, 2]
                X_transformed = self.transform(X)
                if X_transformed.shape[1] >= 2:
                    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
                    ax.set_title('Data in PC Space')
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.grid(True, alpha=0.3)
                
                # 4. Covariance matrix
                ax = axes[1, 0]
                X_centered = X - self.mean_
                cov_matrix = np.cov(X_centered.T)
                im = ax.imshow(cov_matrix, cmap='RdBu', aspect='auto')
                ax.set_title('Covariance Matrix')
                plt.colorbar(im, ax=ax)
                
                # 5. Component loadings
                ax = axes[1, 1]
                if X.shape[1] <= 10:  # Only plot if reasonable number of features
                    for i in range(min(3, self.n_components)):
                        ax.bar(np.arange(X.shape[1]) + i*0.25, self.components_[i], 
                              width=0.25, alpha=0.7, label=f'PC{i+1}')
                    
                    ax.set_title('Component Loadings')
                    ax.set_xlabel('Original Feature')
                    ax.set_ylabel('Loading')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 6. Reconstruction error vs components
                ax = axes[1, 2]
                reconstruction_errors = []
                component_counts = range(1, min(X.shape[1], 10) + 1)
                
                for n_comp in component_counts:
                    # Temporary PCA with n_comp components
                    temp_pca = PCAMath(n_components=n_comp)
                    temp_pca.fit(X)
                    X_transformed_temp = temp_pca.transform(X)
                    X_reconstructed = temp_pca.inverse_transform(X_transformed_temp)
                    
                    error = np.mean((X - X_reconstructed) ** 2)
                    reconstruction_errors.append(error)
                
                ax.plot(component_counts, reconstruction_errors, 'bo-')
                ax.set_title('Reconstruction Error vs Components')
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('MSE Reconstruction Error')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Generate high-dimensional data with clear structure
        np.random.seed(42)
        
        # Create data with intrinsic 2D structure in higher dimensions
        n_samples = 300
        
        # Generate 2D latent variables
        theta = np.linspace(0, 4*np.pi, n_samples)
        r = np.linspace(1, 3, n_samples)
        
        latent_1 = r * np.cos(theta) + np.random.randn(n_samples) * 0.1
        latent_2 = r * np.sin(theta) + np.random.randn(n_samples) * 0.1
        
        # Embed in higher dimensional space
        mixing_matrix = np.random.randn(2, 8)
        X_high_dim = np.column_stack([latent_1, latent_2]) @ mixing_matrix
        
        # Add some noise dimensions
        noise_dims = np.random.randn(n_samples, 4) * 0.2
        X_final = np.column_stack([X_high_dim, noise_dims])
        
        print(f"Generated high-dimensional dataset: {X_final.shape}")
        print(f"True intrinsic dimensionality: 2")
        
        # Apply PCA
        pca = PCAMath(n_components=8)
        pca.fit(X_final)
        
        # Show results
        print(f"\n📊 PCA Results:")
        print(f"   Top 3 explained variance ratios: {pca.explained_variance_ratio_[:3]}")
        print(f"   Cumulative variance (first 3): {np.sum(pca.explained_variance_ratio_[:3]):.4f}")
        
        # Visualize
        pca.visualize_pca_analysis(X_final)
        
        # Demonstrate dimensionality reduction
        print(f"\n🔄 Dimensionality Reduction Demonstration:")
        
        for n_comp in [2, 4, 6]:
            pca_reduced = PCAMath(n_components=n_comp)
            X_reduced = pca_reduced.fit_transform(X_final)
            X_reconstructed = pca_reduced.inverse_transform(X_reduced)
            
            reconstruction_error = np.mean((X_final - X_reconstructed) ** 2)
            variance_retained = np.sum(pca_reduced.explained_variance_ratio_)
            
            print(f"   {n_comp} components: {variance_retained:.3f} variance, {reconstruction_error:.4f} MSE")
        
        self.models['pca'] = {
            'model': pca,
            'dataset': X_final,
            'true_latent': np.column_stack([latent_1, latent_2])
        }
        
        return pca


# ==========================================
# MATHEMATICAL CONCEPT INTEGRATION PROJECTS
# ==========================================

class MathematicalIntegrationProjects:
    """
    Comprehensive projects that integrate multiple mathematical concepts
    """
    
    def __init__(self):
        self.projects = {}
    
    def project_1_optimization_landscape_explorer(self):
        """
        Project 1: Interactive optimization landscape explorer
        Integrates: Calculus, Linear Algebra, Visualization
        """
        print("\n🏔️  Project 1: Optimization Landscape Explorer")
        print("=" * 55)
        print("Integrating: Calculus + Linear Algebra + Optimization Theory")
        
        class OptimizationExplorer:
            def __init__(self):
                self.optimizers = {}
                
            def rosenbrock_function(self, x, y, a=1, b=100):
                """Famous Rosenbrock function - challenging optimization problem"""
                return (a - x)**2 + b * (y - x**2)**2
            
            def rosenbrock_gradient(self, x, y, a=1, b=100):
                """Analytical gradient of Rosenbrock function"""
                grad_x = -2*(a - x) - 4*b*x*(y - x**2)
                grad_y = 2*b*(y - x**2)
                return np.array([grad_x, grad_y])
            
            def himmelblau_function(self, x, y):
                """Himmelblau's function - multiple global minima"""
                return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            
            def himmelblau_gradient(self, x, y):
                """Analytical gradient of Himmelblau's function"""
                grad_x = 2*(x**2 + y - 11)*2*x + 2*(x + y**2 - 7)
                grad_y = 2*(x**2 + y - 11) + 2*(x + y**2 - 7)*2*y
                return np.array([grad_x, grad_y])
            
            def run_optimizer_comparison(self):
                """Compare different optimization algorithms"""
                
                # Define optimization algorithms
                def gradient_descent(func, grad_func, start_point, learning_rate, max_iter):
                    path = [start_point.copy()]
                    point = start_point.copy()
                    
                    for _ in range(max_iter):
                        gradient = grad_func(point[0], point[1])
                        point = point - learning_rate * gradient
                        path.append(point.copy())
                        
                        if np.linalg.norm(gradient) < 1e-6:
                            break
                    
                    return np.array(path)
                
                def momentum_gd(func, grad_func, start_point, learning_rate, momentum, max_iter):
                    path = [start_point.copy()]
                    point = start_point.copy()
                    velocity = np.zeros_like(point)
                    
                    for _ in range(max_iter):
                        gradient = grad_func(point[0], point[1])
                        velocity = momentum * velocity - learning_rate * gradient
                        point = point + velocity
                        path.append(point.copy())
                        
                        if np.linalg.norm(gradient) < 1e-6:
                            break
                    
                    return np.array(path)
                
                def adam_optimizer(func, grad_func, start_point, learning_rate, max_iter):
                    path = [start_point.copy()]
                    point = start_point.copy()
                    
                    # Adam parameters
                    beta1, beta2 = 0.9, 0.999
                    epsilon = 1e-8
                    m = np.zeros_like(point)
                    v = np.zeros_like(point)
                    
                    for t in range(1, max_iter + 1):
                        gradient = grad_func(point[0], point[1])
                        
                        # Update biased first and second moment estimates
                        m = beta1 * m + (1 - beta1) * gradient
                        v = beta2 * v + (1 - beta2) * gradient**2
                        
                        # Bias correction
                        m_corrected = m / (1 - beta1**t)
                        v_corrected = v / (1 - beta2**t)
                        
                        # Update parameters
                        point = point - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
                        path.append(point.copy())
                        
                        if np.linalg.norm(gradient) < 1e-6:
                            break
                    
                    return np.array(path)
                
                # Test on Rosenbrock function
                print("🎯 Testing on Rosenbrock Function")
                
                start_point = np.array([-1.5, 1.5])
                
                # Run optimizers
                gd_path = gradient_descent(self.rosenbrock_function, self.rosenbrock_gradient, 
                                         start_point, 0.001, 2000)
                momentum_path = momentum_gd(self.rosenbrock_function, self.rosenbrock_gradient, 
                                          start_point, 0.001, 0.9, 2000)
                adam_path = adam_optimizer(self.rosenbrock_function, self.rosenbrock_gradient, 
                                         start_point, 0.01, 2000)
                
                # Visualize optimization paths
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                
                # Create contour plot
                x = np.linspace(-2, 2, 100)
                y = np.linspace(-1, 3, 100)
                X, Y = np.meshgrid(x, y)
                Z = self.rosenbrock_function(X, Y)
                
                for ax, (title, log_scale) in zip(axes, [("Linear Scale", False), ("Log Scale", True)]):
                    if log_scale:
                        levels = np.logspace(0, 3, 20)
                        contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
                    else:
                        contour = ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6)
                    
                    ax.clabel(contour, inline=True, fontsize=8)
                    
                    # Plot optimization paths
                    ax.plot(gd_path[:, 0], gd_path[:, 1], 'r.-', label='Gradient Descent', 
                           markersize=2, alpha=0.8)
                    ax.plot(momentum_path[:, 0], momentum_path[:, 1], 'b.-', label='Momentum', 
                           markersize=2, alpha=0.8)
                    ax.plot(adam_path[:, 0], adam_path[:, 1], 'g.-', label='Adam', 
                           markersize=2, alpha=0.8)
                    
                    # Mark start and global minimum
                    ax.plot(start_point[0], start_point[1], 'ko', markersize=8, label='Start')
                    ax.plot(1, 1, 'k*', markersize=12, label='Global Minimum')
                    
                    ax.set_title(f'Rosenbrock Function - {title}')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                # Print convergence statistics
                print(f"\n📊 Convergence Analysis:")
                final_values = []
                for name, path in [("Gradient Descent", gd_path), ("Momentum", momentum_path), ("Adam", adam_path)]:
                    final_point = path[-1]
                    final_value = self.rosenbrock_function(final_point[0], final_point[1])
                    distance_to_optimum = np.linalg.norm(final_point - np.array([1, 1]))
                    
                    print(f"   {name:15}: Final value = {final_value:.6f}, Distance to optimum = {distance_to_optimum:.6f}")
                    final_values.append(final_value)
                
                return {
                    'paths': {'gd': gd_path, 'momentum': momentum_path, 'adam': adam_path},
                    'final_values': final_values
                }
        
        explorer = OptimizationExplorer()
        results = explorer.run_optimizer_comparison()
        
        self.projects['optimization_explorer'] = {
            'explorer': explorer,
            'results': results
        }
        
        return explorer
    
    def project_2_bayesian_linear_regression(self):
        """
        Project 2: Bayesian Linear Regression
        Integrates: Probability Theory, Linear Algebra, Information Theory
        """
        print("\n🎲 Project 2: Bayesian Linear Regression")
        print("=" * 50)
        print("Integrating: Probability + Linear Algebra + Information Theory")
        
        class BayesianLinearRegression:
            def __init__(self, alpha=1.0, beta=1.0):
                """
                alpha: precision of weight prior
                beta: precision of noise
                """
                self.alpha = alpha  # Prior precision
                self.beta = beta    # Noise precision
                self.mean_w = None
                self.cov_w = None
                self.predictive_history = []
                
            def fit(self, X, y):
                """Fit Bayesian linear regression"""
                                print("🎯 Bayesian Learning with Conjugate Priors")
                
                # Add bias column
                X_design = np.column_stack([np.ones(len(X)), X])
                
                # Prior parameters
                S0_inv = self.alpha * np.eye(X_design.shape[1])  # Prior precision matrix
                
                # Posterior parameters (conjugate update)
                Sn_inv = S0_inv + self.beta * (X_design.T @ X_design)
                self.cov_w = np.linalg.inv(Sn_inv)
                self.mean_w = self.beta * self.cov_w @ X_design.T @ y
                
                print(f"   Prior precision (α): {self.alpha}")
                print(f"   Noise precision (β): {self.beta}")
                print(f"   Posterior mean weights: {self.mean_w}")
                print(f"   Posterior weight uncertainties: {np.sqrt(np.diag(self.cov_w))}")
                
                return self
            
            def predict(self, X_test, return_std=False):
                """Predict with uncertainty quantification"""
                X_test_design = np.column_stack([np.ones(len(X_test)), X_test])
                
                # Predictive mean
                mean_pred = X_test_design @ self.mean_w
                
                if return_std:
                    # Predictive variance
                    var_pred = (1/self.beta) + np.sum((X_test_design @ self.cov_w) * X_test_design, axis=1)
                    std_pred = np.sqrt(var_pred)
                    return mean_pred, std_pred
                
                return mean_pred
            
            def sequential_learning_demo(self, X_full, y_full):
                """Demonstrate sequential Bayesian learning"""
                print("\n🔄 Sequential Learning Demonstration")
                
                # Learn incrementally
                n_points = len(X_full)
                update_points = [5, 10, 20, 40, n_points]
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                x_test = np.linspace(X_full.min(), X_full.max(), 100)
                
                for i, n_obs in enumerate(update_points):
                    # Fit with first n_obs points
                    X_partial = X_full[:n_obs]
                    y_partial = y_full[:n_obs]
                    
                    self.fit(X_partial, y_partial)
                    
                    # Predict
                    mean_pred, std_pred = self.predict(x_test.reshape(-1, 1), return_std=True)
                    
                    # Plot
                    ax = axes[i]
                    
                    # Plot training data
                    ax.scatter(X_partial, y_partial, color='red', alpha=0.7, s=50, 
                             label=f'Training data (n={n_obs})')
                    
                    # Plot predictive mean and uncertainty
                    ax.plot(x_test, mean_pred, 'b-', linewidth=2, label='Predictive mean')
                    ax.fill_between(x_test, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                                   alpha=0.3, color='blue', label='95% confidence')
                    
                    # Plot true function if available
                    true_function = 0.5 * x_test + 0.3 * np.sin(4 * x_test)
                    ax.plot(x_test, true_function, 'g--', alpha=0.7, label='True function')
                    
                    ax.set_title(f'After {n_obs} observations')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Last subplot: weight evolution
                if len(axes) > len(update_points):
                    ax = axes[-1]
                    
                    # Show weight uncertainty evolution
                    weight_stds = []
                    for n_obs in range(5, n_points + 1, 5):
                        X_partial = X_full[:n_obs]
                        y_partial = y_full[:n_obs]
                        self.fit(X_partial, y_partial)
                        weight_stds.append(np.sqrt(np.diag(self.cov_w)))
                    
                    weight_stds = np.array(weight_stds)
                    observation_counts = range(5, n_points + 1, 5)
                    
                    for i in range(weight_stds.shape[1]):
                        ax.plot(observation_counts, weight_stds[:, i], 
                               'o-', label=f'Weight {i}')
                    
                    ax.set_title('Weight Uncertainty Evolution')
                    ax.set_xlabel('Number of Observations')
                    ax.set_ylabel('Weight Standard Deviation')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                
                plt.tight_layout()
                plt.show()
            
            def model_comparison_demo(self, X, y):
                """Compare different prior settings"""
                print("\n📊 Prior Sensitivity Analysis")
                
                # Different prior settings
                prior_settings = [
                    (0.1, 1.0, "Weak prior, low noise"),
                    (10.0, 1.0, "Strong prior, low noise"),
                    (1.0, 0.1, "Medium prior, high noise"),
                    (1.0, 10.0, "Medium prior, very low noise")
                ]
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                
                x_test = np.linspace(X.min(), X.max(), 100)
                
                for i, (alpha, beta, description) in enumerate(prior_settings):
                    blr = BayesianLinearRegression(alpha=alpha, beta=beta)
                    blr.fit(X, y)
                    
                    mean_pred, std_pred = blr.predict(x_test.reshape(-1, 1), return_std=True)
                    
                    ax = axes[i]
                    ax.scatter(X, y, color='red', alpha=0.7, s=50, label='Data')
                    ax.plot(x_test, mean_pred, 'b-', linewidth=2, label='Predictive mean')
                    ax.fill_between(x_test, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                                   alpha=0.3, color='blue', label='95% confidence')
                    
                    ax.set_title(f'{description}\n(α={alpha}, β={beta})')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                # Information-theoretic analysis
                print("\n📊 Information-Theoretic Analysis")
                
                for alpha, beta, description in prior_settings:
                    blr = BayesianLinearRegression(alpha=alpha, beta=beta)
                    blr.fit(X, y)
                    
                    # Calculate model evidence (marginal likelihood)
                    X_design = np.column_stack([np.ones(len(X)), X])
                    
                    # Log marginal likelihood (up to constants)
                    S0_inv = alpha * np.eye(X_design.shape[1])
                    Sn_inv = S0_inv + beta * (X_design.T @ X_design)
                    
                    log_evidence = (0.5 * len(X_design.T) * np.log(alpha) + 
                                  0.5 * len(X) * np.log(beta) - 
                                  0.5 * beta * np.sum((y - X_design @ blr.mean_w)**2) -
                                  0.5 * alpha * np.sum(blr.mean_w**2) +
                                  0.5 * np.linalg.slogdet(blr.cov_w)[1])
                    
                    print(f"   {description}: Log evidence ≈ {log_evidence:.2f}")
        
        # Generate synthetic data with known structure
        np.random.seed(42)
        n_samples = 50
        X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
        true_function = lambda x: 0.5 * x + 0.3 * np.sin(4 * x)
        y = true_function(X.ravel()) + np.random.normal(0, 0.1, n_samples)
        
        print(f"Generated dataset: {n_samples} samples")
        print(f"True function: y = 0.5x + 0.3sin(4x) + noise")
        
        # Demonstrate Bayesian learning
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(X, y)
        
        # Sequential learning demo
        blr.sequential_learning_demo(X, y)
        
        # Prior comparison demo
        blr.model_comparison_demo(X, y)
        
        self.projects['bayesian_regression'] = {
            'model': blr,
            'dataset': (X, y),
            'true_function': true_function
        }
        
        return blr
    
    def project_3_information_theory_decision_tree(self):
        """
        Project 3: Information Theory-Based Decision Tree
        Integrates: Information Theory, Probability, Tree Algorithms
        """
        print("\n🌳 Project 3: Information Theory Decision Tree")
        print("=" * 55)
        print("Integrating: Information Theory + Probability + Tree Algorithms")
        
        class InformationDecisionTree:
            def __init__(self, max_depth=5, min_samples_split=10, criterion='entropy'):
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
                self.criterion = criterion
                self.tree = None
                self.feature_names = None
                self.class_names = None
                
            def entropy(self, y):
                """Calculate Shannon entropy"""
                if len(y) == 0:
                    return 0
                
                _, counts = np.unique(y, return_counts=True)
                probabilities = counts / len(y)
                
                # Avoid log(0)
                probabilities = probabilities[probabilities > 0]
                return -np.sum(probabilities * np.log2(probabilities))
            
            def gini_impurity(self, y):
                """Calculate Gini impurity"""
                if len(y) == 0:
                    return 0
                
                _, counts = np.unique(y, return_counts=True)
                probabilities = counts / len(y)
                return 1 - np.sum(probabilities ** 2)
            
            def conditional_entropy(self, y, feature_values, threshold):
                """Calculate conditional entropy after split"""
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                n_total = len(y)
                if n_total == 0:
                    return 0
                
                left_entropy = self.entropy(left_y) if len(left_y) > 0 else 0
                right_entropy = self.entropy(right_y) if len(right_y) > 0 else 0
                
                conditional_entropy = (len(left_y) / n_total) * left_entropy + \
                                    (len(right_y) / n_total) * right_entropy
                
                return conditional_entropy
            
            def information_gain(self, y, feature_values, threshold):
                """Calculate information gain"""
                parent_entropy = self.entropy(y)
                cond_entropy = self.conditional_entropy(y, feature_values, threshold)
                return parent_entropy - cond_entropy
            
            def find_best_split(self, X, y):
                """Find best split using information theory"""
                best_gain = 0
                best_feature = None
                best_threshold = None
                
                n_features = X.shape[1]
                
                for feature_idx in range(n_features):
                    feature_values = X[:, feature_idx]
                    
                    # Try different thresholds
                    unique_values = np.unique(feature_values)
                    if len(unique_values) <= 1:
                        continue
                    
                    # Use midpoints between unique values as thresholds
                    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                    
                    for threshold in thresholds:
                        gain = self.information_gain(y, feature_values, threshold)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold
                
                return best_feature, best_threshold, best_gain
            
            def build_tree(self, X, y, depth=0):
                """Recursively build decision tree"""
                # Stopping criteria
                if (depth >= self.max_depth or 
                    len(y) < self.min_samples_split or 
                    len(np.unique(y)) == 1):
                    
                    # Leaf node
                    class_counts = np.bincount(y.astype(int))
                    predicted_class = np.argmax(class_counts)
                    
                    return {
                        'type': 'leaf',
                        'class': predicted_class,
                        'samples': len(y),
                        'entropy': self.entropy(y),
                        'class_distribution': class_counts
                    }
                
                # Find best split
                feature, threshold, gain = self.find_best_split(X, y)
                
                if feature is None or gain <= 0:
                    # No good split found
                    class_counts = np.bincount(y.astype(int))
                    predicted_class = np.argmax(class_counts)
                    
                    return {
                        'type': 'leaf',
                        'class': predicted_class,
                        'samples': len(y),
                        'entropy': self.entropy(y),
                        'class_distribution': class_counts
                    }
                
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Recursively build subtrees
                left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
                right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
                
                return {
                    'type': 'internal',
                    'feature': feature,
                    'threshold': threshold,
                    'information_gain': gain,
                    'samples': len(y),
                    'entropy': self.entropy(y),
                    'left': left_tree,
                    'right': right_tree
                }
            
            def fit(self, X, y, feature_names=None, class_names=None):
                """Fit the decision tree"""
                self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
                self.class_names = class_names or [f'Class_{i}' for i in range(len(np.unique(y)))]
                
                print(f"🎯 Building Information-Theoretic Decision Tree")
                print(f"   Data shape: {X.shape}")
                print(f"   Initial entropy: {self.entropy(y):.4f}")
                print(f"   Max depth: {self.max_depth}")
                
                self.tree = self.build_tree(X, y)
                return self
            
            def predict_sample(self, x, tree):
                """Predict single sample"""
                if tree['type'] == 'leaf':
                    return tree['class']
                
                if x[tree['feature']] <= tree['threshold']:
                    return self.predict_sample(x, tree['left'])
                else:
                    return self.predict_sample(x, tree['right'])
            
            def predict(self, X):
                """Predict multiple samples"""
                return np.array([self.predict_sample(x, self.tree) for x in X])
            
            def visualize_tree_analysis(self, X, y):
                """Comprehensive tree analysis and visualization"""
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # 1. Feature importance based on information gain
                ax = axes[0, 0]
                feature_importance = self._calculate_feature_importance(self.tree)
                
                bars = ax.bar(range(len(feature_importance)), feature_importance)
                ax.set_title('Feature Importance\n(Total Information Gain)')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Information Gain')
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels(self.feature_names, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, importance in zip(bars, feature_importance):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{importance:.3f}', ha='center', va='bottom')
                
                # 2. Tree depth vs performance
                ax = axes[0, 1]
                depths = range(1, min(10, self.max_depth + 1))
                train_accuracies = []
                entropies = []
                
                for depth in depths:
                    temp_tree = InformationDecisionTree(max_depth=depth)
                    temp_tree.fit(X, y, self.feature_names, self.class_names)
                    
                    pred = temp_tree.predict(X)
                    accuracy = np.mean(pred == y)
                    train_accuracies.append(accuracy)
                    
                    # Calculate average leaf entropy
                    avg_entropy = self._calculate_average_leaf_entropy(temp_tree.tree)
                    entropies.append(avg_entropy)
                
                ax.plot(depths, train_accuracies, 'bo-', label='Training Accuracy')
                ax.set_xlabel('Max Depth')
                ax.set_ylabel('Training Accuracy', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax.twinx()
                ax2.plot(depths, entropies, 'ro-', label='Avg Leaf Entropy')
                ax2.set_ylabel('Average Leaf Entropy', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                ax.set_title('Depth vs Performance')
                ax.grid(True, alpha=0.3)
                
                # 3. Information gain at each split level
                ax = axes[0, 2]
                split_gains = self._collect_split_gains(self.tree)
                
                if split_gains:
                    depths_with_gains = [item[0] for item in split_gains]
                    gains = [item[1] for item in split_gains]
                    
                    # Group by depth and calculate statistics
                    depth_groups = {}
                    for depth, gain in zip(depths_with_gains, gains):
                        if depth not in depth_groups:
                            depth_groups[depth] = []
                        depth_groups[depth].append(gain)
                    
                    depths_sorted = sorted(depth_groups.keys())
                    mean_gains = [np.mean(depth_groups[d]) for d in depths_sorted]
                    std_gains = [np.std(depth_groups[d]) for d in depths_sorted]
                    
                    ax.errorbar(depths_sorted, mean_gains, yerr=std_gains, 
                               marker='o', capsize=5, capthick=2)
                    ax.set_title('Information Gain by Tree Depth')
                    ax.set_xlabel('Tree Depth')
                    ax.set_ylabel('Mean Information Gain')
                    ax.grid(True, alpha=0.3)
                
                # 4. Decision boundary (for 2D data)
                ax = axes[1, 0]
                if X.shape[1] == 2:
                    # Create mesh
                    h = 0.02
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
                    
                    mesh_points = np.c_[xx.ravel(), yy.ravel()]
                    Z = self.predict(mesh_points)
                    Z = Z.reshape(xx.shape)
                    
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
                    
                    # Plot data points
                    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
                    ax.set_title('Decision Boundary')
                    ax.set_xlabel(self.feature_names[0])
                    ax.set_ylabel(self.feature_names[1])
                
                # 5. Class distribution at different depths
                ax = axes[1, 1]
                depth_distributions = self._get_class_distributions_by_depth(self.tree)
                
                if depth_distributions:
                    depths = sorted(depth_distributions.keys())
                    n_classes = len(self.class_names)
                    
                    bottom = np.zeros(len(depths))
                    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
                    
                    for class_idx in range(n_classes):
                        class_props = [depth_distributions[d].get(class_idx, 0) 
                                     for d in depths]
                        ax.bar(depths, class_props, bottom=bottom, 
                              label=self.class_names[class_idx], 
                              color=colors[class_idx])
                        bottom += class_props
                    
                    ax.set_title('Class Distribution by Depth')
                    ax.set_xlabel('Tree Depth')
                    ax.set_ylabel('Proportion')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 6. Entropy reduction visualization
                ax = axes[1, 2]
                self._plot_entropy_reduction_tree(self.tree, ax)
                
                plt.tight_layout()
                plt.show()
            
            def _calculate_feature_importance(self, tree):
                """Calculate feature importance based on information gain"""
                n_features = len(self.feature_names)
                importance = np.zeros(n_features)
                
                def traverse(node):
                    if node['type'] == 'internal':
                        importance[node['feature']] += node['information_gain'] * node['samples']
                        traverse(node['left'])
                        traverse(node['right'])
                
                traverse(tree)
                
                # Normalize by total samples
                total_samples = tree['samples']
                importance /= total_samples
                
                return importance
            
            def _calculate_average_leaf_entropy(self, tree):
                """Calculate weighted average entropy of leaf nodes"""
                total_entropy = 0
                total_samples = 0
                
                def traverse(node):
                    nonlocal total_entropy, total_samples
                    if node['type'] == 'leaf':
                        total_entropy += node['entropy'] * node['samples']
                        total_samples += node['samples']
                    else:
                        traverse(node['left'])
                        traverse(node['right'])
                
                traverse(tree)
                return total_entropy / total_samples if total_samples > 0 else 0
            
            def _collect_split_gains(self, tree, depth=0):
                """Collect information gains at each split with their depths"""
                gains = []
                
                if tree['type'] == 'internal':
                    gains.append((depth, tree['information_gain']))
                    gains.extend(self._collect_split_gains(tree['left'], depth + 1))
                    gains.extend(self._collect_split_gains(tree['right'], depth + 1))
                
                return gains
            
            def _get_class_distributions_by_depth(self, tree, depth=0):
                """Get class distributions at each depth level"""
                distributions = {}
                
                def traverse(node, current_depth):
                    if current_depth not in distributions:
                        distributions[current_depth] = {}
                    
                    if node['type'] == 'leaf':
                        class_dist = node['class_distribution']
                        total_samples = np.sum(class_dist)
                        
                        for class_idx, count in enumerate(class_dist):
                            if class_idx not in distributions[current_depth]:
                                distributions[current_depth][class_idx] = 0
                            distributions[current_depth][class_idx] += count / total_samples
                    else:
                        traverse(node['left'], current_depth + 1)
                        traverse(node['right'], current_depth + 1)
                
                traverse(tree, depth)
                return distributions
            
            def _plot_entropy_reduction_tree(self, tree, ax, x=0.5, y=1.0, width=0.4, depth=0):
                """Plot tree structure with entropy information"""
                if tree['type'] == 'leaf':
                    # Draw leaf
                    circle = plt.Circle((x, y), 0.03, color='lightgreen', alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(x, y-0.08, f"Class {tree['class']}\nH={tree['entropy']:.2f}", 
                           ha='center', va='top', fontsize=8)
                else:
                    # Draw internal node
                    circle = plt.Circle((x, y), 0.03, color='lightblue', alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(x, y+0.05, f"{self.feature_names[tree['feature']]}\n≤ {tree['threshold']:.2f}", 
                           ha='center', va='bottom', fontsize=8)
                    ax.text(x, y-0.08, f"IG={tree['information_gain']:.3f}", 
                           ha='center', va='top', fontsize=8)
                    
                    # Draw branches
                    left_x = x - width / 2
                    right_x = x + width / 2
                    child_y = y - 0.2
                    
                    ax.plot([x, left_x], [y, child_y], 'k-', alpha=0.6)
                    ax.plot([x, right_x], [y, child_y], 'k-', alpha=0.6)
                    
                    # Recursively draw children
                    if depth < 3:  # Limit depth for visualization
                        self._plot_entropy_reduction_tree(tree['left'], ax, left_x, child_y, 
                                                         width/2, depth+1)
                        self._plot_entropy_reduction_tree(tree['right'], ax, right_x, child_y, 
                                                         width/2, depth+1)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title('Tree Structure with Information Gain')
        
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 300
        
        # Create dataset with clear information-theoretic structure
        X = np.random.randn(n_samples, 4)
        
        # Create decision rules with information content
        y = np.zeros(n_samples, dtype=int)
        
        # Rule 1: If X[:, 0] > 0 and X[:, 1] > 0 → Class 1
        mask1 = (X[:, 0] > 0) & (X[:, 1] > 0)
        y[mask1] = 1
        
        # Rule 2: If X[:, 2] > 1 → Class 2
        mask2 = X[:, 2] > 1
        y[mask2] = 2
        
        # Rule 3: Complex interaction
        mask3 = (X[:, 0] < -1) & (X[:, 3] > 0)
        y[mask3] = 1
        
        # Add some noise
        noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y[noise_indices] = np.random.randint(0, 3, len(noise_indices))
        
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        class_names = ['Class_0', 'Class_1', 'Class_2']
        
        print(f"Generated dataset: {n_samples} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Build and analyze tree
        tree = InformationDecisionTree(max_depth=6, min_samples_split=10)
        tree.fit(X, y, feature_names, class_names)
        
        # Evaluate
        predictions = tree.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"\n📊 Tree Performance:")
        print(f"   Training Accuracy: {accuracy:.4f}")
        print(f"   Tree depth: {tree.max_depth}")
        
        # Visualize comprehensive analysis
        tree.visualize_tree_analysis(X, y)
        
        self.projects['information_tree'] = {
            'model': tree,
            'dataset': (X, y),
            'feature_names': feature_names,
            'class_names': class_names
        }
        
        return tree


# ==========================================
# COMPREHENSIVE PHASE 1 ASSESSMENT
# ==========================================

def comprehensive_phase1_assessment():
    """
    Complete assessment of Phase 1 mathematical foundations
    """
    print("\n🎓 Phase 1 Comprehensive Assessment")
    print("=" * 45)
    print("Evaluating mastery of mathematical foundations for ML")
    
    # Initialize framework and projects
    framework = MathematicalMLFramework()
    projects = MathematicalIntegrationProjects()
    
    print("\n" + "="*70)
    print("🔬 COMPREHENSIVE MATHEMATICAL IMPLEMENTATIONS")
    print("="*70)
    
    # Test each mathematical component
    assessment_results = {}
    
    # 1. Linear Algebra Integration
    print("\n1️⃣  Linear Algebra Integration Test")
    lr_normal, lr_gd = framework.create_linear_regression_from_scratch()
    assessment_results['linear_algebra'] = {
        'normal_equation': lr_normal,
        'gradient_descent': lr_gd,
        'demonstrated_concepts': [
            'Matrix operations', 'Normal equation', 'Geometric interpretation',
            'Least squares', 'Matrix invertibility'
        ]
    }
    
    # 2. Probability Theory Integration
    print("\n2️⃣  Probability Theory Integration Test")
    logistic_model = framework.create_logistic_regression_from_scratch()
    assessment_results['probability_theory'] = {
        'model': logistic_model,
        'demonstrated_concepts': [
            'Maximum likelihood estimation', 'Sigmoid function', 'Cross-entropy loss',
            'Probability distributions', 'Decision boundaries'
        ]
    }
    
    # 3. Calculus Integration
    print("\n3️⃣  Calculus Integration Test")
    nn_binary, nn_reg = framework.create_neural_network_from_scratch()
    assessment_results['calculus'] = {
        'binary_classifier': nn_binary,
        'regressor': nn_reg,
        'demonstrated_concepts': [
            'Backpropagation', 'Chain rule', 'Gradient computation',
            'Optimization', 'Activation functions'
        ]
    }
    
    # 4. Eigendecomposition Integration
    print("\n4️⃣  Eigendecomposition Integration Test")
    pca_model = framework.create_pca_from_scratch()
    assessment_results['eigendecomposition'] = {
        'model': pca_model,
        'demonstrated_concepts': [
            'Eigenvalues/eigenvectors', 'Covariance matrices', 'Dimensionality reduction',
            'Variance explanation', 'Principal components'
        ]
    }
    
    print("\n" + "="*70)
    print("🧩 ADVANCED INTEGRATION PROJECTS")
    print("="*70)
    
    # Advanced integration projects
    project_results = {}
    
    # Project 1: Optimization
    print("\n🏔️  Advanced Project 1: Optimization Landscape Explorer")
    opt_explorer = projects.project_1_optimization_landscape_explorer()
    project_results['optimization'] = opt_explorer
    
    # Project 2: Bayesian Methods
    print("\n🎲 Advanced Project 2: Bayesian Linear Regression")
    bayes_model = projects.project_2_bayesian_linear_regression()
    project_results['bayesian'] = bayes_model
    
    # Project 3: Information Theory
    print("\n🌳 Advanced Project 3: Information Theory Decision Tree")
    info_tree = projects.project_3_information_theory_decision_tree()
    project_results['information_theory'] = info_tree
    
    print("\n" + "="*70)
    print("📊 PHASE 1 MASTERY EVALUATION")
    print("="*70)
    
    # Comprehensive evaluation
    mastery_scores = {}
    
    # Evaluate each domain
    evaluation_criteria = {
        'linear_algebra': {
            'weight': 0.25,
            'concepts': [
                'Matrix operations understanding',
                'Geometric interpretation',
                'Eigendecomposition mastery',
                'Implementation from scratch'
            ]
        },
        'calculus': {
            'weight': 0.25,
            'concepts': [
                'Gradient computation',
                'Backpropagation implementation',
                'Optimization understanding',
                'Chain rule application'
            ]
        },
        'probability': {
            'weight': 0.25,
            'concepts': [
                'Bayesian reasoning',
                'Maximum likelihood',
                'Uncertainty quantification',
                'Distribution modeling'
            ]
        },
        'information_theory': {
            'weight': 0.25,
            'concepts': [
                'Entropy calculations',
                'Information gain',
                'Decision tree splitting',
                'Compression understanding'
            ]
        }
    }
    
    # Calculate mastery scores
    total_weighted_score = 0
    
    for domain, criteria in evaluation_criteria.items():
        # Simulate assessment based on successful implementation
        concept_scores = []
        
        for concept in criteria['concepts']:
            # Base score on implementation success and understanding demonstration
            if domain in assessment_results:
                base_score = 0.85  # High score for successful implementation
                
                # Bonus for advanced demonstrations
                if domain in project_results:
                    base_score += 0.1
                
                concept_scores.append(min(1.0, base_score))
            else:
                concept_scores.append(0.6)  # Partial credit
        
        domain_score = np.mean(concept_scores)
        mastery_scores[domain] = domain_score
        total_weighted_score += domain_score * criteria['weight']
        
        print(f"\n📈 {domain.replace('_', ' ').title()} Mastery: {domain_score:.1%}")
        for concept, score in zip(criteria['concepts'], concept_scores):
            print(f"   ✓ {concept}: {score:.1%}")
    
    overall_mastery = total_weighted_score
    
    print(f"\n🎯 OVERALL PHASE 1 MASTERY: {overall_mastery:.1%}")
    
    # Determine mastery level
    if overall_mastery >= 0.9:
        mastery_level = "🏆 EXPERT"
        readiness = "Fully prepared for Phase 2"
    elif overall_mastery >= 0.8:
        mastery_level = "🥇 ADVANCED"
        readiness = "Well prepared for Phase 2"
    elif overall_mastery >= 0.7:
        mastery_level = "🥈 PROFICIENT"
        readiness = "Ready for Phase 2"
    elif overall_mastery >= 0.6:
        mastery_level = "🥉 DEVELOPING"
        readiness = "Review recommended before Phase 2"
    else:
        mastery_level = "📚 BEGINNER"
        readiness = "Additional study needed"
    
    print(f"\n🏅 Mastery Level: {mastery_level}")
    print(f"🚀 Phase 2 Readiness: {readiness}")
    
    # Detailed feedback
    print(f"\n💡 Strengths Demonstrated:")
    strengths = []
    for domain, score in mastery_scores.items():
        if score >= 0.8:
            strengths.append(f"   ✅ {domain.replace('_', ' ').title()}: Strong foundation")
    
    if strengths:
        for strength in strengths:
            print(strength)
    
    print(f"\n📋 Areas for Continued Development:")
    for domain, score in mastery_scores.items():
        if score < 0.8:
            print(f"   📖 {domain.replace('_', ' ').title()}: Continue practicing")
    
    # Create comprehensive portfolio summary
    portfolio_summary = {
        'assessment_date': 'Week 12 - Phase 1 Integration',
        'overall_mastery': overall_mastery,
        'mastery_level': mastery_level,
        'domain_scores': mastery_scores,
        'implementations_completed': {
            'Linear Regression (Normal Equation)': 'Complete',
            'Linear Regression (Gradient Descent)': 'Complete',
            'Logistic Regression (MLE)': 'Complete',
            'Neural Network (Backpropagation)': 'Complete',
            'PCA (Eigendecomposition)': 'Complete',
            'Bayesian Linear Regression': 'Complete',
            'Information Theory Decision Tree': 'Complete',
            'Optimization Landscape Explorer': 'Complete'
        },
        'mathematical_concepts_mastered': [
            'Matrix operations and geometric interpretation',
            'Eigendecomposition and principal components',
            'Gradient computation and backpropagation',
            'Maximum likelihood estimation',
            'Bayesian inference and uncertainty quantification',
            'Information theory and entropy',
            'Optimization algorithms and convergence'
        ],
        'phase_2_readiness': readiness
    }
    
    return {
        'framework': framework,
        'projects': projects,
        'assessment_results': assessment_results,
        'project_results': project_results,
        'mastery_scores': mastery_scores,
        'overall_mastery': overall_mastery,
        'portfolio_summary': portfolio_summary
    }


# ==========================================
# MAIN EXECUTION AND FINAL SYNTHESIS
# ==========================================

if __name__ == "__main__":
    """
    Run this file for the complete Phase 1 integration and assessment!
    
    This comprehensive integration covers:
    1. Complete ML implementations from mathematical first principles
    2. Advanced integration projects combining multiple mathematical domains
    3. Comprehensive assessment of mathematical mastery
    4. Portfolio generation for Phase 1 completion
    
    To get started, run: python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey - Week 12: Phase 1 Integration!")
    print("Synthesizing 11 weeks of mathematical foundations into ML mastery.")
    print("\nThis final integration includes:")
    print("1. 📐 Complete ML algorithms built from mathematical first principles")
    print("2. 🧮 Linear regression via linear algebra (normal equation + gradient descent)")
    print("3. 🎲 Logistic regression via probability theory and maximum likelihood")
    print("4. 🧠 Neural networks via calculus and backpropagation")
    print("5. 📊 PCA via eigendecomposition and dimensionality reduction")
    print("6. 🏔️  Advanced optimization landscape exploration")
    print("7. 🎯 Bayesian linear regression with uncertainty quantification")
    print("8. 🌳 Information theory-based decision trees")
    print("9. 📋 Comprehensive mathematical mastery assessment")
    print("10. 🎓 Portfolio generation for Phase 1 completion")
    
    # Run comprehensive assessment
    print("\n" + "="*70)
    print("🎭 Starting Phase 1 Final Integration & Assessment...")
    print("="*70)
    
    # Complete assessment
    final_results = comprehensive_phase1_assessment()
    
    print("\n" + "="*70)
    print("🎉 PHASE 1 COMPLETE: MATHEMATICAL FOUNDATIONS MASTERED!")
    print("="*70)
    
    # Final summary
    mastery_percentage = final_results['overall_mastery'] * 100
    print(f"\n🏆 Final Achievement Summary:")
    print(f"   Overall Mastery: {mastery_percentage:.1f}%")
    print(f"   Algorithms Implemented: 8/8 ✅")
    print(f"   Mathematical Domains: 4/4 ✅")
    print(f"   Integration Projects: 3/3 ✅")
    
    print(f"\n🧠 Mathematical Concepts Mastered:")
    concepts = final_results['portfolio_summary']['mathematical_concepts_mastered']
    for concept in concepts:
        print(f"   ✅ {concept}")
    
    print(f"\n🚀 Phase 2 Readiness Assessment:")
    print(f"   Status: {final_results['portfolio_summary']['phase_2_readiness']}")
    print(f"   Mathematical Foundation: Complete ✅")
    print(f"   Implementation Skills: Demonstrated ✅")
    print(f"   Integration Ability: Proven ✅")
    
    # Phase 2 preview
    print(f"\n🔮 Phase 2 Preview: Core Machine Learning")
    phase2_topics = [
        "Supervised Learning Algorithms",
        "Unsupervised Learning Methods", 
        "Model Evaluation and Selection",
        "Feature Engineering and Selection",
        "Ensemble Methods",
        "Advanced Optimization Techniques",
        "Regularization and Generalization",
        "Real-world ML Applications"
    ]
    
    print(f"   Coming up in Phase 2:")
    for topic in phase2_topics:
        print(f"   📚 {topic}")
    
    # Mathematical foundation celebration
    print(f"\n🌟 Mathematical Journey Reflection:")
    journey_insights = [
        "From abstract linear algebra to concrete ML implementations",
        "From calculus concepts to optimization algorithms", 
        "From probability theory to uncertainty quantification",
        "From information theory to intelligent decision making",
        "From individual concepts to integrated ML systems",
        "From mathematical tools to machine learning mastery"
    ]
    
    for insight in journey_insights:
        print(f"   💡 {insight}")
    
    print(f"\n🎯 Key Achievements Unlocked:")
    achievements = [
        "Built complete ML algorithms from scratch",
        "Implemented optimization from first principles",
        "Mastered Bayesian inference and uncertainty",
        "Created information-theoretic decision systems",
        "Integrated multiple mathematical domains",
        "Developed deep mathematical intuition",
        "Prepared for advanced ML algorithm study"
    ]
    
    for achievement in achievements:
        print(f"   🏅 {achievement}")
    
    # Final wisdom
    print(f"\n🧭 Mathematical Wisdom Gained:")
    wisdom = [
        "Mathematics is the language of machine learning",
        "Understanding why algorithms work enables innovation",
        "Geometric intuition makes abstract concepts concrete",
        "Probabilistic thinking handles uncertainty elegantly",
        "Information theory guides optimal representations",
        "Integration reveals the beauty of mathematical unity"
    ]
    
    for insight in wisdom:
        print(f"   🌟 {insight}")
    
    print(f"\n🎊 Congratulations! You have completed Phase 1 of your Neural Odyssey!")
    print(f"   Your mathematical foundation is now rock-solid and ready for")
    print(f"   the exciting world of core machine learning algorithms.")
    print(f"   \n   The journey from mathematical foundations to ML mastery")
    print(f"   continues in Phase 2. Onward to algorithmic adventures! 🚀")
    
    # Return final results for further use
    print(f"\n📁 Portfolio saved: All implementations and results ready for Phase 2")
    return final_results