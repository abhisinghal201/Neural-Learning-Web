"""
Neural Odyssey - Week 7: Neural Network Foundations
Exercises for building neural networks from mathematical first principles

This module implements core concepts that power modern AI:
- Artificial neurons and activation functions
- Forward propagation through networks
- Backpropagation and automatic differentiation
- Neural network architectures and design principles

Complete the TODO functions to build your neural network toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: ARTIFICIAL NEURONS AND ACTIVATIONS
# ==============================================

def artificial_neuron(inputs, weights, bias, activation_func):
    """
    TODO: Implement a single artificial neuron
    
    A neuron computes: activation(w·x + b)
    This is the fundamental building block of all neural networks.
    
    Args:
        inputs: Input vector
        weights: Weight vector (same length as inputs)
        bias: Bias term (scalar)
        activation_func: Activation function to apply
        
    Returns:
        Neuron output after applying activation function
    """
    # TODO: Implement neuron computation
    # 1. Compute weighted sum: z = w·x + b
    # 2. Apply activation function: output = activation(z)
    
    pass

def sigmoid_activation(z):
    """
    TODO: Implement sigmoid activation function
    
    σ(z) = 1 / (1 + exp(-z))
    
    Sigmoid squashes any real number to (0,1), making it useful for
    probability outputs and historically important for neural networks.
    
    Args:
        z: Input value(s)
        
    Returns:
        Sigmoid output(s)
    """
    # TODO: Implement sigmoid function
    # Handle numerical stability for large positive/negative values
    
    pass

def sigmoid_derivative(z):
    """
    TODO: Implement derivative of sigmoid function
    
    σ'(z) = σ(z) * (1 - σ(z))
    
    This elegant property makes sigmoid convenient for backpropagation.
    
    Args:
        z: Input value(s)
        
    Returns:
        Derivative of sigmoid at z
    """
    # TODO: Implement sigmoid derivative
    # Use the fact that if s = σ(z), then σ'(z) = s * (1 - s)
    
    pass

def relu_activation(z):
    """
    TODO: Implement ReLU activation function
    
    ReLU(z) = max(0, z)
    
    ReLU is the most popular activation in modern deep learning
    due to its simplicity and effectiveness.
    
    Args:
        z: Input value(s)
        
    Returns:
        ReLU output(s)
    """
    # TODO: Implement ReLU function
    # Simply return max(0, z) for each element
    
    pass

def relu_derivative(z):
    """
    TODO: Implement derivative of ReLU function
    
    ReLU'(z) = 1 if z > 0, 0 if z ≤ 0
    
    The derivative is a step function, which can cause issues
    with "dead neurons" but works well in practice.
    
    Args:
        z: Input value(s)
        
    Returns:
        Derivative of ReLU at z
    """
    # TODO: Implement ReLU derivative
    # Return 1 where z > 0, 0 elsewhere
    
    pass

def tanh_activation(z):
    """
    TODO: Implement hyperbolic tangent activation
    
    tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    
    Tanh outputs in range (-1, 1) and is zero-centered,
    which can be beneficial for optimization.
    
    Args:
        z: Input value(s)
        
    Returns:
        Tanh output(s)
    """
    # TODO: Implement tanh function
    # Can use np.tanh or implement from scratch
    
    pass

def compare_activation_functions():
    """
    TODO: Visualize and compare different activation functions
    
    Shows the behavior and derivatives of common activations.
    This helps understand why different activations are used.
    """
    # TODO: Create visualization showing:
    # 1. Sigmoid, ReLU, Tanh functions
    # 2. Their derivatives
    # 3. Discussion of pros/cons for each
    
    pass

# ==============================================
# PART 2: FORWARD PROPAGATION
# ==============================================

def linear_layer_forward(X, W, b):
    """
    TODO: Implement forward pass through a linear layer
    
    Z = XW + b
    
    This is the core computation in neural networks - a linear transformation
    followed by bias addition.
    
    Args:
        X: Input matrix (batch_size, input_dim)
        W: Weight matrix (input_dim, output_dim)
        b: Bias vector (output_dim,)
        
    Returns:
        Z: Linear output (batch_size, output_dim)
    """
    # TODO: Implement linear layer forward pass
    # Handle broadcasting for bias addition
    
    pass

def activation_forward(Z, activation='relu'):
    """
    TODO: Apply activation function element-wise
    
    Different activations serve different purposes in the network.
    
    Args:
        Z: Linear layer output
        activation: Type of activation ('relu', 'sigmoid', 'tanh')
        
    Returns:
        A: Activated output
    """
    # TODO: Apply the specified activation function
    # Support multiple activation types
    
    pass

def neural_network_forward(X, layers):
    """
    TODO: Implement forward propagation through entire network
    
    Chains together linear transformations and activations.
    This is how neural networks transform inputs to outputs.
    
    Args:
        X: Input data (batch_size, input_dim)
        layers: List of layer parameters [(W1, b1, activation1), ...]
        
    Returns:
        outputs: Final network output
        activations: Intermediate activations for backprop
    """
    # TODO: Implement multi-layer forward propagation
    # Store intermediate values needed for backpropagation
    # Return both final output and intermediate activations
    
    pass

def softmax_activation(z):
    """
    TODO: Implement softmax activation for multi-class classification
    
    softmax(z_i) = exp(z_i) / Σ exp(z_j)
    
    Softmax converts logits to probability distribution.
    Essential for multi-class classification.
    
    Args:
        z: Logits (batch_size, num_classes)
        
    Returns:
        Probability distribution over classes
    """
    # TODO: Implement numerically stable softmax
    # Subtract max for numerical stability: exp(z - max(z))
    
    pass

def predict_classification(X, layers):
    """
    TODO: Make classification predictions using trained network
    
    Args:
        X: Input data
        layers: Trained network parameters
        
    Returns:
        Class predictions and probabilities
    """
    # TODO: Forward propagate and apply softmax for classification
    
    pass

# ==============================================
# PART 3: LOSS FUNCTIONS
# ==============================================

def mean_squared_error(y_true, y_pred):
    """
    TODO: Implement mean squared error loss
    
    MSE = (1/n) * Σ (y_true - y_pred)²
    
    Used for regression problems.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        MSE loss value
    """
    # TODO: Implement MSE loss
    # Average over all samples and dimensions
    
    pass

def mse_gradient(y_true, y_pred):
    """
    TODO: Compute gradient of MSE loss
    
    ∂MSE/∂y_pred = (2/n) * (y_pred - y_true)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Gradient with respect to predictions
    """
    # TODO: Implement MSE gradient
    
    pass

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    TODO: Implement cross-entropy loss for classification
    
    CE = -Σ y_true * log(y_pred)
    
    This is the standard loss for classification problems.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        epsilon: Small value to prevent log(0)
        
    Returns:
        Cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    # Add epsilon to prevent numerical issues
    # Handle both binary and multi-class cases
    
    pass

def cross_entropy_gradient(y_true, y_pred):
    """
    TODO: Compute gradient of cross-entropy loss
    
    For softmax + cross-entropy: ∂CE/∂z = y_pred - y_true
    
    This simple gradient is why softmax + cross-entropy work so well together.
    
    Args:
        y_true: True labels (one-hot)
        y_pred: Predicted probabilities
        
    Returns:
        Gradient with respect to logits
    """
    # TODO: Implement cross-entropy gradient
    # This assumes softmax was used to get y_pred
    
    pass

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    TODO: Implement binary cross-entropy loss
    
    BCE = -y*log(p) - (1-y)*log(1-p)
    
    Specialized version for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        epsilon: Numerical stability
        
    Returns:
        Binary cross-entropy loss
    """
    # TODO: Implement binary cross-entropy
    
    pass

# ==============================================
# PART 4: BACKPROPAGATION
# ==============================================

def linear_layer_backward(dZ, X, W):
    """
    TODO: Implement backward pass through linear layer
    
    Given gradient w.r.t. output, compute gradients w.r.t. inputs and parameters.
    
    Args:
        dZ: Gradient w.r.t. layer output (batch_size, output_dim)
        X: Layer input (batch_size, input_dim)
        W: Layer weights (input_dim, output_dim)
        
    Returns:
        dX: Gradient w.r.t. input
        dW: Gradient w.r.t. weights
        db: Gradient w.r.t. bias
    """
    # TODO: Implement linear layer backward pass
    # dX = dZ @ W.T
    # dW = X.T @ dZ
    # db = sum(dZ, axis=0)
    
    pass

def activation_backward(dA, Z, activation='relu'):
    """
    TODO: Implement backward pass through activation function
    
    Chain rule: dZ = dA * activation'(Z)
    
    Args:
        dA: Gradient w.r.t. activation output
        Z: Pre-activation values
        activation: Type of activation function
        
    Returns:
        dZ: Gradient w.r.t. pre-activation
    """
    # TODO: Compute activation derivative and apply chain rule
    
    pass

def neural_network_backward(X, y, layers, activations, loss_type='mse'):
    """
    TODO: Implement full backpropagation algorithm
    
    Computes gradients for all layers using the chain rule.
    This is the heart of neural network training.
    
    Args:
        X: Input data
        y: True targets
        layers: Network parameters [(W1, b1, activation1), ...]
        activations: Stored forward pass activations
        loss_type: Type of loss function used
        
    Returns:
        gradients: List of gradients [(dW1, db1), (dW2, db2), ...]
    """
    # TODO: Implement complete backpropagation
    # 1. Compute loss gradient
    # 2. Backpropagate through each layer
    # 3. Return gradients for all parameters
    
    pass

def gradient_check(X, y, layers, epsilon=1e-7):
    """
    TODO: Verify backpropagation using numerical gradients
    
    Essential for debugging neural network implementations.
    Compare analytical gradients with numerical approximations.
    
    Args:
        X: Input data
        y: Target values
        layers: Network parameters
        epsilon: Small value for finite differences
        
    Returns:
        Boolean indicating if gradients are correct
    """
    # TODO: Implement gradient checking
    # 1. Compute analytical gradients using backprop
    # 2. Compute numerical gradients using finite differences
    # 3. Compare and check if they match within tolerance
    
    pass

# ==============================================
# PART 5: TRAINING AND OPTIMIZATION
# ==============================================

def initialize_weights(layer_sizes, initialization='xavier'):
    """
    TODO: Initialize neural network weights
    
    Proper initialization is crucial for training success.
    Different methods work better for different activations.
    
    Args:
        layer_sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        initialization: Initialization scheme ('xavier', 'he', 'random')
        
    Returns:
        List of initialized weight matrices and bias vectors
    """
    # TODO: Implement different initialization schemes
    # Xavier/Glorot: good for sigmoid/tanh
    # He initialization: good for ReLU
    # Random: baseline comparison
    
    pass

def train_neural_network(X, y, layer_sizes, activations, learning_rate=0.01, 
                        epochs=100, batch_size=32, loss_type='mse', 
                        initialization='xavier', verbose=True):
    """
    TODO: Complete neural network training loop
    
    Implements the full training procedure:
    1. Initialize weights
    2. For each epoch:
       - Forward propagation
       - Compute loss
       - Backpropagation
       - Update weights
    
    Args:
        X: Training data
        y: Training targets
        layer_sizes: Network architecture
        activations: Activation functions for each layer
        learning_rate: Step size for gradient descent
        epochs: Number of training iterations
        batch_size: Size of mini-batches
        loss_type: Type of loss function
        initialization: Weight initialization scheme
        verbose: Whether to print training progress
        
    Returns:
        Trained network and training history
    """
    # TODO: Implement complete training loop
    # 1. Initialize network weights
    # 2. Training loop with mini-batches
    # 3. Track loss and accuracy over time
    # 4. Return trained model and metrics
    
    pass

def evaluate_network(X, y, layers, loss_type='mse'):
    """
    TODO: Evaluate trained network performance
    
    Args:
        X: Test data
        y: Test targets
        layers: Trained network parameters
        loss_type: Type of loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    # TODO: Compute relevant metrics:
    # For regression: MSE, MAE, R²
    # For classification: accuracy, cross-entropy loss
    
    pass

def learning_curve_analysis(X_train, y_train, X_val, y_val, layer_sizes, 
                          activations, epochs=100):
    """
    TODO: Analyze learning curves for overfitting detection
    
    Plots training and validation loss over time.
    Essential for understanding model behavior.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        layer_sizes: Network architecture
        activations: Activation functions
        epochs: Number of training epochs
        
    Returns:
        Learning curves and overfitting analysis
    """
    # TODO: Train network while tracking both training and validation metrics
    # Plot learning curves and identify overfitting
    
    pass

# ==============================================
# PART 6: NEURAL NETWORK ARCHITECTURES
# ==============================================

def create_mlp_classifier(input_dim, hidden_dims, num_classes, activation='relu'):
    """
    TODO: Create multi-layer perceptron for classification
    
    Standard feedforward architecture for classification tasks.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        activation: Hidden layer activation function
        
    Returns:
        Network layer specifications
    """
    # TODO: Create MLP architecture
    # Hidden layers use specified activation
    # Output layer uses softmax for multi-class or sigmoid for binary
    
    pass

def create_mlp_regressor(input_dim, hidden_dims, output_dim, activation='relu'):
    """
    TODO: Create multi-layer perceptron for regression
    
    Standard feedforward architecture for regression tasks.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Hidden layer activation function
        
    Returns:
        Network layer specifications
    """
    # TODO: Create MLP architecture for regression
    # Hidden layers use specified activation
    # Output layer uses linear activation (no activation)
    
    pass

def universal_approximation_demo():
    """
    TODO: Demonstrate universal approximation theorem
    
    Shows that neural networks can approximate any continuous function
    with sufficient hidden units.
    
    Returns:
        Demonstration of function approximation capabilities
    """
    # TODO: Create demonstration showing:
    # 1. Complex target function (e.g., sine wave, polynomial)
    # 2. Neural network approximation with different numbers of hidden units
    # 3. How approximation improves with more neurons
    
    pass

# ==============================================
# PART 7: REGULARIZATION AND IMPROVEMENTS
# ==============================================

def add_l2_regularization(loss, weights, lambda_reg):
    """
    TODO: Add L2 regularization to loss function
    
    L2_loss = original_loss + λ * Σ ||w||²
    
    Regularization prevents overfitting by penalizing large weights.
    
    Args:
        loss: Original loss value
        weights: List of weight matrices
        lambda_reg: Regularization strength
        
    Returns:
        Regularized loss
    """
    # TODO: Add L2 penalty to loss
    # Sum of squared weights across all layers
    
    pass

def l2_regularization_gradient(weights, lambda_reg):
    """
    TODO: Compute gradient of L2 regularization term
    
    ∂(λ||w||²)/∂w = 2λw
    
    Args:
        weights: Weight matrix
        lambda_reg: Regularization strength
        
    Returns:
        Regularization gradient
    """
    # TODO: Compute L2 regularization gradient
    
    pass

def dropout_forward(X, dropout_rate=0.5, training=True):
    """
    TODO: Implement dropout during forward pass
    
    Dropout randomly sets neurons to zero during training
    to prevent co-adaptation and overfitting.
    
    Args:
        X: Layer input
        dropout_rate: Fraction of neurons to drop
        training: Whether in training mode
        
    Returns:
        Dropout output and mask for backprop
    """
    # TODO: Implement dropout
    # During training: randomly zero out neurons and scale remaining
    # During inference: use all neurons without scaling
    
    pass

def dropout_backward(dA, dropout_mask, dropout_rate):
    """
    TODO: Implement dropout backward pass
    
    Apply the same mask used in forward pass.
    
    Args:
        dA: Gradient from next layer
        dropout_mask: Mask from forward pass
        dropout_rate: Dropout rate used
        
    Returns:
        Gradient after applying dropout mask
    """
    # TODO: Apply dropout mask to gradients
    
    pass

def batch_normalization_forward(X, gamma, beta, moving_mean=None, moving_var=None, 
                               training=True, momentum=0.9, epsilon=1e-8):
    """
    TODO: Implement batch normalization forward pass
    
    BatchNorm normalizes inputs to have zero mean and unit variance,
    then applies learnable scaling and shifting.
    
    Args:
        X: Input to normalize
        gamma: Scale parameter
        beta: Shift parameter
        moving_mean: Running average of mean (for inference)
        moving_var: Running average of variance (for inference)
        training: Whether in training mode
        momentum: Momentum for running averages
        epsilon: Small value for numerical stability
        
    Returns:
        Normalized output and cache for backward pass
    """
    # TODO: Implement batch normalization
    # Training: use batch statistics, update running averages
    # Inference: use running averages
    
    pass

# ==============================================
# PART 8: COMPREHENSIVE NEURAL NETWORK CLASS
# ==============================================

class NeuralNetwork:
    """
    TODO: Build comprehensive neural network class
    
    This class should encapsulate all neural network functionality:
    - Forward and backward propagation
    - Training with various optimizers
    - Regularization techniques
    - Model evaluation and analysis
    """
    
    def __init__(self, layer_sizes, activations, loss_type='mse', 
                 regularization=None, dropout_rate=0.0):
        """
        TODO: Initialize neural network
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_type = loss_type
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.layers = None
        self.training_history = {}
    
    def initialize_parameters(self, initialization='xavier'):
        """
        TODO: Initialize network parameters
        """
        # TODO: Initialize weights and biases for all layers
        
        pass
    
    def forward(self, X, training=True):
        """
        TODO: Forward propagation through network
        """
        # TODO: Implement forward pass with dropout and batch norm if enabled
        
        pass
    
    def backward(self, X, y):
        """
        TODO: Backward propagation through network
        """
        # TODO: Implement backpropagation with regularization
        
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, learning_rate=0.01, 
              optimizer='sgd', verbose=True):
        """
        TODO: Train the neural network
        """
        # TODO: Implement complete training procedure
        # Support different optimizers (SGD, Adam, etc.)
        # Track training and validation metrics
        
        pass
    
    def predict(self, X):
        """
        TODO: Make predictions using trained network
        """
        # TODO: Forward pass in inference mode
        
        pass
    
    def evaluate(self, X, y):
        """
        TODO: Evaluate network performance
        """
        # TODO: Compute relevant metrics for the task
        
        pass
    
    def plot_training_history(self):
        """
        TODO: Visualize training progress
        """
        # TODO: Plot loss and accuracy curves
        
        pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_neural_fundamentals():
    """Demonstrate fundamental neural network concepts."""
    
    print("🧠 Neural Network Fundamentals")
    print("=" * 50)
    
    try:
        # Test individual neuron
        inputs = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, -0.3, 0.8])
        bias = 0.1
        
        output = artificial_neuron(inputs, weights, bias, sigmoid_activation)
        print(f"Single neuron output: {output:.4f}")
        
        # Compare activation functions
        z_values = np.linspace(-5, 5, 100)
        
        print("Testing activation functions...")
        sigmoid_vals = sigmoid_activation(z_values)
        relu_vals = relu_activation(z_values)
        tanh_vals = tanh_activation(z_values)
        
        print("✅ Activation functions implemented")
        
        # Visualize activations
        compare_activation_functions()
        print("✅ Activation function comparison created")
        
    except Exception as e:
        print(f"❌ Neural fundamentals demo failed: {e}")
        print("Implement the basic neural network functions!")

def demonstrate_forward_propagation():
    """Demonstrate forward propagation through networks."""
    
    print("\n➡️ Forward Propagation Demonstration")
    print("=" * 50)
    
    try:
        # Create simple network
        np.random.seed(42)
        X = np.random.randn(5, 3)  # 5 samples, 3 features
        
        # Define network layers
        layers = [
            (np.random.randn(3, 4), np.random.randn(4), 'relu'),    # Hidden layer
            (np.random.randn(4, 2), np.random.randn(2), 'sigmoid')  # Output layer
        ]
        
        # Forward propagation
        output, activations = neural_network_forward(X, layers)
        print(f"Network output shape: {output.shape}")
        print(f"Number of stored activations: {len(activations)}")
        
        # Test softmax
        logits = np.random.randn(3, 5)  # 3 samples, 5 classes
        probabilities = softmax_activation(logits)
        print(f"Softmax output sums: {np.sum(probabilities, axis=1)}")  # Should be [1, 1, 1]
        
        print("✅ Forward propagation working correctly")
        
    except Exception as e:
        print(f"❌ Forward propagation demo failed: {e}")
        print("Implement the forward propagation functions!")

def demonstrate_backpropagation():
    """Demonstrate backpropagation algorithm."""
    
    print("\n⬅️ Backpropagation Demonstration")
    print("=" * 50)
    
    try:
        # Create small dataset
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 2)
        
        # Create simple network
        layers = [
            (np.random.randn(3, 4) * 0.1, np.random.randn(4) * 0.1, 'relu'),
            (np.random.randn(4, 2) * 0.1, np.random.randn(2) * 0.1, 'linear')
        ]
        
        # Forward pass
        output, activations = neural_network_forward(X, layers)
        
        # Backward pass
        gradients = neural_network_backward(X, y, layers, activations, 'mse')
        print(f"Number of gradient sets: {len(gradients)}")
        
        # Gradient checking
        gradient_correct = gradient_check(X, y, layers)
        print(f"Gradient check passed: {gradient_correct}")
        
        print("✅ Backpropagation implemented correctly")
        
    except Exception as e:
        print(f"❌ Backpropagation demo failed: {e}")
        print("Implement the backpropagation functions!")

def demonstrate_neural_network_training():
    """Demonstrate complete neural network training."""
    
    print("\n🏋️ Neural Network Training Demonstration")
    print("=" * 50)
    
    try:
        # Create classification dataset
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                                 n_informative=8, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert labels to one-hot
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.transform(y_test)
        
        # Define network architecture
        layer_sizes = [10, 16, 8, 3]  # input -> hidden -> hidden -> output
        activations = ['relu', 'relu', 'softmax']
        
        # Train network
        trained_model, history = train_neural_network(
            X_train, y_train_onehot, layer_sizes, activations,
            learning_rate=0.01, epochs=50, loss_type='cross_entropy'
        )
        
        # Evaluate
        train_metrics = evaluate_network(X_train, y_train_onehot, trained_model, 'cross_entropy')
        test_metrics = evaluate_network(X_test, y_test_onehot, trained_model, 'cross_entropy')
        
        print(f"Training accuracy: {train_metrics['accuracy']:.3f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
        
        print("✅ Neural network training completed successfully")
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        print("Implement the training functions!")

def comprehensive_neural_network_demo():
    """Demonstrate comprehensive neural network class."""
    
    print("\n🚀 Comprehensive Neural Network Demonstration")
    print("=" * 50)
    
    try:
        # Create regression dataset
        X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape targets for network
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Create neural network
        nn = NeuralNetwork(
            layer_sizes=[5, 10, 5, 1],
            activations=['relu', 'relu', 'linear'],
            loss_type='mse',
            regularization='l2',
            dropout_rate=0.1
        )
        
        # Train network
        nn.train(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01)
        
        # Evaluate
        train_metrics = nn.evaluate(X_train, y_train)
        test_metrics = nn.evaluate(X_test, y_test)
        
        print(f"Training MSE: {train_metrics['mse']:.4f}")
        print(f"Test MSE: {test_metrics['mse']:.4f}")
        
        # Plot training history
        nn.plot_training_history()
        
        print("✅ Comprehensive neural network class working")
        
        print("\n🎉 Congratulations! You've built neural networks from scratch!")
        print("You now understand the mathematical foundations of modern AI.")
        
    except Exception as e:
        print(f"❌ Comprehensive demo failed: {e}")
        print("Implement the NeuralNetwork class!")

if __name__ == "__main__