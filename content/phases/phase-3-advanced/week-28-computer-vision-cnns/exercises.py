#!/usr/bin/env python3
"""
Neural Odyssey - Week 28: Computer Vision with Convolutional Neural Networks
Phase 2: Core ML Algorithms

Hands-on exercises for building CNNs from scratch and understanding computer vision.
This week you'll implement convolution operations, build CNN architectures, and 
apply them to real computer vision problems.

Key Learning Objectives:
1. Understand convolution operation mathematically and implementationally
2. Build CNN layers (convolution, pooling, fully connected) from scratch
3. Implement complete CNN architectures (LeNet, AlexNet-style)
4. Apply CNNs to image classification tasks
5. Understand feature maps and learned representations
6. Explore data augmentation and regularization techniques

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ConvolutionLayer:
    """
    Convolutional layer implementation from scratch
    
    This layer performs the core convolution operation that gives CNNs
    their power to detect spatial patterns in images.
    """
    
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initialize convolutional layer
        
        Args:
            input_channels: Number of input feature maps
            output_channels: Number of output feature maps (filters)
            kernel_size: Size of convolution kernel (assumes square)
            stride: Step size for convolution
            padding: Zero-padding around input
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.normal(
            0, np.sqrt(2.0 / (input_channels * kernel_size * kernel_size)),
            (output_channels, input_channels, kernel_size, kernel_size)
        )
        
        # Initialize biases to zero
        self.biases = np.zeros((output_channels, 1))
        
        # For storing gradients during backpropagation
        self.weight_gradients = None
        self.bias_gradients = None
        self.input_cache = None
        
    def add_padding(self, X: np.ndarray) -> np.ndarray:
        """Add zero padding to input"""
        if self.padding == 0:
            return X
        
        batch_size, channels, height, width = X.shape
        padded = np.zeros((batch_size, channels, 
                          height + 2*self.padding, width + 2*self.padding))
        
        padded[:, :, self.padding:height+self.padding, 
               self.padding:width+self.padding] = X
        
        return padded
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through convolutional layer
        
        Args:
            X: Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        self.input_cache = X  # Store for backpropagation
        
        # Add padding
        X_padded = self.add_padding(X)
        batch_size, input_channels, input_height, input_width = X_padded.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.output_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        # Extract region
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        region = X_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Compute convolution
                        output[b, f, h, w] = np.sum(region * self.weights[f]) + self.biases[f]
        
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through convolutional layer
        
        Args:
            dout: Gradient from layer above
            
        Returns:
            Gradient to pass to layer below
        """
        X = self.input_cache
        X_padded = self.add_padding(X)
        batch_size, input_channels, input_height, input_width = X_padded.shape
        
        # Initialize gradients
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)
        dX_padded = np.zeros_like(X_padded)
        
        output_height, output_width = dout.shape[2], dout.shape[3]
        
        # Compute gradients
        for b in range(batch_size):
            for f in range(self.output_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Weight gradients
                        region = X_padded[b, :, h_start:h_end, w_start:w_end]
                        self.weight_gradients[f] += dout[b, f, h, w] * region
                        
                        # Input gradients
                        dX_padded[b, :, h_start:h_end, w_start:w_end] += \
                            dout[b, f, h, w] * self.weights[f]
                        
                        # Bias gradients
                        self.bias_gradients[f] += dout[b, f, h, w]
        
        # Remove padding from input gradients
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        return dX


class PoolingLayer:
    """
    Max pooling layer implementation
    
    Reduces spatial dimensions while preserving important features
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Initialize pooling layer
        
        Args:
            pool_size: Size of pooling window
            stride: Step size for pooling
        """
        self.pool_size = pool_size
        self.stride = stride
        self.input_cache = None
        self.mask_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through pooling layer"""
        self.input_cache = X
        batch_size, channels, input_height, input_width = X.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        # Initialize output and mask for backpropagation
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.mask_cache = np.zeros_like(X)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Extract region and find max
                        region = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        output[b, c, h, w] = max_val
                        
                        # Create mask for backpropagation
                        mask = (region == max_val)
                        self.mask_cache[b, c, h_start:h_end, w_start:w_end] = mask
        
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through pooling layer"""
        X = self.input_cache
        dX = np.zeros_like(X)
        
        batch_size, channels = X.shape[0], X.shape[1]
        output_height, output_width = dout.shape[2], dout.shape[3]
        
        # Distribute gradients using mask
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Apply gradient only where max was found
                        mask = self.mask_cache[b, c, h_start:h_end, w_start:w_end]
                        dX[b, c, h_start:h_end, w_start:w_end] += dout[b, c, h, w] * mask
        
        return dX


class ActivationLayer:
    """Activation function layer"""
    
    def __init__(self, activation: str = 'relu'):
        """
        Initialize activation layer
        
        Args:
            activation: Type of activation ('relu', 'sigmoid', 'tanh')
        """
        self.activation = activation
        self.input_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through activation"""
        self.input_cache = X
        
        if self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(X)
        else:
            return X  # Linear activation
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through activation"""
        X = self.input_cache
        
        if self.activation == 'relu':
            return dout * (X > 0)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
            return dout * s * (1 - s)
        elif self.activation == 'tanh':
            return dout * (1 - np.tanh(X)**2)
        else:
            return dout


class FlattenLayer:
    """Flatten layer to convert from conv features to dense features"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Flatten spatial dimensions"""
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Reshape back to original dimensions"""
        return dout.reshape(self.input_shape)


class DenseLayer:
    """Fully connected layer"""
    
    def __init__(self, input_size: int, output_size: int):
        """Initialize dense layer"""
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization
        self.weights = np.random.normal(0, np.sqrt(2.0/input_size), 
                                       (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
        self.weight_gradients = None
        self.bias_gradients = None
        self.input_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.input_cache = X
        return np.dot(X, self.weights) + self.biases
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass"""
        X = self.input_cache
        
        self.weight_gradients = np.dot(X.T, dout)
        self.bias_gradients = np.sum(dout, axis=0, keepdims=True)
        
        return np.dot(dout, self.weights.T)


class CNN:
    """
    Complete Convolutional Neural Network implementation
    
    A CNN that can be configured with different architectures
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize CNN
        
        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        self.training_history = {'loss': [], 'accuracy': []}
    
    def add_conv_layer(self, output_channels: int, kernel_size: int, 
                      stride: int = 1, padding: int = 0, activation: str = 'relu'):
        """Add convolutional layer followed by activation"""
        if len(self.layers) == 0:
            input_channels = self.input_shape[0]
        else:
            # Find the last conv layer to get input channels
            input_channels = None
            for layer in reversed(self.layers):
                if isinstance(layer, ConvolutionLayer):
                    input_channels = layer.output_channels
                    break
            if input_channels is None:
                input_channels = self.input_shape[0]
        
        conv_layer = ConvolutionLayer(input_channels, output_channels, 
                                     kernel_size, stride, padding)
        self.layers.append(conv_layer)
        
        if activation:
            self.layers.append(ActivationLayer(activation))
    
    def add_pooling_layer(self, pool_size: int = 2, stride: int = 2):
        """Add pooling layer"""
        self.layers.append(PoolingLayer(pool_size, stride))
    
    def add_flatten_layer(self):
        """Add flatten layer"""
        self.layers.append(FlattenLayer())
    
    def add_dense_layer(self, output_size: int, activation: str = None):
        """Add dense layer with optional activation"""
        # Calculate input size for first dense layer
        if len(self.layers) == 0 or not isinstance(self.layers[-1], (DenseLayer, FlattenLayer)):
            # Need to calculate flattened size
            input_size = np.prod(self.input_shape)
        else:
            # Find the last dense layer
            for layer in reversed(self.layers):
                if isinstance(layer, DenseLayer):
                    input_size = layer.output_size
                    break
            else:
                input_size = np.prod(self.input_shape)
        
        dense_layer = DenseLayer(input_size, output_size)
        self.layers.append(dense_layer)
        
        if activation:
            self.layers.append(ActivationLayer(activation))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through entire network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dout: np.ndarray) -> None:
        """Backward pass through entire network"""
        gradient = dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Softmax
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Cross-entropy loss
        m = y_true.shape[0]
        log_likelihood = -np.log(probs[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        
        # Gradient of loss
        dout = probs.copy()
        dout[range(m), y_true.argmax(axis=1)] -= 1
        dout /= m
        
        return loss, dout
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using computed gradients"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learning_rate * layer.weight_gradients
                layer.biases -= learning_rate * layer.bias_gradients
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10, learning_rate: float = 0.001, 
              batch_size: int = 32, verbose: bool = True) -> Dict[str, List]:
        """
        Train the CNN
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss, dout = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(dout)
                
                # Update weights
                self.update_weights(learning_rate)
                
                # Calculate accuracy
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == true_labels)
            
            # Calculate epoch metrics
            epoch_loss /= n_batches
            epoch_accuracy = epoch_correct / n_samples
            
            # Validation accuracy
            val_pred = self.forward(X_val)
            val_predictions = np.argmax(val_pred, axis=1)
            val_true = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_predictions == val_true)
            
            # Store history
            self.training_history['loss'].append(epoch_loss)
            self.training_history['accuracy'].append(val_accuracy)
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, "
                      f"Train Acc = {epoch_accuracy:.4f}, Val Acc = {val_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


# ==========================================
# DATASET GENERATION AND UTILITIES
# ==========================================

def generate_simple_image_dataset(n_samples: int = 1000, image_size: int = 28, 
                                 n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple synthetic image dataset for CNN testing
    
    Creates images with different geometric patterns:
    - Class 0: Vertical lines
    - Class 1: Horizontal lines
    - Class 2: Diagonal lines
    """
    X = np.zeros((n_samples, 1, image_size, image_size))
    y = np.zeros(n_samples, dtype=int)
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_samples):
        class_idx = i // samples_per_class
        class_idx = min(class_idx, n_classes - 1)  # Handle remainder
        
        img = np.zeros((image_size, image_size))
        
        if class_idx == 0:  # Vertical lines
            for _ in range(3):  # 3 vertical lines
                col = np.random.randint(5, image_size - 5)
                thickness = np.random.randint(1, 3)
                img[:, col:col+thickness] = 1.0
                
        elif class_idx == 1:  # Horizontal lines
            for _ in range(3):  # 3 horizontal lines
                row = np.random.randint(5, image_size - 5)
                thickness = np.random.randint(1, 3)
                img[row:row+thickness, :] = 1.0
                
        elif class_idx == 2:  # Diagonal lines
            for _ in range(2):  # 2 diagonal lines
                if np.random.random() > 0.5:
                    # Main diagonal
                    for j in range(image_size):
                        if 0 <= j < image_size:
                            img[j, j] = 1.0
                            if j+1 < image_size:
                                img[j, j+1] = 1.0
                else:
                    # Anti-diagonal
                    for j in range(image_size):
                        if 0 <= image_size-1-j < image_size:
                            img[j, image_size-1-j] = 1.0
                            if image_size-2-j >= 0:
                                img[j, image_size-2-j] = 1.0
        
        # Add noise
        noise = np.random.normal(0, 0.1, (image_size, image_size))
        img = np.clip(img + noise, 0, 1)
        
        X[i, 0] = img
        y[i] = class_idx
    
    return X, y


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def visualize_conv_features(cnn: CNN, X_sample: np.ndarray, layer_idx: int = 0):
    """
    Visualize feature maps from a convolutional layer
    
    Args:
        cnn: Trained CNN model
        X_sample: Single input sample
        layer_idx: Index of conv layer to visualize
    """
    # Forward pass up to the specified layer
    output = X_sample
    conv_layer_count = 0
    
    for i, layer in enumerate(cnn.layers):
        output = layer.forward(output)
        
        if isinstance(layer, ConvolutionLayer):
            if conv_layer_count == layer_idx:
                feature_maps = output
                break
            conv_layer_count += 1
    
    if 'feature_maps' not in locals():
        print(f"No convolutional layer found at index {layer_idx}")
        return
    
    # Plot feature maps
    n_features = min(16, feature_maps.shape[1])  # Show max 16 features
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Feature Maps from Conv Layer {layer_idx}', fontsize=16)
    
    for i in range(n_features):
        row, col = i // 4, i % 4
        if i < feature_maps.shape[1]:
            axes[row, col].imshow(feature_maps[0, i], cmap='viridis')
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_filters(cnn: CNN, layer_idx: int = 0):
    """
    Visualize learned filters from a convolutional layer
    
    Args:
        cnn: Trained CNN model
        layer_idx: Index of conv layer to visualize
    """
    conv_layer_count = 0
    target_layer = None
    
    for layer in cnn.layers:
        if isinstance(layer, ConvolutionLayer):
            if conv_layer_count == layer_idx:
                target_layer = layer
                break
            conv_layer_count += 1
    
    if target_layer is None:
        print(f"No convolutional layer found at index {layer_idx}")
        return
    
    # Get filters
    filters = target_layer.weights  # Shape: (out_channels, in_channels, h, w)
    
    # Plot filters
    n_filters = min(16, filters.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Learned Filters from Conv Layer {layer_idx}', fontsize=16)
    
    for i in range(n_filters):
        row, col = i // 4, i % 4
        if i < filters.shape[0]:
            # For multi-channel inputs, show first channel or RGB composite
            if filters.shape[1] == 1:
                filter_img = filters[i, 0]
            else:
                # Take first channel or create composite
                filter_img = filters[i, 0]
            
            axes[row, col].imshow(filter_img, cmap='RdBu', vmin=-1, vmax=1)
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# EXERCISE IMPLEMENTATIONS
# ==========================================

def exercise_1_convolution_from_scratch():
    """
    Exercise 1: Implement and understand convolution operation
    
    Build convolution from scratch and visualize how it detects patterns
    """
    print("\n🔍 Exercise 1: Convolution Operation from Scratch")
    print("=" * 60)
    
    # Create a simple test image
    image = np.zeros((1, 1, 8, 8))
    # Add vertical line
    image[0, 0, :, 3:5] = 1.0
    # Add horizontal line
    image[0, 0, 3:5, :] = 1.0
    
    print("Original 8x8 test image with cross pattern:")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image[0, 0], cmap='gray')
    plt.title('Original Image')
    plt.colorbar()
    
    # Test different kernels
    kernels = {
        'Vertical Edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).reshape(1, 1, 3, 3),
        'Horizontal Edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).reshape(1, 1, 3, 3),
        'Blur': np.ones((1, 1, 3, 3)) / 9
    }
    
    conv_layer = ConvolutionLayer(input_channels=1, output_channels=1, 
                                 kernel_size=3, stride=1, padding=1)
    
    for i, (name, kernel) in enumerate(kernels.items()):
        conv_layer.weights = kernel
        conv_layer.biases = np.array([[0]])
        
        result = conv_layer.forward(image)
        
        plt.subplot(1, 4, i + 2)
        plt.imshow(result[0, 0], cmap='RdBu')
        plt.title(f'{name} Detection')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Convolution operation implemented and visualized!")
    print("Notice how different kernels detect different patterns.")


def exercise_2_build_simple_cnn():
    """
    Exercise 2: Build a simple CNN architecture
    
    Create LeNet-style CNN for image classification
    """
    print("\n🏗️ Exercise 2: Build Simple CNN Architecture")
    print("=" * 60)
    
    # Generate synthetic dataset
    print("Generating synthetic image dataset...")
    X, y = generate_simple_image_dataset(n_samples=600, image_size=16, n_classes=3)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to one-hot encoding
    y_train_onehot = one_hot_encode(y_train, 3)
    y_test_onehot = one_hot_encode(y_test, 3)
    
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"Image shape: {X_train.shape[1:]} (channels, height, width)")
    
    # Visualize sample data
    plt.figure(figsize=(12, 4))
    for i in range(3):
        for j in range(3):
            idx = np.where(y_train == i)[0][j]
            plt.subplot(3, 3, i*3 + j + 1)
            plt.imshow(X_train[idx, 0], cmap='gray')
            plt.title(f'Class {i}: {"Vertical" if i==0 else "Horizontal" if i==1 else "Diagonal"}')
            plt.axis('off')
    
    plt.suptitle('Sample Images from Dataset')
    plt.tight_layout()
    plt.show()
    
    # Build simple CNN architecture
    print("\nBuilding CNN architecture:")
    cnn = CNN(input_shape=(1, 16, 16), num_classes=3)
    
    # Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax
    cnn.add_conv_layer(output_channels=8, kernel_size=3, padding=1, activation='relu')
    cnn.add_pooling_layer(pool_size=2, stride=2)
    cnn.add_conv_layer(output_channels=16, kernel_size=3, padding=1, activation='relu')
    cnn.add_pooling_layer(pool_size=2, stride=2)
    cnn.add_flatten_layer()
    cnn.add_dense_layer(output_size=32, activation='relu')
    cnn.add_dense_layer(output_size=3, activation=None)  # Output layer
    
    print("CNN Architecture:")
    print("Input (1, 16, 16) -> Conv(8,3x3) -> ReLU -> MaxPool(2x2)")
    print("-> Conv(16,3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(32) -> ReLU -> Dense(3)")
    
    # Train the CNN
    print("\nTraining CNN...")
    start_time = time.time()
    
    history = cnn.train(
        X_train, y_train_onehot, X_test, y_test_onehot,
        epochs=30, learning_rate=0.01, batch_size=16, verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate model
    predictions = cnn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Vertical', 'Horizontal', 'Diagonal'],
                yticklabels=['Vertical', 'Horizontal', 'Diagonal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("✅ Simple CNN architecture built and trained successfully!")
    return cnn, X_test


def exercise_3_visualize_learned_features():
    """
    Exercise 3: Visualize what CNNs learn
    
    Examine filters and feature maps to understand CNN representations
    """
    print("\n👁️ Exercise 3: Visualize Learned Features")
    print("=" * 60)
    
    # Get trained model from previous exercise
    cnn, X_test = exercise_2_build_simple_cnn()
    
    # Visualize learned filters
    print("Visualizing learned filters...")
    visualize_filters(cnn, layer_idx=0)  # First conv layer
    
    # Visualize feature maps
    print("Visualizing feature maps...")
    sample_image = X_test[0:1]  # First test image
    visualize_conv_features(cnn, sample_image, layer_idx=0)
    
    print("✅ Feature visualization completed!")
    print("Notice how filters learn to detect edges and patterns relevant to the task.")


def exercise_4_advanced_cnn_architecture():
    """
    Exercise 4: Build more advanced CNN architecture
    
    Implement deeper CNN with multiple feature extraction stages
    """
    print("\n🏛️ Exercise 4: Advanced CNN Architecture")
    print("=" * 60)
    
    # Generate larger, more complex dataset
    print("Generating complex dataset...")
    X, y = generate_simple_image_dataset(n_samples=1000, image_size=32, n_classes=3)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to one-hot encoding
    y_train_onehot = one_hot_encode(y_train, 3)
    y_test_onehot = one_hot_encode(y_test, 3)
    
    print(f"Dataset: {X_train.shape[0]} training samples, image size: {X_train.shape[2]}x{X_train.shape[3]}")
    
    # Build advanced CNN architecture (AlexNet-inspired)
    print("\nBuilding advanced CNN architecture...")
    cnn = CNN(input_shape=(1, 32, 32), num_classes=3)
    
    # First conv block
    cnn.add_conv_layer(output_channels=32, kernel_size=5, stride=1, padding=2, activation='relu')
    cnn.add_pooling_layer(pool_size=2, stride=2)
    
    # Second conv block
    cnn.add_conv_layer(output_channels=64, kernel_size=3, stride=1, padding=1, activation='relu')
    cnn.add_pooling_layer(pool_size=2, stride=2)
    
    # Third conv block
    cnn.add_conv_layer(output_channels=128, kernel_size=3, stride=1, padding=1, activation='relu')
    cnn.add_pooling_layer(pool_size=2, stride=2)
    
    # Fully connected layers
    cnn.add_flatten_layer()
    cnn.add_dense_layer(output_size=256, activation='relu')
    cnn.add_dense_layer(output_size=64, activation='relu')
    cnn.add_dense_layer(output_size=3, activation=None)  # Output layer
    
    print("Advanced CNN Architecture:")
    print("Input (1, 32, 32)")
    print("-> Conv(32, 5x5) -> ReLU -> MaxPool(2x2)")
    print("-> Conv(64, 3x3) -> ReLU -> MaxPool(2x2)")
    print("-> Conv(128, 3x3) -> ReLU -> MaxPool(2x2)")
    print("-> Flatten -> Dense(256) -> ReLU -> Dense(64) -> ReLU -> Dense(3)")
    
    # Train the advanced CNN
    print("\nTraining advanced CNN...")
    start_time = time.time()
    
    history = cnn.train(
        X_train, y_train_onehot, X_test, y_test_onehot,
        epochs=40, learning_rate=0.005, batch_size=32, verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate model
    predictions = cnn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    class_names = ['Vertical', 'Horizontal', 'Diagonal']
    print(classification_report(y_test, predictions, target_names=class_names))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Advanced CNN architecture implemented and trained!")
    return cnn


def exercise_5_cnn_analysis_and_interpretation():
    """
    Exercise 5: Analyze and interpret CNN behavior
    
    Deep dive into understanding what makes CNNs effective
    """
    print("\n🔬 Exercise 5: CNN Analysis and Interpretation")
    print("=" * 60)
    
    # Create analysis CNN
    cnn = exercise_4_advanced_cnn_architecture()
    
    print("\n📊 Analyzing CNN Properties:")
    
    # 1. Parameter count analysis
    total_params = 0
    trainable_params = 0
    
    print("\nLayer-wise Parameter Analysis:")
    for i, layer in enumerate(cnn.layers):
        if hasattr(layer, 'weights'):
            layer_params = np.prod(layer.weights.shape) + np.prod(layer.biases.shape)
            total_params += layer_params
            trainable_params += layer_params
            
            if isinstance(layer, ConvolutionLayer):
                print(f"Conv Layer {i}: {layer_params:,} parameters "
                      f"({layer.output_channels} filters of {layer.kernel_size}x{layer.kernel_size})")
            elif isinstance(layer, DenseLayer):
                print(f"Dense Layer {i}: {layer_params:,} parameters "
                      f"({layer.input_size} -> {layer.output_size})")
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 2. Feature map size analysis
    print("\nFeature Map Size Analysis:")
    test_input = np.random.randn(1, 1, 32, 32)
    current_shape = test_input.shape
    
    print(f"Input: {current_shape}")
    
    for i, layer in enumerate(cnn.layers):
        if isinstance(layer, (ConvolutionLayer, PoolingLayer, FlattenLayer)):
            test_input = layer.forward(test_input)
            if isinstance(layer, ConvolutionLayer):
                print(f"After Conv {i}: {test_input.shape}")
            elif isinstance(layer, PoolingLayer):
                print(f"After Pool {i}: {test_input.shape}")
            elif isinstance(layer, FlattenLayer):
                print(f"After Flatten {i}: {test_input.shape}")
    
    # 3. Computational complexity analysis
    print("\nComputational Complexity Analysis:")
    
    def compute_conv_ops(input_h, input_w, input_c, kernel_h, kernel_w, output_c, output_h, output_w):
        """Compute number of operations for convolution"""
        return output_h * output_w * output_c * (kernel_h * kernel_w * input_c + 1)  # +1 for bias
    
    # Reset for analysis
    current_h, current_w, current_c = 32, 32, 1
    total_ops = 0
    
    for i, layer in enumerate(cnn.layers):
        if isinstance(layer, ConvolutionLayer):
            output_h = (current_h + 2*layer.padding - layer.kernel_size) // layer.stride + 1
            output_w = (current_w + 2*layer.padding - layer.kernel_size) // layer.stride + 1
            
            ops = compute_conv_ops(current_h, current_w, current_c,
                                 layer.kernel_size, layer.kernel_size,
                                 layer.output_channels, output_h, output_w)
            
            total_ops += ops
            print(f"Conv Layer {i}: {ops:,} operations")
            
            current_h, current_w, current_c = output_h, output_w, layer.output_channels
            
        elif isinstance(layer, PoolingLayer):
            current_h = (current_h - layer.pool_size) // layer.stride + 1
            current_w = (current_w - layer.pool_size) // layer.stride + 1
            
        elif isinstance(layer, DenseLayer):
            ops = layer.input_size * layer.output_size + layer.output_size  # weights + bias
            total_ops += ops
            print(f"Dense Layer {i}: {ops:,} operations")
    
    print(f"\nTotal Operations per Forward Pass: {total_ops:,}")
    
    # 4. Receptive field analysis
    print("\nReceptive Field Analysis:")
    
    def calculate_receptive_field():
        """Calculate receptive field size at each layer"""
        rf = 1  # Start with 1x1 receptive field
        stride_product = 1
        
        for layer in cnn.layers:
            if isinstance(layer, ConvolutionLayer):
                rf = rf + (layer.kernel_size - 1) * stride_product
                stride_product *= layer.stride
                print(f"After conv layer: RF = {rf}x{rf}, stride = {stride_product}")
                
            elif isinstance(layer, PoolingLayer):
                rf = rf + (layer.pool_size - 1) * stride_product
                stride_product *= layer.stride
                print(f"After pool layer: RF = {rf}x{rf}, stride = {stride_product}")
        
        return rf
    
    final_rf = calculate_receptive_field()
    print(f"\nFinal Receptive Field: {final_rf}x{final_rf} pixels")
    print(f"This means the output depends on a {final_rf}x{final_rf} region of the input image.")
    
    print("✅ CNN analysis completed!")
    print("\n🎓 Key Insights:")
    print("1. Early layers detect simple features (edges, corners)")
    print("2. Deeper layers combine features to detect complex patterns")
    print("3. Pooling reduces spatial size while increasing semantic content")
    print("4. Receptive field grows with depth, allowing global pattern recognition")


def exercise_6_cnn_optimization_techniques():
    """
    Exercise 6: CNN optimization and regularization techniques
    
    Explore data augmentation, dropout, and other optimization methods
    """
    print("\n⚡ Exercise 6: CNN Optimization Techniques")
    print("=" * 60)
    
    def augment_data(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple data augmentation: rotation, flipping, noise
        """
        original_size = X.shape[0]
        augmented_size = original_size * augmentation_factor
        
        X_aug = np.zeros((augmented_size, *X.shape[1:]))
        y_aug = np.zeros(augmented_size, dtype=y.dtype)
        
        # Original data
        X_aug[:original_size] = X
        y_aug[:original_size] = y
        
        # Augmented data
        for i in range(original_size, augmented_size):
            original_idx = i % original_size
            img = X[original_idx, 0].copy()
            
            # Random transformations
            if np.random.random() > 0.5:
                # Horizontal flip
                img = np.fliplr(img)
            
            if np.random.random() > 0.5:
                # Add noise
                noise = np.random.normal(0, 0.05, img.shape)
                img = np.clip(img + noise, 0, 1)
            
            if np.random.random() > 0.5:
                # Slight rotation (90 degrees for simplicity)
                img = np.rot90(img, k=np.random.randint(1, 4))
            
            X_aug[i, 0] = img
            y_aug[i] = y[original_idx]
        
        return X_aug, y_aug
    
    # Generate dataset
    print("Generating dataset...")
    X, y = generate_simple_image_dataset(n_samples=400, image_size=24, n_classes=3)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compare with and without augmentation
    print("\n🔄 Testing Data Augmentation:")
    
    # Without augmentation
    print("Training without data augmentation...")
    cnn_no_aug = CNN(input_shape=(1, 24, 24), num_classes=3)
    cnn_no_aug.add_conv_layer(16, 3, padding=1, activation='relu')
    cnn_no_aug.add_pooling_layer(2, 2)
    cnn_no_aug.add_conv_layer(32, 3, padding=1, activation='relu')
    cnn_no_aug.add_pooling_layer(2, 2)
    cnn_no_aug.add_flatten_layer()
    cnn_no_aug.add_dense_layer(64, activation='relu')
    cnn_no_aug.add_dense_layer(3, activation=None)
    
    y_train_onehot = one_hot_encode(y_train, 3)
    y_test_onehot = one_hot_encode(y_test, 3)
    
    history_no_aug = cnn_no_aug.train(
        X_train, y_train_onehot, X_test, y_test_onehot,
        epochs=25, learning_rate=0.01, batch_size=16, verbose=False
    )
    
    # With augmentation
    print("Training with data augmentation...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor=3)
    y_train_aug_onehot = one_hot_encode(y_train_aug, 3)
    
    cnn_with_aug = CNN(input_shape=(1, 24, 24), num_classes=3)
    cnn_with_aug.add_conv_layer(16, 3, padding=1, activation='relu')
    cnn_with_aug.add_pooling_layer(2, 2)
    cnn_with_aug.add_conv_layer(32, 3, padding=1, activation='relu')
    cnn_with_aug.add_pooling_layer(2, 2)
    cnn_with_aug.add_flatten_layer()
    cnn_with_aug.add_dense_layer(64, activation='relu')
    cnn_with_aug.add_dense_layer(3, activation=None)
    
    history_with_aug = cnn_with_aug.train(
        X_train_aug, y_train_aug_onehot, X_test, y_test_onehot,
        epochs=25, learning_rate=0.01, batch_size=16, verbose=False
    )
    
    # Compare results
    acc_no_aug = accuracy_score(y_test, cnn_no_aug.predict(X_test))
    acc_with_aug = accuracy_score(y_test, cnn_with_aug.predict(X_test))
    
    print(f"Accuracy without augmentation: {acc_no_aug:.4f}")
    print(f"Accuracy with augmentation: {acc_with_aug:.4f}")
    print(f"Improvement: {(acc_with_aug - acc_no_aug)*100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_no_aug['accuracy'], label='No Augmentation', linewidth=2)
    plt.plot(history_with_aug['accuracy'], label='With Augmentation', linewidth=2)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_no_aug['loss'], label='No Augmentation', linewidth=2)
    plt.plot(history_with_aug['loss'], label='With Augmentation', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Show sample augmented images
    plt.subplot(1, 3, 3)
    sample_idx = 0
    original = X_train[sample_idx, 0]
    
    # Create augmented versions
    augmented_samples = []
    for _ in range(4):
        img = original.copy()
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        if np.random.random() > 0.5:
            img = np.rot90(img, k=np.random.randint(1, 4))
        augmented_samples.append(img)
    
    # Display original and augmented
    combined = np.hstack([original] + augmented_samples)
    plt.imshow(combined, cmap='gray')
    plt.title('Original + Augmented Samples')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ CNN optimization techniques demonstrated!")
    print("\n📈 Optimization Benefits:")
    print("1. Data augmentation increases dataset size and diversity")
    print("2. Helps model generalize better to unseen data")
    print("3. Reduces overfitting by providing more training examples")
    print("4. Particularly effective for small datasets")


def comprehensive_cnn_demonstration():
    """
    Comprehensive demonstration of CNN concepts and implementations
    """
    print("\n" + "="*80)
    print("🧠 NEURAL ODYSSEY - WEEK 28: COMPUTER VISION WITH CNNs")
    print("="*80)
    print("\nWelcome to the fascinating world of Convolutional Neural Networks!")
    print("Today you'll build CNNs from scratch and understand computer vision.")
    print("\n🎯 Learning Objectives:")
    print("• Understand convolution operation mathematically")
    print("• Implement CNN layers from scratch")
    print("• Build and train complete CNN architectures")
    print("• Visualize learned features and representations")
    print("• Apply optimization techniques for better performance")
    
    # Run all exercises
    try:
        exercise_1_convolution_from_scratch()
        exercise_2_build_simple_cnn()
        exercise_3_visualize_learned_features()
        exercise_4_advanced_cnn_architecture()
        exercise_5_cnn_analysis_and_interpretation()
        exercise_6_cnn_optimization_techniques()
        
        print("\n" + "="*80)
        print("🎉 CONGRATULATIONS! CNN MASTERY ACHIEVED!")
        print("="*80)
        print("\n✨ You have successfully:")
        print("• ✅ Implemented convolution operation from scratch")
        print("• ✅ Built complete CNN architectures")
        print("• ✅ Trained CNNs on image classification tasks")
        print("• ✅ Visualized learned filters and feature maps")
        print("• ✅ Analyzed CNN properties and complexity")
        print("• ✅ Applied optimization and regularization techniques")
        
        print("\n🚀 Next Steps in Your Computer Vision Journey:")
        print("• Explore object detection (YOLO, R-CNN)")
        print("• Study image segmentation techniques")
        print("• Learn about modern architectures (ResNet, Vision Transformers)")
        print("• Apply CNNs to real-world problems")
        print("• Understand transfer learning and pre-trained models")
        
        print("\n💡 Key Insights Gained:")
        print("• CNNs excel at spatial pattern recognition")
        print("• Convolution preserves spatial relationships")
        print("• Hierarchical features: edges → shapes → objects")
        print("• Pooling provides translation invariance")
        print("• Data augmentation improves generalization")
        
        print("\n🎓 You're now equipped with deep understanding of computer vision!")
        print("The foundation you've built will serve you throughout your AI journey.")
        
    except Exception as e:
        print(f"\n❌ Error in CNN demonstration: {e}")
        print("Review the implementation and try again!")


# ==========================================
# BONUS: REAL-WORLD APPLICATION EXAMPLE
# ==========================================

def bonus_mnist_style_example():
    """
    Bonus: Apply CNN to MNIST-style digit recognition
    
    Demonstrate CNN on a problem similar to the famous MNIST dataset
    """
    print("\n🌟 BONUS: MNIST-Style Digit Recognition")
    print("=" * 60)
    
    def create_digit_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic digit-like patterns"""
        X = np.zeros((n_samples, 1, 20, 20))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            digit = i % 3  # 3 different "digits"
            img = np.zeros((20, 20))
            
            if digit == 0:  # Circle-like
                center = (10, 10)
                radius = 6
                for r in range(20):
                    for c in range(20):
                        dist = np.sqrt((r - center[0])**2 + (c - center[1])**2)
                        if 4 < dist < radius:
                            img[r, c] = 1.0
                            
            elif digit == 1:  # Line-like
                img[5:15, 9:11] = 1.0  # Vertical line
                img[4:6, 8:12] = 1.0   # Top
                
            elif digit == 2:  # Square-like
                img[5:15, 5:7] = 1.0   # Left
                img[5:15, 13:15] = 1.0 # Right
                img[5:7, 5:15] = 1.0   # Top
                img[13:15, 5:15] = 1.0 # Bottom
            
            # Add noise
            noise = np.random.normal(0, 0.1, (20, 20))
            img = np.clip(img + noise, 0, 1)
            
            X[i, 0] = img
            y[i] = digit
        
        return X, y
    
    # Generate digit dataset
    print("Creating synthetic digit dataset...")
    X, y = create_digit_dataset(n_samples=900)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_train_onehot = one_hot_encode(y_train, 3)
    y_test_onehot = one_hot_encode(y_test, 3)
    
    # Visualize samples
    plt.figure(figsize=(12, 4))
    digit_names = ['Circle', 'Line', 'Square']
    for i in range(3):
        for j in range(4):
            idx = np.where(y_train == i)[0][j]
            plt.subplot(3, 4, i*4 + j + 1)
            plt.imshow(X_train[idx, 0], cmap='gray')
            plt.title(f'{digit_names[i]}')
            plt.axis('off')
    
    plt.suptitle('Synthetic Digit Dataset')
    plt.tight_layout()
    plt.show()
    
    # Build specialized CNN for digits
    print("Building digit recognition CNN...")
    cnn = CNN(input_shape=(1, 20, 20), num_classes=3)
    
    # LeNet-inspired architecture
    cnn.add_conv_layer(6, 5, stride=1, activation='relu')
    cnn.add_pooling_layer(2, 2)
    cnn.add_conv_layer(16, 5, stride=1, activation='relu')
    cnn.add_pooling_layer(2, 2)
    cnn.add_flatten_layer()
    cnn.add_dense_layer(120, activation='relu')
    cnn.add_dense_layer(84, activation='relu')
    cnn.add_dense_layer(3, activation=None)
    
    # Train model
    print("Training digit recognition model...")
    history = cnn.train(
        X_train, y_train_onehot, X_test, y_test_onehot,
        epochs=50, learning_rate=0.01, batch_size=32, verbose=True
    )
    
    # Evaluate final performance
    predictions = cnn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    # Detailed analysis
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=digit_names))
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Confusion matrix
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=digit_names, yticklabels=digit_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Sample predictions
    plt.subplot(2, 3, 4)
    sample_indices = np.random.choice(len(X_test), 6, replace=False)
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 6, 7 + i)
        plt.imshow(X_test[idx, 0], cmap='gray')
        true_label = digit_names[y_test[idx]]
        pred_label = digit_names[predictions[idx]]
        color = 'green' if y_test[idx] == predictions[idx] else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize learned filters
    print("Visualizing learned filters...")
    visualize_filters(cnn, layer_idx=0)
    
    print("✅ MNIST-style digit recognition completed!")
    print(f"Achieved {accuracy*100:.1f}% accuracy on synthetic digit classification")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🧠 Neural Odyssey - Week 28: Computer Vision with CNNs")
    print("=" * 60)
    print("Choose an exercise to run:")
    print("1. Convolution from Scratch")
    print("2. Build Simple CNN")
    print("3. Visualize Features") 
    print("4. Advanced CNN Architecture")
    print("5. CNN Analysis")
    print("6. Optimization Techniques")
    print("7. Comprehensive Demo (All Exercises)")
    print("8. Bonus: MNIST-style Recognition")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("👋 Happy learning! Keep exploring computer vision!")
                break
            elif choice == '1':
                exercise_1_convolution_from_scratch()
            elif choice == '2':
                exercise_2_build_simple_cnn()
            elif choice == '3':
                exercise_3_visualize_learned_features()
            elif choice == '4':
                exercise_4_advanced_cnn_architecture()
            elif choice == '5':
                exercise_5_cnn_analysis_and_interpretation()
            elif choice == '6':
                exercise_6_cnn_optimization_techniques()
            elif choice == '7':
                comprehensive_cnn_demonstration()
            elif choice == '8':
                bonus_mnist_style_example()
            else:
                print("❌ Invalid choice. Please enter 0-8.")
                continue
                
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for learning about CNNs! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or choose a different exercise.")


# ==========================================
# UTILITY FUNCTIONS FOR ADVANCED USERS
# ==========================================

class CNNMetrics:
    """
    Advanced metrics and analysis tools for CNNs
    """
    
    @staticmethod
    def calculate_flops(cnn: CNN, input_shape: Tuple[int, int, int]) -> int:
        """
        Calculate FLOPs (Floating Point Operations) for CNN
        
        Args:
            cnn: CNN model
            input_shape: Input tensor shape (C, H, W)
            
        Returns:
            Total FLOPs count
        """
        total_flops = 0
        current_shape = input_shape
        
        for layer in cnn.layers:
            if isinstance(layer, ConvolutionLayer):
                c_in, h_in, w_in = current_shape
                h_out = (h_in + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                w_out = (w_in + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                
                # Convolution FLOPs: output_size * (kernel_ops + bias)
                kernel_ops = layer.kernel_size * layer.kernel_size * c_in
                conv_flops = h_out * w_out * layer.output_channels * (kernel_ops + 1)
                total_flops += conv_flops
                
                current_shape = (layer.output_channels, h_out, w_out)
                
            elif isinstance(layer, PoolingLayer):
                c, h, w = current_shape
                h_out = (h - layer.pool_size) // layer.stride + 1
                w_out = (w - layer.pool_size) // layer.stride + 1
                
                # Pooling FLOPs: comparisons for max operation
                pool_flops = h_out * w_out * c * (layer.pool_size * layer.pool_size - 1)
                total_flops += pool_flops
                
                current_shape = (c, h_out, w_out)
                
            elif isinstance(layer, FlattenLayer):
                current_shape = (np.prod(current_shape),)
                
            elif isinstance(layer, DenseLayer):
                # Dense layer FLOPs: matrix multiplication + bias
                dense_flops = layer.input_size * layer.output_size + layer.output_size
                total_flops += dense_flops
                current_shape = (layer.output_size,)
        
        return total_flops
    
    @staticmethod
    def memory_usage(cnn: CNN, input_shape: Tuple[int, int, int], batch_size: int = 1) -> Dict[str, int]:
        """
        Calculate memory usage for CNN
        
        Args:
            cnn: CNN model
            input_shape: Input tensor shape (C, H, W)
            batch_size: Batch size for memory calculation
            
        Returns:
            Dictionary with memory usage statistics
        """
        # Parameter memory
        param_memory = 0
        for layer in cnn.layers:
            if hasattr(layer, 'weights'):
                param_memory += layer.weights.nbytes + layer.biases.nbytes
        
        # Activation memory (forward pass)
        activation_memory = 0
        current_shape = (batch_size,) + input_shape
        
        for layer in cnn.layers:
            if isinstance(layer, ConvolutionLayer):
                h_out = (current_shape[2] + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                w_out = (current_shape[3] + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                current_shape = (batch_size, layer.output_channels, h_out, w_out)
                
            elif isinstance(layer, PoolingLayer):
                h_out = (current_shape[2] - layer.pool_size) // layer.stride + 1
                w_out = (current_shape[3] - layer.pool_size) // layer.stride + 1
                current_shape = (batch_size, current_shape[1], h_out, w_out)
                
            elif isinstance(layer, FlattenLayer):
                current_shape = (batch_size, np.prod(current_shape[1:]))
                
            elif isinstance(layer, DenseLayer):
                current_shape = (batch_size, layer.output_size)
            
            # 4 bytes per float32
            layer_memory = np.prod(current_shape) * 4
            activation_memory += layer_memory
        
        # Gradient memory (approximately same as parameters)
        gradient_memory = param_memory
        
        return {
            'parameters': param_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'total': param_memory + activation_memory + gradient_memory
        }
    
    @staticmethod
    def analyze_architecture(cnn: CNN, input_shape: Tuple[int, int, int]) -> None:
        """
        Comprehensive architecture analysis
        
        Args:
            cnn: CNN model to analyze
            input_shape: Input tensor shape (C, H, W)
        """
        print("\n🔍 Comprehensive CNN Architecture Analysis")
        print("=" * 60)
        
        # Basic info
        total_layers = len(cnn.layers)
        conv_layers = sum(1 for layer in cnn.layers if isinstance(layer, ConvolutionLayer))
        dense_layers = sum(1 for layer in cnn.layers if isinstance(layer, DenseLayer))
        
        print(f"Total Layers: {total_layers}")
        print(f"Convolutional Layers: {conv_layers}")
        print(f"Dense Layers: {dense_layers}")
        
        # Parameter analysis
        total_params = 0
        for layer in cnn.layers:
            if hasattr(layer, 'weights'):
                layer_params = np.prod(layer.weights.shape) + np.prod(layer.biases.shape)
                total_params += layer_params
        
        print(f"Total Parameters: {total_params:,}")
        
        # Computational complexity
        flops = CNNMetrics.calculate_flops(cnn, input_shape)
        print(f"FLOPs per inference: {flops:,}")
        
        # Memory usage
        memory = CNNMetrics.memory_usage(cnn, input_shape)
        print(f"Memory Usage:")
        print(f"  Parameters: {memory['parameters']/1024/1024:.2f} MB")
        print(f"  Activations: {memory['activations']/1024/1024:.2f} MB")
        print(f"  Total: {memory['total']/1024/1024:.2f} MB")
        
        # Efficiency metrics
        params_per_flop = total_params / flops if flops > 0 else 0
        print(f"Parameter Efficiency: {params_per_flop:.2e} params/FLOP")


def advanced_cnn_tutorial():
    """
    Advanced tutorial covering modern CNN concepts
    """
    print("\n🎓 Advanced CNN Concepts Tutorial")
    print("=" * 60)
    
    print("📚 Modern CNN Developments:")
    print("\n1. **Residual Networks (ResNet)**")
    print("   - Skip connections solve vanishing gradient problem")
    print("   - Enable training of very deep networks (100+ layers)")
    print("   - Identity shortcuts: output = F(x) + x")
    
    print("\n2. **Batch Normalization**")
    print("   - Normalizes layer inputs to reduce internal covariate shift")
    print("   - Accelerates training and improves convergence")
    print("   - Acts as regularization technique")
    
    print("\n3. **Depthwise Separable Convolutions**")
    print("   - Reduces parameters and computation")
    print("   - Used in MobileNets for efficient mobile deployment")
    print("   - Separates spatial and channel-wise convolutions")
    
    print("\n4. **Attention Mechanisms**")
    print("   - Focus on important regions of the input")
    print("   - Bridge between CNNs and Transformer architectures")
    print("   - Enable better feature selection")
    
    print("\n5. **Vision Transformers (ViTs)**")
    print("   - Apply Transformer architecture to computer vision")
    print("   - Treat image patches as tokens")
    print("   - Competitive with CNNs on large datasets")
    
    print("\n🔬 Research Directions:")
    print("• Neural Architecture Search (NAS)")
    print("• Efficient model compression and pruning")
    print("• Self-supervised learning for vision")
    print("• Multi-modal learning (vision + language)")
    print("• 3D vision and video understanding")
    
    print("\n💼 Industry Applications:")
    print("• Autonomous vehicles (object detection, segmentation)")
    print("• Medical imaging (diagnosis, analysis)")
    print("• Manufacturing (quality control, defect detection)")
    print("• Agriculture (crop monitoring, disease detection)")
    print("• Retail (visual search, inventory management)")
    
    print("\n✅ Congratulations on completing the advanced CNN tutorial!")


# Final summary message
def print_week_summary():
    """Print comprehensive summary of Week 28 learning"""
    print("\n" + "🎯"*20)
    print("WEEK 28 COMPLETE: COMPUTER VISION WITH CNNs")
    print("🎯"*20)
    
    print("\n📈 Skills Mastered:")
    print("✅ Mathematical understanding of convolution")
    print("✅ CNN architecture design and implementation")
    print("✅ Training and optimization techniques")
    print("✅ Feature visualization and interpretation")
    print("✅ Performance analysis and debugging")
    print("✅ Data augmentation and regularization")
    
    print("\n🏆 Achievements Unlocked:")
    print("🥇 Built CNN from scratch in pure NumPy")
    print("🥈 Implemented multiple CNN architectures")
    print("🥉 Applied CNNs to image classification")
    print("🏅 Visualized learned representations")
    print("🎖️  Analyzed computational complexity")
    
    print("\n🚀 Ready for Next Challenges:")
    print("📍 Object Detection (YOLO, R-CNN)")
    print("📍 Image Segmentation (U-Net, Mask R-CNN)")
    print("📍 Generative Models (GANs, VAEs)")
    print("📍 Transfer Learning and Fine-tuning")
    print("📍 Modern Architectures (ResNet, Vision Transformers)")
    
    print(f"\n💡 Total Lines of Code: ~{2000}+")
    print("💡 Mathematical Concepts: Convolution, Backpropagation, Optimization")
    print("💡 Deep Learning: CNNs, Feature Learning, Computer Vision")
    
    print("\n🌟 You've gained deep understanding of computer vision!")
    print("🌟 Your CNN knowledge forms the foundation for advanced CV work!")


# Execute summary at module level
if __name__ == "__main__":
    # Uncomment to see week summary
    # print_week_summary()