#!/usr/bin/env python3
"""
Neural Odyssey - Week 29: Deep Learning Architectures
Phase 2: Core ML Algorithms

Advanced exercises for building modern deep learning architectures from scratch.
This week you'll implement ResNet, DenseNet, attention mechanisms, and explore
the evolution of neural network architectures.

Key Learning Objectives:
1. Understand and implement residual connections (ResNet)
2. Build dense connectivity patterns (DenseNet)
3. Implement attention mechanisms from scratch
4. Explore efficient architectures (MobileNet concepts)
5. Compare architectural design choices and trade-offs
6. Analyze computational complexity and efficiency
7. Build Vision Transformer components

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
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResidualBlock:
    """
    Residual Block implementation from ResNet paper
    
    Implements the core innovation of ResNet: skip connections that enable
    training of very deep networks by allowing gradients to flow directly
    through identity mappings.
    """
    
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        """
        Initialize residual block
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            stride: Stride for the first convolution
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        
        # First convolution
        self.conv1_weights = np.random.normal(
            0, np.sqrt(2.0 / (input_channels * 3 * 3)),
            (output_channels, input_channels, 3, 3)
        )
        self.conv1_bias = np.zeros((output_channels, 1))
        
        # Second convolution
        self.conv2_weights = np.random.normal(
            0, np.sqrt(2.0 / (output_channels * 3 * 3)),
            (output_channels, output_channels, 3, 3)
        )
        self.conv2_bias = np.zeros((output_channels, 1))
        
        # Shortcut connection (if dimensions don't match)
        self.use_shortcut_conv = (input_channels != output_channels) or (stride != 1)
        if self.use_shortcut_conv:
            self.shortcut_weights = np.random.normal(
                0, np.sqrt(2.0 / (input_channels * 1 * 1)),
                (output_channels, input_channels, 1, 1)
            )
            self.shortcut_bias = np.zeros((output_channels, 1))
        
        # Batch normalization parameters (simplified)
        self.bn1_gamma = np.ones((output_channels, 1))
        self.bn1_beta = np.zeros((output_channels, 1))
        self.bn2_gamma = np.ones((output_channels, 1))
        self.bn2_beta = np.zeros((output_channels, 1))
        
        # Cache for backpropagation
        self.cache = {}
    
    def batch_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-5) -> np.ndarray:
        """Simplified batch normalization"""
        # Compute statistics along batch and spatial dimensions
        batch_size, channels, height, width = x.shape
        
        # Reshape for computation
        x_reshaped = x.reshape(batch_size, channels, -1)
        
        # Compute mean and variance
        mean = np.mean(x_reshaped, axis=(0, 2), keepdims=True)
        var = np.var(x_reshaped, axis=(0, 2), keepdims=True)
        
        # Normalize
        x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
        
        # Scale and shift
        x_norm = gamma.reshape(1, -1, 1) * x_norm + beta.reshape(1, -1, 1)
        
        # Reshape back
        return x_norm.reshape(batch_size, channels, height, width)
    
    def conv2d(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray,
               stride: int = 1, padding: int = 1) -> np.ndarray:
        """Simplified 2D convolution"""
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = weights.shape
        
        # Add padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                             mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_height = (x_padded.shape[2] - kernel_size) // stride + 1
        out_width = (x_padded.shape[3] - kernel_size) // stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        
                        # Extract region and compute convolution
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, h, w] = np.sum(region * weights[oc]) + bias[oc]
        
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through residual block
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor after residual connection
        """
        # Store input for skip connection
        identity = x
        
        # First convolution + batch norm + ReLU
        out = self.conv2d(x, self.conv1_weights, self.conv1_bias, 
                         stride=self.stride, padding=1)
        out = self.batch_norm(out, self.bn1_gamma, self.bn1_beta)
        out = self.relu(out)
        
        # Second convolution + batch norm
        out = self.conv2d(out, self.conv2_weights, self.conv2_bias, 
                         stride=1, padding=1)
        out = self.batch_norm(out, self.bn2_gamma, self.bn2_beta)
        
        # Shortcut connection
        if self.use_shortcut_conv:
            identity = self.conv2d(identity, self.shortcut_weights, self.shortcut_bias,
                                 stride=self.stride, padding=0)
        
        # Residual connection: F(x) + x
        out = out + identity
        
        # Final ReLU
        out = self.relu(out)
        
        return out


class DenseBlock:
    """
    Dense Block implementation from DenseNet paper
    
    Implements dense connectivity where each layer receives feature maps
    from all preceding layers, promoting feature reuse and gradient flow.
    """
    
    def __init__(self, input_channels: int, growth_rate: int, num_layers: int):
        """
        Initialize dense block
        
        Args:
            input_channels: Number of input channels
            growth_rate: Number of feature maps each layer adds
            num_layers: Number of dense layers in the block
        """
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        
        # Initialize weights for each layer
        self.layers = []
        current_channels = input_channels
        
        for i in range(num_layers):
            # Bottleneck layer (1x1 conv to reduce dimensions)
            bottleneck_channels = 4 * growth_rate
            
            # 1x1 convolution weights
            conv1x1_weights = np.random.normal(
                0, np.sqrt(2.0 / current_channels),
                (bottleneck_channels, current_channels, 1, 1)
            )
            
            # 3x3 convolution weights
            conv3x3_weights = np.random.normal(
                0, np.sqrt(2.0 / (bottleneck_channels * 3 * 3)),
                (growth_rate, bottleneck_channels, 3, 3)
            )
            
            layer = {
                'conv1x1_weights': conv1x1_weights,
                'conv1x1_bias': np.zeros((bottleneck_channels, 1)),
                'conv3x3_weights': conv3x3_weights,
                'conv3x3_bias': np.zeros((growth_rate, 1)),
                'bn1_gamma': np.ones((bottleneck_channels, 1)),
                'bn1_beta': np.zeros((bottleneck_channels, 1)),
                'bn2_gamma': np.ones((growth_rate, 1)),
                'bn2_beta': np.zeros((growth_rate, 1))
            }
            
            self.layers.append(layer)
            current_channels += growth_rate  # Concatenation increases channels
    
    def conv2d_simple(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Simplified convolution for demonstration"""
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = weights.shape
        
        if kernel_size == 1:
            # 1x1 convolution
            output = np.zeros((batch_size, out_channels, in_height, in_width))
            for b in range(batch_size):
                for oc in range(out_channels):
                    for h in range(in_height):
                        for w in range(in_width):
                            output[b, oc, h, w] = np.sum(x[b, :, h, w] * weights[oc, :, 0, 0]) + bias[oc]
        else:
            # 3x3 convolution with padding
            padding = kernel_size // 2
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            
            out_height, out_width = in_height, in_width
            output = np.zeros((batch_size, out_channels, out_height, out_width))
            
            for b in range(batch_size):
                for oc in range(out_channels):
                    for h in range(out_height):
                        for w in range(out_width):
                            region = x_padded[b, :, h:h+kernel_size, w:w+kernel_size]
                            output[b, oc, h, w] = np.sum(region * weights[oc]) + bias[oc]
        
        return output
    
    def batch_norm_simple(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Simplified batch normalization"""
        batch_size, channels, height, width = x.shape
        
        # Compute statistics
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        
        # Scale and shift
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through dense block
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with concatenated features
        """
        features = [x]  # List to store all feature maps
        
        for layer in self.layers:
            # Get input as concatenation of all previous features
            layer_input = np.concatenate(features, axis=1)
            
            # Bottleneck layer: BN + ReLU + 1x1 Conv
            out = self.batch_norm_simple(layer_input, layer['bn1_gamma'], layer['bn1_beta'])
            out = self.relu(out)
            out = self.conv2d_simple(out, layer['conv1x1_weights'], layer['conv1x1_bias'])
            
            # Composite function: BN + ReLU + 3x3 Conv
            out = self.batch_norm_simple(out, layer['bn2_gamma'], layer['bn2_beta'])
            out = self.relu(out)
            out = self.conv2d_simple(out, layer['conv3x3_weights'], layer['conv3x3_bias'])
            
            # Add new features to the list
            features.append(out)
        
        # Return concatenation of all features
        return np.concatenate(features, axis=1)


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism from "Attention Is All You Need"
    
    Implements the core attention mechanism that revolutionized deep learning
    by allowing models to attend to different parts of the input simultaneously.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize projection matrices
        self.W_q = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_k = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_v = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_o = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        
        # Bias terms
        self.b_q = np.zeros((d_model,))
        self.b_k = np.zeros((d_model,))
        self.b_v = np.zeros((d_model,))
        self.b_o = np.zeros((d_model,))
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention
        
        Args:
            Q: Query matrix (batch_size, seq_len, d_k)
            K: Key matrix (batch_size, seq_len, d_k)
            V: Value matrix (batch_size, seq_len, d_k)
            mask: Optional attention mask
            
        Returns:
            Attention output and attention weights
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through multi-head attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.dot(x, self.W_q) + self.b_q
        K = np.dot(x, self.W_k) + self.b_k
        V = np.dot(x, self.W_v) + self.b_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply attention to each head
        attention_outputs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            head_output, head_weights = self.scaled_dot_product_attention(
                Q[:, head], K[:, head], V[:, head], mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)
        
        # Concatenate heads
        concat_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear projection
        output = np.dot(concat_output, self.W_o) + self.b_o
        
        return output


class DepthwiseSeparableConv:
    """
    Depthwise Separable Convolution from MobileNet
    
    Factorizes standard convolution into depthwise and pointwise operations
    to significantly reduce computational cost while maintaining expressiveness.
    """
    
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3):
        """
        Initialize depthwise separable convolution
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Size of depthwise kernels
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution weights (one kernel per input channel)
        self.depthwise_weights = np.random.normal(
            0, np.sqrt(2.0 / (kernel_size * kernel_size)),
            (input_channels, 1, kernel_size, kernel_size)
        )
        
        # Pointwise convolution weights (1x1 convolution)
        self.pointwise_weights = np.random.normal(
            0, np.sqrt(2.0 / input_channels),
            (output_channels, input_channels, 1, 1)
        )
        
        # Bias terms
        self.depthwise_bias = np.zeros((input_channels, 1))
        self.pointwise_bias = np.zeros((output_channels, 1))
    
    def depthwise_conv2d(self, x: np.ndarray) -> np.ndarray:
        """
        Depthwise convolution: apply separate kernel to each input channel
        """
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Add padding
        padding = self.kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        
        # Output dimensions
        out_height = in_height
        out_width = in_width
        
        # Initialize output
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        
        # Apply depthwise convolution
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # Extract region
                        region = x_padded[b, c, h:h+self.kernel_size, w:w+self.kernel_size]
                        
                        # Apply kernel for this channel
                        output[b, c, h, w] = np.sum(region * self.depthwise_weights[c, 0]) + self.depthwise_bias[c]
        
        return output
    
    def pointwise_conv2d(self, x: np.ndarray) -> np.ndarray:
        """
        Pointwise convolution: 1x1 convolution to combine channels
        """
        batch_size, in_channels, height, width = x.shape
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, height, width))
        
        # Apply pointwise convolution
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for h in range(height):
                    for w in range(width):
                        # 1x1 convolution across all input channels
                        output[b, oc, h, w] = (
                            np.sum(x[b, :, h, w] * self.pointwise_weights[oc, :, 0, 0]) + 
                            self.pointwise_bias[oc]
                        )
        
        return output
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through depthwise separable convolution
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after depthwise and pointwise convolutions
        """
        # Depthwise convolution
        depthwise_out = self.depthwise_conv2d(x)
        
        # Pointwise convolution
        output = self.pointwise_conv2d(depthwise_out)
        
        return output


class VisionTransformerBlock:
    """
    Vision Transformer Block
    
    Implements the core building block of Vision Transformers (ViTs)
    which applies self-attention to image patches.
    """
    
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4):
        """
        Initialize ViT block
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = d_model * mlp_ratio
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalization parameters
        self.ln1_gamma = np.ones((d_model,))
        self.ln1_beta = np.zeros((d_model,))
        self.ln2_gamma = np.ones((d_model,))
        self.ln2_beta = np.zeros((d_model,))
        
        # MLP layers
        self.mlp_fc1 = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, self.mlp_dim))
        self.mlp_fc2 = np.random.normal(0, np.sqrt(2.0 / self.mlp_dim), (self.mlp_dim, d_model))
        self.mlp_bias1 = np.zeros((self.mlp_dim,))
        self.mlp_bias2 = np.zeros((d_model,))
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def mlp_forward(self, x: np.ndarray) -> np.ndarray:
        """MLP forward pass"""
        # First layer
        x = np.dot(x, self.mlp_fc1) + self.mlp_bias1
        x = self.gelu(x)
        
        # Second layer
        x = np.dot(x, self.mlp_fc2) + self.mlp_bias2
        
        return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ViT block
        
        Args:
            x: Input token sequences (batch_size, seq_len, d_model)
            
        Returns:
            Output sequences with same shape
        """
        # Multi-head attention with residual connection
        attention_input = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attention_output = self.attention.forward(attention_input)
        x = x + attention_output  # Residual connection
        
        # MLP with residual connection
        mlp_input = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)
        mlp_output = self.mlp_forward(mlp_input)
        x = x + mlp_output  # Residual connection
        
        return x


# ==========================================
# ARCHITECTURAL ANALYSIS TOOLS
# ==========================================

class ArchitectureAnalyzer:
    """
    Tools for analyzing and comparing different architectures
    """
    
    @staticmethod
    def count_parameters(layers: List) -> int:
        """Count total parameters in a list of layers"""
        total_params = 0
        
        for layer in layers:
            if hasattr(layer, 'conv1_weights'):
                # ResNet block
                total_params += np.prod(layer.conv1_weights.shape)
                total_params += np.prod(layer.conv2_weights.shape)
                if hasattr(layer, 'shortcut_weights'):
                    total_params += np.prod(layer.shortcut_weights.shape)
                    
            elif hasattr(layer, 'layers'):
                # DenseNet block
                for sub_layer in layer.layers:
                    total_params += np.prod(sub_layer['conv1x1_weights'].shape)
                    total_params += np.prod(sub_layer['conv3x3_weights'].shape)
                    
            elif hasattr(layer, 'W_q'):
                # Attention layer
                total_params += np.prod(layer.W_q.shape)
                total_params += np.prod(layer.W_k.shape)
                total_params += np.prod(layer.W_v.shape)
                total_params += np.prod(layer.W_o.shape)
                
            elif hasattr(layer, 'depthwise_weights'):
                # Depthwise separable conv
                total_params += np.prod(layer.depthwise_weights.shape)
                total_params += np.prod(layer.pointwise_weights.shape)
        
        return total_params
    
    @staticmethod
    def compute_flops(input_shape: Tuple[int, ...], architecture_type: str) -> int:
        """Estimate FLOPs for different architectures"""
        batch_size, channels, height, width = input_shape
        
        if architecture_type == "resnet_block":
            # Rough estimate for ResNet block
            conv1_flops = height * width * channels * channels * 3 * 3
            conv2_flops = height * width * channels * channels * 3 * 3
            return conv1_flops + conv2_flops
            
        elif architecture_type == "attention":
            seq_len = height * width  # Flattened spatial dimensions
            d_model = channels
            
            # Attention computation: Q*K^T, softmax, attention*V
            qk_flops = seq_len * seq_len * d_model
            av_flops = seq_len * seq_len * d_model
            projection_flops = 4 * seq_len * d_model * d_model  # Q,K,V,O projections
            
            return qk_flops + av_flops + projection_flops
            
        elif architecture_type == "depthwise_separable":
            # Depthwise: channels * height * width * kernel_size^2
            # Pointwise: height * width * in_channels * out_channels
            depthwise_flops = channels * height * width * 9  # 3x3 kernel
            pointwise_flops = height * width * channels * channels  # Assume same out channels
            
            return depthwise_flops + pointwise_flops
        
        return 0
    
    @staticmethod
    def analyze_memory_usage(input_shape: Tuple[int, ...], layer) -> Dict[str, int]:
        """Analyze memory usage of a layer"""
        batch_size, channels, height, width = input_shape
        
        # Input memory
        input_memory = batch_size * channels * height * width * 4  # 4 bytes per float
        
        # Parameter memory
        param_memory = 0
        if hasattr(layer, 'conv1_weights'):
            param_memory += np.prod(layer.conv1_weights.shape) * 4
            param_memory += np.prod(layer.conv2_weights.shape) * 4
        elif hasattr(layer, 'W_q'):
            param_memory += np.prod(layer.W_q.shape) * 4
            param_memory += np.prod(layer.W_k.shape) * 4
            param_memory += np.prod(layer.W_v.shape) * 4
            param_memory += np.prod(layer.W_o.shape) * 4
        
        # Activation memory (rough estimate)
        activation_memory = input_memory  # Assume similar size
        
        return {
            'input_memory': input_memory,
            'parameter_memory': param_memory,
            'activation_memory': activation_memory,
            'total_memory': input_memory + param_memory + activation_memory
        }


# ==========================================
# EXERCISE IMPLEMENTATIONS
# ==========================================

def exercise_1_residual_connections():
    """
    Exercise 1: Implement and understand ResNet residual blocks
    
    Build ResNet blocks from scratch and understand how skip connections
    enable training of very deep networks.
    """
    print("\n🏗️ Exercise 1: Residual Connections (ResNet)")
    print("=" * 60)
    
    # Create sample input (small for demonstration)
    batch_size, channels, height, width = 2, 16, 8, 8
    input_tensor = np.random.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Test basic residual block
    print("\n1. Basic Residual Block (same dimensions)")
    basic_block = ResidualBlock(input_channels=16, output_channels=16, stride=1)
    
    start_time = time.time()
    output_basic = basic_block.forward(input_tensor)
    forward_time = time.time() - start_time
    
    print(f"   Output shape: {output_basic.shape}")
    print(f"   Forward pass time: {forward_time:.4f} seconds")
    print(f"   Parameters: {ArchitectureAnalyzer.count_parameters([basic_block]):,}")
    
    # Test residual block with dimension change
    print("\n2. Residual Block with Dimension Change")
    dimension_block = ResidualBlock(input_channels=16, output_channels=32, stride=2)
    
    # Need to adjust input for stride=2
    downsample_input = input_tensor[:, :, ::2, ::2]  # Simple downsampling
    output_dimension = dimension_block.forward(downsample_input)
    
    print(f"   Input shape: {downsample_input.shape}")
    print(f"   Output shape: {output_dimension.shape}")
    print(f"   Uses shortcut conv: {dimension_block.use_shortcut_conv}")
    
    # Visualize the effect of residual connections
    print("\n3. Analyzing Residual Connection Benefits")
    
    # Compare with and without residual connection
    def simulate_gradient_flow(num_layers: int, with_residual: bool = True) -> List[float]:
        """Simulate gradient magnitudes through deep network"""
        gradients = [1.0]  # Start with gradient magnitude of 1
        
        for layer in range(num_layers):
            if with_residual:
                # With residual: gradient can flow directly
                grad_scale = np.random.uniform(0.8, 1.2)  # Small variation
            else:
                # Without residual: gradient gets multiplied by weights
                grad_scale = np.random.uniform(0.1, 0.9)  # More degradation
            
            gradients.append(gradients[-1] * grad_scale)
        
        return gradients
    
    # Simulate gradient flow for different depths
    depths = [5, 10, 20, 50]
    
    plt.figure(figsize=(15, 10))
    
    for i, depth in enumerate(depths):
        plt.subplot(2, 2, i + 1)
        
        # Without residual connections
        grad_without = simulate_gradient_flow(depth, with_residual=False)
        layers_without = list(range(len(grad_without)))
        
        # With residual connections
        grad_with = simulate_gradient_flow(depth, with_residual=True)
        layers_with = list(range(len(grad_with)))
        
        plt.plot(layers_without, grad_without, 'r-', label='Without Residual', linewidth=2)
        plt.plot(layers_with, grad_with, 'b-', label='With Residual', linewidth=2)
        
        plt.title(f'Gradient Flow - {depth} Layers')
        plt.xlabel('Layer Depth')
        plt.ylabel('Gradient Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.suptitle('Impact of Residual Connections on Gradient Flow')
    plt.tight_layout()
    plt.show()
    
    # Memory and computational analysis
    print("\n4. Computational Analysis")
    memory_analysis = ArchitectureAnalyzer.analyze_memory_usage(
        input_tensor.shape, basic_block
    )
    
    print("Memory Usage Analysis:")
    for key, value in memory_analysis.items():
        print(f"   {key}: {value/1024:.2f} KB")
    
    flops = ArchitectureAnalyzer.compute_flops(input_tensor.shape, "resnet_block")
    print(f"   Estimated FLOPs: {flops:,}")
    
    print("✅ Residual connections implemented and analyzed!")
    print("\n💡 Key Insights:")
    print("• Skip connections enable gradient flow through identity mappings")
    print("• Deeper networks become trainable without vanishing gradients")
    print("• Residual learning: F(x) + x allows learning residual functions")
    print("• Shortcut connections handle dimension mismatches")


def exercise_2_dense_connectivity():
    """
    Exercise 2: Implement DenseNet dense blocks
    
    Build DenseNet blocks that connect each layer to every other layer,
    promoting feature reuse and improving gradient flow.
    """
    print("\n🌐 Exercise 2: Dense Connectivity (DenseNet)")
    print("=" * 60)
    
    # Create sample input
    batch_size, channels, height, width = 2, 16, 8, 8
    input_tensor = np.random.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Test dense block with different growth rates
    growth_rates = [12, 24, 32]
    num_layers = 4
    
    plt.figure(figsize=(15, 10))
    
    for i, growth_rate in enumerate(growth_rates):
        print(f"\n{i+1}. Dense Block - Growth Rate: {growth_rate}")
        
        # Create dense block
        dense_block = DenseBlock(
            input_channels=channels, 
            growth_rate=growth_rate, 
            num_layers=num_layers
        )
        
        # Forward pass
        start_time = time.time()
        output = dense_block.forward(input_tensor)
        forward_time = time.time() - start_time
        
        expected_output_channels = channels + (growth_rate * num_layers)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output channels: {expected_output_channels}")
        print(f"   Forward pass time: {forward_time:.4f} seconds")
        print(f"   Parameters: {ArchitectureAnalyzer.count_parameters([dense_block]):,}")
        
        # Visualize channel growth
        plt.subplot(2, 2, i + 1)
        
        channel_counts = [channels]
        for layer in range(num_layers):
            channel_counts.append(channel_counts[-1] + growth_rate)
        
        layers = list(range(len(channel_counts)))
        plt.plot(layers, channel_counts, 'o-', linewidth=2, markersize=8)
        plt.title(f'Channel Growth - Growth Rate {growth_rate}')
        plt.xlabel('Layer')
        plt.ylabel('Number of Channels')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        for j, count in enumerate(channel_counts):
            plt.annotate(f'{count}', (j, count), textcoords="offset points", 
                        xytext=(0,10), ha='center')
    
    # Compare parameter efficiency
    plt.subplot(2, 2, 4)
    
    param_counts = []
    flop_counts = []
    
    for growth_rate in growth_rates:
        dense_block = DenseBlock(channels, growth_rate, num_layers)
        params = ArchitectureAnalyzer.count_parameters([dense_block])
        param_counts.append(params)
        
        # Estimate FLOPs (simplified)
        flops = growth_rate * num_layers * height * width * 100  # Rough estimate
        flop_counts.append(flops)
    
    x = np.arange(len(growth_rates))
    width_bar = 0.35
    
    plt.bar(x - width_bar/2, param_counts, width_bar, label='Parameters', alpha=0.8)
    plt.bar(x + width_bar/2, [f/1000 for f in flop_counts], width_bar, 
            label='FLOPs (K)', alpha=0.8)
    
    plt.xlabel('Growth Rate')
    plt.ylabel('Count')
    plt.title('Parameter vs FLOP Trade-offs')
    plt.xticks(x, growth_rates)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('DenseNet Dense Connectivity Analysis')
    plt.tight_layout()
    plt.show()
    
    # Feature reuse analysis
    print("\n3. Feature Reuse Analysis")
    
    def analyze_feature_reuse(growth_rate: int, num_layers: int):
        """Analyze how features are reused in dense blocks"""
        input_channels = 16
        
        # Track channel usage through layers
        layer_inputs = [input_channels]
        
        for layer in range(num_layers):
            # Each layer receives all previous features
            current_input = input_channels + layer * growth_rate
            layer_inputs.append(current_input)
        
        return layer_inputs
    
    # Compare feature reuse for different configurations
    configurations = [(12, 4), (24, 3), (32, 2)]
    
    plt.figure(figsize=(12, 5))
    
    for i, (gr, nl) in enumerate(configurations):
        inputs = analyze_feature_reuse(gr, nl)
        layers = list(range(len(inputs)))
        
        plt.subplot(1, 3, i + 1)
        plt.plot(layers, inputs, 'o-', linewidth=2, markersize=8)
        plt.title(f'Feature Reuse\nGR={gr}, Layers={nl}')
        plt.xlabel('Layer')
        plt.ylabel('Input Channels')
        plt.grid(True, alpha=0.3)
        
        # Calculate reuse ratio
        total_new_features = gr * nl
        total_reused_features = sum(inputs) - inputs[0] - total_new_features
        reuse_ratio = total_reused_features / (total_reused_features + total_new_features)
        
        plt.text(0.5, 0.95, f'Reuse Ratio: {reuse_ratio:.2f}', 
                transform=plt.gca().transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Feature Reuse Patterns in Dense Blocks')
    plt.tight_layout()
    plt.show()
    
    print("✅ Dense connectivity implemented and analyzed!")
    print("\n💡 Key Insights:")
    print("• Dense connections maximize feature reuse")
    print("• Growth rate controls parameter vs accuracy trade-off")
    print("• Each layer receives gradients from all subsequent layers")
    print("• Bottleneck layers (1x1 conv) reduce computational cost")


def exercise_3_attention_mechanisms():
    """
    Exercise 3: Implement attention mechanisms from scratch
    
    Build multi-head attention and understand how it enables models
    to focus on relevant parts of the input.
    """
    print("\n👁️ Exercise 3: Attention Mechanisms")
    print("=" * 60)
    
    # Create sample sequence data
    batch_size, seq_len, d_model = 2, 10, 64
    input_sequence = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"Input sequence shape: {input_sequence.shape}")
    
    # Test different attention configurations
    attention_configs = [
        (64, 1),   # Single head
        (64, 4),   # Multi-head
        (64, 8),   # More heads
    ]
    
    print("\n1. Multi-Head Attention Comparison")
    
    attention_outputs = []
    computation_times = []
    
    for d_model, num_heads in attention_configs:
        print(f"\n   Configuration: d_model={d_model}, num_heads={num_heads}")
        
        # Create attention layer
        attention = MultiHeadAttention(d_model, num_heads)
        
        # Forward pass
        start_time = time.time()
        output = attention.forward(input_sequence)
        forward_time = time.time() - start_time
        
        print(f"   Output shape: {output.shape}")
        print(f"   Forward time: {forward_time:.4f} seconds")
        print(f"   Parameters: {ArchitectureAnalyzer.count_parameters([attention]):,}")
        
        attention_outputs.append(output)
        computation_times.append(forward_time)
        
        # Analyze attention patterns
        if num_heads == 4:  # Detailed analysis for 4-head case
            print("   Analyzing attention patterns...")
            
            # Get attention weights for visualization
            Q = np.dot(input_sequence, attention.W_q)
            K = np.dot(input_sequence, attention.W_k)
            
            # Reshape for multi-head
            Q = Q.reshape(batch_size, seq_len, num_heads, attention.d_k).transpose(0, 2, 1, 3)
            K = K.reshape(batch_size, seq_len, num_heads, attention.d_k).transpose(0, 2, 1, 3)
            
            # Compute attention scores for first head
            scores = np.matmul(Q[0, 0], K[0, 0].T) / np.sqrt(attention.d_k)
            attention_weights = attention.softmax(scores)
            
            # Visualize attention pattern
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.imshow(attention_weights, cmap='Blues')
            plt.title('Attention Weights (Head 1)')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.colorbar()
            
            # Analyze attention distribution
            plt.subplot(2, 2, 2)
            attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-9), axis=1)
            plt.plot(attention_entropy, 'o-', linewidth=2)
            plt.title('Attention Entropy by Position')
            plt.xlabel('Query Position')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
            
            # Show attention focus patterns
            plt.subplot(2, 2, 3)
            max_attention_pos = np.argmax(attention_weights, axis=1)
            plt.plot(max_attention_pos, 'o-', linewidth=2, color='red')
            plt.plot(range(seq_len), range(seq_len), '--', alpha=0.5, color='gray', label='Diagonal')
            plt.title('Peak Attention Positions')
            plt.xlabel('Query Position')
            plt.ylabel('Peak Key Position')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Computational complexity analysis
            plt.subplot(2, 2, 4)
            seq_lengths = [5, 10, 20, 50, 100]
            attention_flops = [length**2 * d_model for length in seq_lengths]
            
            plt.loglog(seq_lengths, attention_flops, 'o-', linewidth=2)
            plt.title('Attention Complexity O(n²)')
            plt.xlabel('Sequence Length')
            plt.ylabel('FLOPs')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle('Multi-Head Attention Analysis')
            plt.tight_layout()
            plt.show()
    
    # Compare computational efficiency
    print("\n2. Computational Efficiency Analysis")
    
    plt.figure(figsize=(12, 5))
    
    # Execution time comparison
    plt.subplot(1, 2, 1)
    configs = [f'{h} heads' for _, h in attention_configs]
    plt.bar(configs, computation_times, alpha=0.8, color='skyblue')
    plt.title('Forward Pass Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Parameter count comparison
    plt.subplot(1, 2, 2)
    param_counts = []
    for d_model, num_heads in attention_configs:
        attention = MultiHeadAttention(d_model, num_heads)
        params = ArchitectureAnalyzer.count_parameters([attention])
        param_counts.append(params)
    
    plt.bar(configs, param_counts, alpha=0.8, color='lightcoral')
    plt.title('Parameter Count')
    plt.ylabel('Parameters')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Mechanism Efficiency')
    plt.tight_layout()
    plt.show()
    
    # Demonstrate attention vs convolution
    print("\n3. Attention vs Convolution Comparison")
    
    def compare_receptive_fields():
        """Compare receptive fields of attention vs convolution"""
        seq_len = 20
        
        # Attention: every position can attend to every other position
        attention_receptive = np.ones((seq_len, seq_len))
        
        # Convolution: local receptive field grows with layers
        conv_receptive = np.zeros((seq_len, seq_len))
        kernel_size = 3
        
        for i in range(seq_len):
            start = max(0, i - kernel_size//2)
            end = min(seq_len, i + kernel_size//2 + 1)
            conv_receptive[i, start:end] = 1
        
        return attention_receptive, conv_receptive
    
    att_rf, conv_rf = compare_receptive_fields()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(conv_rf, cmap='Reds', alpha=0.8)
    plt.title('Convolution Receptive Field')
    plt.xlabel('Input Position')
    plt.ylabel('Output Position')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(att_rf, cmap='Blues', alpha=0.8)
    plt.title('Attention Receptive Field')
    plt.xlabel('Input Position')
    plt.ylabel('Output Position')
    plt.colorbar()
    
    plt.suptitle('Receptive Field Comparison')
    plt.tight_layout()
    plt.show()
    
    print("✅ Attention mechanisms implemented and analyzed!")
    print("\n💡 Key Insights:")
    print("• Attention enables global receptive fields")
    print("• Multi-head attention captures different types of relationships")
    print("• Computational complexity is O(n²) in sequence length")
    print("• Attention weights provide interpretability")


def exercise_4_efficient_architectures():
    """
    Exercise 4: Implement efficient architectures
    
    Build MobileNet-style depthwise separable convolutions and analyze
    efficiency trade-offs in neural network design.
    """
    print("\n⚡ Exercise 4: Efficient Architectures")
    print("=" * 60)
    
    # Create sample input
    batch_size, channels, height, width = 2, 32, 16, 16
    input_tensor = np.random.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Compare standard conv vs depthwise separable conv
    print("\n1. Standard vs Depthwise Separable Convolution")
    
    def standard_conv_params(in_channels: int, out_channels: int, kernel_size: int) -> int:
        """Calculate parameters for standard convolution"""
        return in_channels * out_channels * kernel_size * kernel_size + out_channels
    
    def depthwise_separable_params(in_channels: int, out_channels: int, kernel_size: int) -> int:
        """Calculate parameters for depthwise separable convolution"""
        depthwise_params = in_channels * kernel_size * kernel_size + in_channels
        pointwise_params = in_channels * out_channels + out_channels
        return depthwise_params + pointwise_params
    
    # Compare for different channel configurations
    configurations = [
        (32, 64, 3),
        (64, 128, 3),
        (128, 256, 3),
        (256, 512, 3)
    ]
    
    standard_params = []
    depthwise_params = []
    param_ratios = []
    
    for in_ch, out_ch, kernel in configurations:
        std_params = standard_conv_params(in_ch, out_ch, kernel)
        dw_params = depthwise_separable_params(in_ch, out_ch, kernel)
        
        standard_params.append(std_params)
        depthwise_params.append(dw_params)
        param_ratios.append(dw_params / std_params)
        
        print(f"   {in_ch}→{out_ch}: Standard={std_params:,}, Depthwise={dw_params:,}, "
              f"Ratio={dw_params/std_params:.3f}")
    
    # Visualize parameter efficiency
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    x = range(len(configurations))
    config_labels = [f'{in_ch}→{out_ch}' for in_ch, out_ch, _ in configurations]
    
    width = 0.35
    plt.bar([i - width/2 for i in x], standard_params, width, label='Standard Conv', alpha=0.8)
    plt.bar([i + width/2 for i in x], depthwise_params, width, label='Depthwise Sep', alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Parameters')
    plt.title('Parameter Comparison')
    plt.xticks(x, config_labels, rotation=45)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Parameter reduction ratio
    plt.subplot(2, 3, 2)
    plt.plot(x, param_ratios, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Configuration')
    plt.ylabel('Parameter Ratio (DS/Standard)')
    plt.title('Parameter Reduction')
    plt.xticks(x, config_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Reduction')
    plt.legend()
    
    # Test actual implementation
    print("\n2. Implementing Depthwise Separable Convolution")
    
    # Create depthwise separable conv layer
    ds_conv = DepthwiseSeparableConv(input_channels=32, output_channels=64, kernel_size=3)
    
    start_time = time.time()
    output = ds_conv.forward(input_tensor)
    forward_time = time.time() - start_time
    
    print(f"   Output shape: {output.shape}")
    print(f"   Forward time: {forward_time:.4f} seconds")
    print(f"   Parameters: {ArchitectureAnalyzer.count_parameters([ds_conv]):,}")
    
    # Analyze computational efficiency
    plt.subplot(2, 3, 3)
    
    # Compare FLOPs for different input sizes
    input_sizes = [16, 32, 64, 128]
    standard_flops = []
    depthwise_flops = []
    
    for size in input_sizes:
        # Standard conv FLOPs: output_size * kernel_size^2 * in_channels * out_channels
        std_flops = size * size * 9 * 32 * 64  # 3x3 kernel
        
        # Depthwise separable FLOPs
        dw_flops = size * size * 9 * 32  # Depthwise
        pw_flops = size * size * 32 * 64  # Pointwise
        ds_flops = dw_flops + pw_flops
        
        standard_flops.append(std_flops)
        depthwise_flops.append(ds_flops)
    
    plt.plot(input_sizes, standard_flops, 'o-', label='Standard Conv', linewidth=2)
    plt.plot(input_sizes, depthwise_flops, 's-', label='Depthwise Sep', linewidth=2)
    plt.xlabel('Input Size')
    plt.ylabel('FLOPs')
    plt.title('Computational Efficiency')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Mobile deployment analysis
    plt.subplot(2, 3, 4)
    
    # Simulate different efficiency metrics
    metrics = ['Parameters', 'FLOPs', 'Memory', 'Latency']
    standard_scores = [100, 100, 100, 100]  # Baseline
    depthwise_scores = [25, 30, 35, 40]  # Relative to standard (lower is better)
    
    x_metrics = range(len(metrics))
    plt.bar([i - width/2 for i in x_metrics], standard_scores, width, 
            label='Standard Conv', alpha=0.8, color='red')
    plt.bar([i + width/2 for i in x_metrics], depthwise_scores, width, 
            label='Depthwise Sep', alpha=0.8, color='green')
    plt.xlabel('Efficiency Metric')
    plt.ylabel('Relative Cost')
    plt.title('Mobile Deployment Efficiency')
    plt.xticks(x_metrics, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Width multiplier analysis
    plt.subplot(2, 3, 5)
    
    width_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25]
    base_channels = 64
    
    param_counts = []
    accuracy_estimates = []  # Simulated
    
    for mult in width_multipliers:
        channels = int(base_channels * mult)
        params = depthwise_separable_params(channels, channels, 3)
        param_counts.append(params)
        
        # Simulate accuracy (higher channels = better accuracy, with diminishing returns)
        accuracy = 90 + 8 * (1 - np.exp(-2 * mult))
        accuracy_estimates.append(accuracy)
    
    # Dual y-axis plot
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(width_multipliers, param_counts, 'b-o', linewidth=2, label='Parameters')
    line2 = ax2.plot(width_multipliers, accuracy_estimates, 'r-s', linewidth=2, label='Accuracy')
    
    ax1.set_xlabel('Width Multiplier')
    ax1.set_ylabel('Parameters', color='blue')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax1.set_title('Width Multiplier Trade-offs')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.grid(True, alpha=0.3)
    
    # Efficiency frontier analysis
    plt.subplot(2, 3, 6)
    
    # Plot efficiency frontier
    plt.scatter(param_counts, accuracy_estimates, s=100, alpha=0.7, c=width_multipliers, 
               cmap='viridis')
    
    for i, mult in enumerate(width_multipliers):
        plt.annotate(f'{mult}x', (param_counts[i], accuracy_estimates[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy (%)')
    plt.title('Efficiency Frontier')
    plt.colorbar(label='Width Multiplier')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Efficient Architecture Analysis')
    plt.tight_layout()
    plt.show()
    
    # Architecture scaling analysis
    print("\n3. Architecture Scaling Analysis")
    
    def analyze_scaling_strategies():
        """Analyze different ways to scale neural networks"""
        base_params = 1000000  # 1M parameters
        base_accuracy = 85.0
        
        strategies = {
            'Depth Scaling': {
                'params': [base_params * (1 + 0.5*i) for i in range(5)],
                'accuracy': [base_accuracy + 2*np.sqrt(i) for i in range(5)]
            },
            'Width Scaling': {
                'params': [base_params * (1 + 0.8*i) for i in range(5)],
                'accuracy': [base_accuracy + 3*np.sqrt(i) - 0.5*i for i in range(5)]
            },
            'Resolution Scaling': {
                'params': [base_params * (1 + 0.3*i) for i in range(5)],
                'accuracy': [base_accuracy + 1.5*np.sqrt(i) for i in range(5)]
            },
            'Compound Scaling': {
                'params': [base_params * (1 + 0.6*i) for i in range(5)],
                'accuracy': [base_accuracy + 4*np.sqrt(i) - 0.3*i for i in range(5)]
            }
        }
        
        return strategies
    
    scaling_data = analyze_scaling_strategies()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for strategy, data in scaling_data.items():
        plt.plot(data['params'], data['accuracy'], 'o-', label=strategy, linewidth=2, markersize=6)
    
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Scaling Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Efficiency ratio analysis
    plt.subplot(1, 2, 2)
    
    efficiency_ratios = {}
    for strategy, data in scaling_data.items():
        # Calculate accuracy gain per parameter increase
        acc_gains = [data['accuracy'][i] - data['accuracy'][0] for i in range(1, 5)]
        param_increases = [data['params'][i] - data['params'][0] for i in range(1, 5)]
        ratios = [acc/param*1000000 for acc, param in zip(acc_gains, param_increases)]
        efficiency_ratios[strategy] = np.mean(ratios)
    
    strategies = list(efficiency_ratios.keys())
    ratios = list(efficiency_ratios.values())
    
    plt.bar(strategies, ratios, alpha=0.8, color=['blue', 'red', 'green', 'orange'])
    plt.ylabel('Accuracy Gain per Million Parameters')
    plt.title('Scaling Efficiency')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Efficient architectures implemented and analyzed!")
    print("\n💡 Key Insights:")
    print("• Depthwise separable convolutions reduce parameters by ~8-9x")
    print("• Width multipliers provide fine-grained efficiency control")
    print("• Compound scaling balances depth, width, and resolution")
    print("• Mobile deployment requires careful parameter-accuracy trade-offs")


def exercise_5_vision_transformers():
    """
    Exercise 5: Implement Vision Transformer components
    
    Build Vision Transformer blocks and understand how attention mechanisms
    can be applied to computer vision tasks.
    """
    print("\n🤖 Exercise 5: Vision Transformers")
    print("=" * 60)
    
    # Simulate image patches as sequence
    batch_size, num_patches, d_model = 2, 64, 256  # 8x8 patches from 32x32 image
    patch_sequence = np.random.randn(batch_size, num_patches, d_model)
    
    print(f"Patch sequence shape: {patch_sequence.shape}")
    print(f"Representing {int(np.sqrt(num_patches))}x{int(np.sqrt(num_patches))} image patches")
    
    # Test Vision Transformer block
    print("\n1. Vision Transformer Block")
    
    vit_block = VisionTransformerBlock(d_model=256, num_heads=8, mlp_ratio=4)
    
    start_time = time.time()
    output = vit_block.forward(patch_sequence)
    forward_time = time.time() - start_time
    
    print(f"   Output shape: {output.shape}")
    print(f"   Forward time: {forward_time:.4f} seconds")
    print(f"   Parameters: {ArchitectureAnalyzer.count_parameters([vit_block]):,}")
    
    # Analyze attention patterns in ViT
    print("\n2. Vision Transformer Attention Analysis")
    
    # Get attention weights from the attention mechanism
    attention_layer = vit_block.attention
    
    # Simulate getting attention weights (simplified)
    Q = np.dot(patch_sequence, attention_layer.W_q)
    K = np.dot(patch_sequence, attention_layer.W_k)
    
    # Reshape for multi-head
    Q = Q.reshape(batch_size, num_patches, attention_layer.num_heads, attention_layer.d_k)
    K = K.reshape(batch_size, num_patches, attention_layer.num_heads, attention_layer.d_k)
    
    # Compute attention for first head
    scores = np.matmul(Q[0, :, 0, :], K[0, :, 0, :].T) / np.sqrt(attention_layer.d_k)
    attention_weights = attention_layer.softmax(scores)
    
    # Visualize attention patterns
    plt.figure(figsize=(15, 10))
    
    # Attention map visualization
    plt.subplot(2, 3, 1)
    plt.imshow(attention_weights, cmap='Blues')
    plt.title('Attention Weights (Head 1)')
    plt.xlabel('Key Patch')
    plt.ylabel('Query Patch')
    plt.colorbar()
    
    # Reshape attention to spatial layout
    patch_size = int(np.sqrt(num_patches))
    spatial_attention = attention_weights.reshape(patch_size, patch_size, patch_size, patch_size)
    
    # Show attention from center patch
    center_patch = patch_size // 2
    plt.subplot(2, 3, 2)
    center_attention = spatial_attention[center_patch, center_patch].reshape(patch_size, patch_size)
    plt.imshow(center_attention, cmap='Reds')
    plt.title(f'Attention from Center Patch ({center_patch},{center_patch})')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.colorbar()
    
    # Show attention from corner patch
    plt.subplot(2, 3, 3)
    corner_attention = spatial_attention[0, 0].reshape(patch_size, patch_size)
    plt.imshow(corner_attention, cmap='Greens')
    plt.title('Attention from Corner Patch (0,0)')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.colorbar()
    
    # Analyze attention distance patterns
    plt.subplot(2, 3, 4)
    
    def compute_attention_distances():
        """Compute average attention distance for each patch"""
        distances = []
        
        for i in range(patch_size):
            for j in range(patch_size):
                query_idx = i * patch_size + j
                patch_attention = attention_weights[query_idx]
                
                # Compute weighted average distance
                total_distance = 0
                total_weight = 0
                
                for ki in range(patch_size):
                    for kj in range(patch_size):
                        key_idx = ki * patch_size + kj
                        distance = np.sqrt((i - ki)**2 + (j - kj)**2)
                        weight = patch_attention[key_idx]
                        
                        total_distance += distance * weight
                        total_weight += weight
                
                avg_distance = total_distance / total_weight if total_weight > 0 else 0
                distances.append(avg_distance)
        
        return np.array(distances).reshape(patch_size, patch_size)
    
    attention_distances = compute_attention_distances()
    im = plt.imshow(attention_distances, cmap='viridis')
    plt.title('Average Attention Distance')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.colorbar(im)
    
    # Compare with CNN receptive fields
    plt.subplot(2, 3, 5)
    
    # Simulate CNN receptive field growth
    def simulate_cnn_receptive_field(num_layers: int, kernel_size: int = 3):
        """Simulate how CNN receptive field grows with depth"""
        rf_sizes = [1]  # Start with 1x1
        
        for layer in range(num_layers):
            new_rf = rf_sizes[-1] + kernel_size - 1
            rf_sizes.append(new_rf)
        
        return rf_sizes
    
    cnn_layers = range(10)
    cnn_rf_sizes = simulate_cnn_receptive_field(9)
    vit_rf_sizes = [patch_size] * 10  # ViT has global receptive field from layer 1
    
    plt.plot(cnn_layers, cnn_rf_sizes, 'o-', label='CNN', linewidth=2)
    plt.plot(cnn_layers, vit_rf_sizes, 's-', label='ViT', linewidth=2)
    plt.xlabel('Layer Depth')
    plt.ylabel('Receptive Field Size')
    plt.title('Receptive Field Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Computational complexity comparison
    plt.subplot(2, 3, 6)
    
    sequence_lengths = [16, 64, 256, 1024]  # Different numbers of patches
    
    # ViT complexity: O(n²) for attention
    vit_complexity = [n**2 * d_model for n in sequence_lengths]
    
    # CNN complexity: O(n) for convolution (assuming constant kernel size)
    cnn_complexity = [n * d_model * 9 for n in sequence_lengths]  # 3x3 kernels
    
    plt.loglog(sequence_lengths, vit_complexity, 'o-', label='ViT (O(n²))', linewidth=2)
    plt.loglog(sequence_lengths, cnn_complexity, 's-', label='CNN (O(n))', linewidth=2)
    plt.xlabel('Number of Patches')
    plt.ylabel('Computational Complexity')
    plt.title('Complexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Vision Transformer Analysis')
    plt.tight_layout()
    plt.show()
    
    # Position encoding analysis
    print("\n3. Position Encoding Analysis")
    
    def create_positional_encoding(num_patches: int, d_model: int) -> np.ndarray:
        """Create 2D positional encoding for image patches"""
        patch_size = int(np.sqrt(num_patches))
        pos_encoding = np.zeros((num_patches, d_model))
        
        for i in range(patch_size):
            for j in range(patch_size):
                patch_idx = i * patch_size + j
                
                # Use different frequencies for x and y coordinates
                for k in range(0, d_model, 4):
                    # X coordinate encoding
                    pos_encoding[patch_idx, k] = np.sin(i / (10000 ** (k / d_model)))
                    if k + 1 < d_model:
                        pos_encoding[patch_idx, k + 1] = np.cos(i / (10000 ** (k / d_model)))
                    
                    # Y coordinate encoding
                    if k + 2 < d_model:
                        pos_encoding[patch_idx, k + 2] = np.sin(j / (10000 ** (k / d_model)))
                    if k + 3 < d_model:
                        pos_encoding[patch_idx, k + 3] = np.cos(j / (10000 ** (k / d_model)))
        
        return pos_encoding
    
    pos_encoding = create_positional_encoding(num_patches, d_model)
    
    plt.figure(figsize=(12, 8))
    
    # Visualize position encoding patterns
    plt.subplot(2, 3, 1)
    plt.imshow(pos_encoding[:, :32], aspect='auto', cmap='RdBu')
    plt.title('Position Encoding (First 32 dims)')
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Patch Index')
    plt.colorbar()
    
    # Show spatial patterns in position encoding
    for i, dim in enumerate([0, 16, 32]):
        plt.subplot(2, 3, i + 2)
        if dim < d_model:
            spatial_encoding = pos_encoding[:, dim].reshape(patch_size, patch_size)
            plt.imshow(spatial_encoding, cmap='RdBu')
            plt.title(f'Position Encoding Dim {dim}')
            plt.xlabel('Patch X')
            plt.ylabel('Patch Y')
            plt.colorbar()
    
    # Compare patch similarities
    plt.subplot(2, 3, 5)
    
    # Compute cosine similarity between position encodings
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Select a reference patch (center)
    ref_patch = num_patches // 2
    similarities = []
    
    for patch_idx in range(num_patches):
        sim = cosine_similarity(pos_encoding[ref_patch], pos_encoding[patch_idx])
        similarities.append(sim)
    
    similarities = np.array(similarities).reshape(patch_size, patch_size)
    plt.imshow(similarities, cmap='viridis')
    plt.title(f'Position Similarity to Center Patch')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.colorbar()
    
    # Training dynamics comparison
    plt.subplot(2, 3, 6)
    
    # Simulate training curves for different architectures
    epochs = np.arange(1, 101)
    
    # ViT typically needs more epochs to converge
    vit_accuracy = 90 * (1 - np.exp(-epochs / 30)) + np.random.normal(0, 1, len(epochs))
    
    # CNN converges faster initially
    cnn_accuracy = 90 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 1, len(epochs))
    
    # Smooth the curves
    from scipy.ndimage import gaussian_filter1d
    vit_smooth = gaussian_filter1d(vit_accuracy, sigma=2)
    cnn_smooth = gaussian_filter1d(cnn_accuracy, sigma=2)
    
    plt.plot(epochs, cnn_smooth, label='CNN', linewidth=2)
    plt.plot(epochs, vit_smooth, label='ViT', linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Vision Transformer Position Encoding Analysis')
    plt.tight_layout()
    plt.show()
    
    print("✅ Vision Transformers implemented and analyzed!")
    print("\n💡 Key Insights:")
    print("• ViTs treat images as sequences of patches")
    print("• Global attention enables long-range spatial relationships")
    print("• Position encoding is crucial for spatial understanding")
    print("• ViTs typically require more data and compute than CNNs")


def exercise_6_architecture_comparison():
    """
    Exercise 6: Comprehensive architecture comparison
    
    Compare different architectural paradigms and analyze their trade-offs
    in terms of performance, efficiency, and applicability.
    """
    print("\n📊 Exercise 6: Architecture Comparison")
    print("=" * 60)
    
    # Define architecture characteristics
    architectures = {
        'VGGNet': {
            'params': 138000000,
            'flops': 15300000000,
            'accuracy': 71.5,
            'memory': 500,
            'year': 2014,
            'paradigm': 'CNN'
        },
        'ResNet-50': {
            'params': 25600000,
            'flops': 4100000000,
            'accuracy': 76.2,
            'memory': 200,
            'year': 2015,
            'paradigm': 'CNN'
        },
        'MobileNet-v2': {
            'params': 3500000,
            'flops': 300000000,
            'accuracy': 72.0,
            'memory': 50,
            'year': 2018,
            'paradigm': 'CNN'
        },
        'EfficientNet-B0': {
            'params': 5300000,
            'flops': 390000000,
            'accuracy': 77.1,
            'memory': 60,
            'year': 2019,
            'paradigm': 'CNN'
        },
        'ViT-Base': {
            'params': 86000000,
            'flops': 17600000000,
            'accuracy': 77.9,
            'memory': 400,
            'year': 2020,
            'paradigm': 'Transformer'
        },
        'DeiT-Small': {
            'params': 22000000,
            'flops': 4600000000,
            'accuracy': 79.8,
            'memory': 180,
            'year': 2020,
            'paradigm': 'Transformer'
        }
    }
    
    print("1. Multi-dimensional Architecture Analysis")
    
    # Create comprehensive comparison
    plt.figure(figsize=(20, 15))
    
    # Accuracy vs Parameters
    plt.subplot(3, 4, 1)
    for name, data in architectures.items():
        color = 'blue' if data['paradigm'] == 'CNN' else 'red'
        plt.scatter(data['params']/1e6, data['accuracy'], s=100, alpha=0.7, 
                   c=color, label=data['paradigm'] if name == list(architectures.keys())[0] or name == 'ViT-Base' else "")
        plt.annotate(name, (data['params']/1e6, data['accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy vs FLOPs
    plt.subplot(3, 4, 2)
    for name, data in architectures.items():
        color = 'blue' if data['paradigm'] == 'CNN' else 'red'
        plt.scatter(data['flops']/1e9, data['accuracy'], s=100, alpha=0.7, c=color)
        plt.annotate(name, (data['flops']/1e9, data['accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('FLOPs (Billions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs FLOPs')
    plt.grid(True, alpha=0.3)
    
    # Memory vs Accuracy
    plt.subplot(3, 4, 3)
    for name, data in architectures.items():
        color = 'blue' if data['paradigm'] == 'CNN' else 'red'
        plt.scatter(data['memory'], data['accuracy'], s=100, alpha=0.7, c=color)
        plt.annotate(name, (data['memory'], data['accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Memory (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Memory vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Evolution over time
    plt.subplot(3, 4, 4)
    for name, data in architectures.items():
        color = 'blue' if data['paradigm'] == 'CNN' else 'red'
        plt.scatter(data['year'], data['accuracy'], s=data['params']/1e6*2, 
                   alpha=0.7, c=color)
        plt.annotate(name, (data['year'], data['accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Evolution Over Time (bubble size = params)')
    plt.grid(True, alpha=0.3)
    
    # Efficiency frontier analysis
    plt.subplot(3, 4, 5)
    
    # Calculate efficiency scores
    efficiency_scores = {}
    for name, data in architectures.items():
        # Higher accuracy, lower params/flops = better efficiency
        efficiency = data['accuracy'] / (data['params']/1e6)
        efficiency_scores[name] = efficiency
    
    names = list(efficiency_scores.keys())
    scores = list(efficiency_scores.values())
    colors = ['blue' if architectures[name]['paradigm'] == 'CNN' else 'red' for name in names]
    
    plt.bar(range(len(names)), scores, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy per Million Parameters')
    plt.title('Parameter Efficiency')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # FLOP efficiency
    plt.subplot(3, 4, 6)
    
    flop_efficiency = {}
    for name, data in architectures.items():
        efficiency = data['accuracy'] / (data['flops']/1e9)
        flop_efficiency[name] = efficiency
    
    scores = list(flop_efficiency.values())
    plt.bar(range(len(names)), scores, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy per Billion FLOPs')
    plt.title('Computational Efficiency')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Paradigm comparison
    plt.subplot(3, 4, 7)
    
    cnn_archs = {k: v for k, v in architectures.items() if v['paradigm'] == 'CNN'}
    transformer_archs = {k: v for k, v in architectures.items() if v['paradigm'] == 'Transformer'}
    
    cnn_metrics = {
        'avg_accuracy': np.mean([v['accuracy'] for v in cnn_archs.values()]),
        'avg_params': np.mean([v['params']/1e6 for v in cnn_archs.values()]),
        'avg_flops': np.mean([v['flops']/1e9 for v in cnn_archs.values()]),
        'avg_memory': np.mean([v['memory'] for v in cnn_archs.values()])
    }
    
    transformer_metrics = {
        'avg_accuracy': np.mean([v['accuracy'] for v in transformer_archs.values()]),
        'avg_params': np.mean([v['params']/1e6 for v in transformer_archs.values()]),
        'avg_flops': np.mean([v['flops']/1e9 for v in transformer_archs.values()]),
        'avg_memory': np.mean([v['memory'] for v in transformer_archs.values()])
    }
    
    metrics = ['Accuracy', 'Params (M)', 'FLOPs (B)', 'Memory (MB)']
    cnn_values = [cnn_metrics['avg_accuracy'], cnn_metrics['avg_params'], 
                  cnn_metrics['avg_flops'], cnn_metrics['avg_memory']]
    transformer_values = [transformer_metrics['avg_accuracy'], transformer_metrics['avg_params'],
                         transformer_metrics['avg_flops'], transformer_metrics['avg_memory']]
    
    # Normalize values for radar chart
    max_values = [max(cnn_values[i], transformer_values[i]) for i in range(len(metrics))]
    cnn_norm = [cnn_values[i] / max_values[i] for i in range(len(metrics))]
    transformer_norm = [transformer_values[i] / max_values[i] for i in range(len(metrics))]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, cnn_norm, width, label='CNN', alpha=0.8)
    plt.bar(x + width/2, transformer_norm, width, label='Transformer', alpha=0.8)
    plt.xlabel('Metric')
    plt.ylabel('Normalized Value')
    plt.title('Paradigm Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mobile deployment suitability
    plt.subplot(3, 4, 8)
    
    mobile_scores = {}
    for name, data in architectures.items():
        # Mobile score: high accuracy, low params, low memory, low flops
        param_score = max(0, 1 - data['params']/50e6)  # Penalize >50M params
        memory_score = max(0, 1 - data['memory']/100)  # Penalize >100MB
        flop_score = max(0, 1 - data['flops']/5e9)     # Penalize >5B FLOPs
        accuracy_bonus = data['accuracy'] / 100
        
        mobile_score = (param_score + memory_score + flop_score + accuracy_bonus) / 4
        mobile_scores[name] = mobile_score
    
    names = list(mobile_scores.keys())
    scores = list(mobile_scores.values())
    colors = ['blue' if architectures[name]['paradigm'] == 'CNN' else 'red' for name in names]
    
    plt.bar(range(len(names)), scores, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Mobile Deployment Score')
    plt.title('Mobile Suitability')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Deployment scenario analysis
    plt.subplot(3, 4, 9)
    
    scenarios = ['Data Center', 'Edge Device', 'Mobile Phone', 'Embedded System']
    scenario_requirements = {
        'Data Center': {'params_weight': 0.1, 'flops_weight': 0.2, 'accuracy_weight': 0.7},
        'Edge Device': {'params_weight': 0.3, 'flops_weight': 0.4, 'accuracy_weight': 0.3},
        'Mobile Phone': {'params_weight': 0.4, 'flops_weight': 0.4, 'accuracy_weight': 0.2},
        'Embedded System': {'params_weight': 0.5, 'flops_weight': 0.4, 'accuracy_weight': 0.1}
    }
    
    best_archs = []
    for scenario, weights in scenario_requirements.items():
        scores = {}
        for name, data in architectures.items():
            # Normalize metrics
            param_norm = 1 - (data['params'] / max([d['params'] for d in architectures.values()]))
            flop_norm = 1 - (data['flops'] / max([d['flops'] for d in architectures.values()]))
            acc_norm = data['accuracy'] / max([d['accuracy'] for d in architectures.values()])
            
            score = (param_norm * weights['params_weight'] + 
                    flop_norm * weights['flops_weight'] + 
                    acc_norm * weights['accuracy_weight'])
            scores[name] = score
        
        best_arch = max(scores, key=scores.get)
        best_archs.append(best_arch)
    
    # Count recommendations
    from collections import Counter
    recommendations = Counter(best_archs)
    
    rec_names = list(recommendations.keys())
    rec_counts = list(recommendations.values())
    colors = ['blue' if architectures[name]['paradigm'] == 'CNN' else 'red' for name in rec_names]
    
    plt.bar(range(len(rec_names)), rec_counts, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Recommendation Count')
    plt.title('Deployment Recommendations')
    plt.xticks(range(len(rec_names)), rec_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Future trends prediction
    plt.subplot(3, 4, 10)
    
    years = np.arange(2014, 2025)
    
    # Extrapolate trends
    cnn_years = [data['year'] for data in architectures.values() if data['paradigm'] == 'CNN']
    cnn_accs = [data['accuracy'] for data in architectures.values() if data['paradigm'] == 'CNN']
    
    transformer_years = [data['year'] for data in architectures.values() if data['paradigm'] == 'Transformer']
    transformer_accs = [data['accuracy'] for data in architectures.values() if data['paradigm'] == 'Transformer']
    
    # Fit simple trend lines
    if len(cnn_years) > 1:
        cnn_trend = np.polyfit(cnn_years, cnn_accs, 1)
        cnn_projection = np.polyval(cnn_trend, years)
    else:
        cnn_projection = [cnn_accs[0]] * len(years)
    
    if len(transformer_years) > 1:
        transformer_trend = np.polyfit(transformer_years, transformer_accs, 1)
        transformer_projection = np.polyval(transformer_trend, years)
    else:
        transformer_projection = [transformer_accs[0]] * len(years)
    
    plt.plot(years, cnn_projection, 'b-', label='CNN Trend', linewidth=2)
    plt.plot(years, transformer_projection, 'r-', label='Transformer Trend', linewidth=2)
    
    # Plot actual data points
    for name, data in architectures.items():
        color = 'blue' if data['paradigm'] == 'CNN' else 'red'
        plt.scatter(data['year'], data['accuracy'], s=80, alpha=0.8, c=color)
        plt.annotate(name, (data['year'], data['accuracy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Architecture Evolution Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(2014, 2024)
    
    # Architecture design principles
    plt.subplot(3, 4, 11)
    
    design_principles = ['Depth', 'Width', 'Efficiency', 'Attention', 'Skip Connections']
    principle_importance = {
        'VGGNet': [0.9, 0.3, 0.1, 0.0, 0.0],
        'ResNet-50': [0.8, 0.4, 0.3, 0.0, 0.9],
        'MobileNet-v2': [0.5, 0.3, 0.9, 0.0, 0.7],
        'EfficientNet-B0': [0.7, 0.6, 0.9, 0.0, 0.8],
        'ViT-Base': [0.6, 0.7, 0.2, 1.0, 0.9],
        'DeiT-Small': [0.5, 0.6, 0.6, 1.0, 0.9]
    }
    
    # Create heatmap
    heatmap_data = []
    arch_names = []
    for name, principles in principle_importance.items():
        heatmap_data.append(principles)
        arch_names.append(name)
    
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    plt.xticks(range(len(design_principles)), design_principles, rotation=45)
    plt.yticks(range(len(arch_names)), arch_names)
    plt.title('Design Principle Importance')
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(len(arch_names)):
        for j in range(len(design_principles)):
            text = plt.text(j, i, f'{heatmap_data[i][j]:.1f}',
                           ha="center", va="center", color="black" if heatmap_data[i][j] < 0.5 else "white")
    
    # Research impact analysis
    plt.subplot(3, 4, 12)
    
    # Simulate citation counts and research impact
    impact_scores = {
        'VGGNet': 8500,
        'ResNet-50': 15000,
        'MobileNet-v2': 3500,
        'EfficientNet-B0': 2800,
        'ViT-Base': 4200,
        'DeiT-Small': 1800
    }
    
    names = list(impact_scores.keys())
    scores = list(impact_scores.values())
    colors = ['blue' if architectures[name]['paradigm'] == 'CNN' else 'red' for name in names]
    
    plt.bar(range(len(names)), scores, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Citation Count (Approximate)')
    plt.title('Research Impact')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Architecture Analysis Dashboard', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Detailed trade-off analysis
    print("\n2. Detailed Trade-off Analysis")
    
    # Create trade-off matrix
    trade_offs = {
        'Accuracy vs Parameters': [],
        'Accuracy vs FLOPs': [],
        'Accuracy vs Memory': [],
        'Efficiency Score': []
    }
    
    arch_names = list(architectures.keys())
    
    for name in arch_names:
        data = architectures[name]
        
        # Calculate various trade-off metrics
        acc_per_param = data['accuracy'] / (data['params'] / 1e6)
        acc_per_flop = data['accuracy'] / (data['flops'] / 1e9)
        acc_per_memory = data['accuracy'] / data['memory']
        
        # Combined efficiency score
        efficiency = (acc_per_param + acc_per_flop + acc_per_memory) / 3
        
        trade_offs['Accuracy vs Parameters'].append(acc_per_param)
        trade_offs['Accuracy vs FLOPs'].append(acc_per_flop)
        trade_offs['Accuracy vs Memory'].append(acc_per_memory)
        trade_offs['Efficiency Score'].append(efficiency)
    
    # Print detailed comparison table
    print("\nArchitecture Comparison Table:")
    print("=" * 100)
    print(f"{'Architecture':<15} {'Year':<6} {'Params(M)':<10} {'FLOPs(B)':<10} {'Memory(MB)':<11} {'Accuracy':<9} {'Paradigm':<12}")
    print("-" * 100)
    
    for name, data in architectures.items():
        print(f"{name:<15} {data['year']:<6} {data['params']/1e6:<10.1f} {data['flops']/1e9:<10.1f} "
              f"{data['memory']:<11.0f} {data['accuracy']:<9.1f} {data['paradigm']:<12}")
    
    print("\nEfficiency Rankings:")
    print("=" * 50)
    
    # Rank architectures by different metrics
    rankings = {}
    for metric, values in trade_offs.items():
        sorted_indices = np.argsort(values)[::-1]  # Descending order
        rankings[metric] = [(arch_names[i], values[i]) for i in sorted_indices]
    
    for metric, ranking in rankings.items():
        print(f"\n{metric}:")
        for i, (name, score) in enumerate(ranking):
            print(f"  {i+1}. {name:<15} ({score:.2f})")
    
    print("✅ Comprehensive architecture comparison completed!")
    print("\n💡 Key Insights:")
    print("• ResNet revolutionized deep learning with skip connections")
    print("• MobileNet optimizes for mobile deployment efficiency")
    print("• Vision Transformers show promise but require more resources")
    print("• EfficientNet provides excellent accuracy-efficiency balance")
    print("• Architecture choice depends heavily on deployment constraints")


def comprehensive_architecture_demonstration():
    """
    Comprehensive demonstration of all deep learning architectures
    """
    print("\n" + "="*80)
    print("🧠 NEURAL ODYSSEY - WEEK 29: DEEP LEARNING ARCHITECTURES")
    print("="*80)
    print("\nWelcome to the evolution of neural network architectures!")
    print("This week you'll implement and understand the key innovations")
    print("that have shaped modern deep learning.")
    
    print("\n🎯 Learning Objectives:")
    print("• Understand residual connections and their impact")
    print("• Implement dense connectivity patterns")
    print("• Build attention mechanisms from scratch")
    print("• Explore efficient architecture design")
    print("• Compare architectural paradigms and trade-offs")
    
    # Run all exercises
    try:
        exercise_1_residual_connections()
        exercise_2_dense_connectivity()
        exercise_3_attention_mechanisms()
        exercise_4_efficient_architectures()
        exercise_5_vision_transformers()
        exercise_6_architecture_comparison()
        
        print("\n" + "="*80)
        print("🎉 CONGRATULATIONS! DEEP LEARNING ARCHITECTURE MASTERY ACHIEVED!")
        print("="*80)
        print("\n✨ You have successfully:")
        print("• ✅ Implemented ResNet residual blocks from scratch")
        print("• ✅ Built DenseNet dense connectivity patterns")
        print("• ✅ Created multi-head attention mechanisms")
        print("• ✅ Developed efficient depthwise separable convolutions")
        print("• ✅ Constructed Vision Transformer components")
        print("• ✅ Analyzed architectural trade-offs comprehensively")
        
        print("\n🚀 Next Steps in Your Deep Learning Journey:")
        print("• Explore Generative Adversarial Networks (GANs)")
        print("• Study advanced optimization techniques")
        print("• Learn about neural architecture search (NAS)")
        print("• Investigate multimodal architectures")
        print("• Apply architectures to real-world problems")
        
        print("\n💡 Key Architectural Insights Gained:")
        print("• Skip connections enable very deep network training")
        print("• Dense connectivity maximizes information flow")
        print("• Attention mechanisms provide global receptive fields")
        print("• Efficient designs balance accuracy and computational cost")
        print("• Architecture choice depends on deployment constraints")
        
        print("\n🔬 Advanced Concepts Mastered:")
        print("• Vanishing gradient problem and solutions")
        print("• Feature reuse and information flow")
        print("• Computational complexity analysis")
        print("• Mobile and edge deployment considerations")
        print("• Evolution of architectural paradigms")
        
        print("\n🎓 You now understand the foundations of modern AI architectures!")
        print("This knowledge will guide you in designing and optimizing")
        print("neural networks for any application domain.")
        
    except Exception as e:
        print(f"\n❌ Error in architecture demonstration: {e}")
        print("Review the implementation and try again!")


# ==========================================
# BONUS: ARCHITECTURE DESIGN CHALLENGE
# ==========================================

def bonus_architecture_design_challenge():
    """
    Bonus: Design your own architecture
    
    Challenge users to design a novel architecture combining
    insights from different paradigms.
    """
    print("\n🌟 BONUS: Architecture Design Challenge")
    print("=" * 60)
    
    print("Design Challenge: Hybrid Vision Architecture")
    print("Combine the best aspects of different paradigms:")
    print("• ResNet skip connections for gradient flow")
    print("• DenseNet feature reuse for efficiency")
    print("• Attention for global relationships")
    print("• Depthwise separable convs for mobile deployment")
    
    class HybridVisionBlock:
        """
        Hybrid architecture combining multiple paradigms
        """
        
        def __init__(self, input_channels: int, growth_rate: int, num_heads: int):
            self.input_channels = input_channels
            self.growth_rate = growth_rate
            self.num_heads = num_heads
            
            # Depthwise separable conv for efficiency
            self.ds_conv = DepthwiseSeparableConv(input_channels, growth_rate)
            
            # Channel attention for feature selection
            self.attention = MultiHeadAttention(growth_rate, num_heads)
            
            # Dense connection weights
            self.dense_weights = np.random.normal(
                0, np.sqrt(2.0 / (input_channels + growth_rate)),
                (input_channels + growth_rate, growth_rate)
            )
        
        def forward(self, x: np.ndarray, previous_features: List[np.ndarray]) -> np.ndarray:
            """Forward pass through hybrid block"""
            # Concatenate with previous features (DenseNet-style)
            if previous_features:
                dense_input = np.concatenate([x] + previous_features, axis=1)
            else:
                dense_input = x
            
            # Efficient convolution
            conv_out = self.ds_conv.forward(dense_input)
            
            # Reshape for attention (spatial -> sequence)
            batch_size, channels, height, width = conv_out.shape
            seq_input = conv_out.reshape(batch_size, channels, height * width).transpose(0, 2, 1)
            
            # Apply attention
            attended = self.attention.forward(seq_input)
            
            # Reshape back to spatial
            spatial_out = attended.transpose(0, 2, 1).reshape(batch_size, channels, height, width)
            
            # Residual connection if dimensions match
            if spatial_out.shape == x.shape:
                output = spatial_out + x
            else:
                output = spatial_out
            
            return output
    
    # Test hybrid architecture
    print("\nTesting Hybrid Architecture:")
    
    batch_size, channels, height, width = 1, 32, 8, 8
    input_tensor = np.random.randn(batch_size, channels, height, width)
    
    hybrid_block = HybridVisionBlock(input_channels=32, growth_rate=16, num_heads=4)
    
    # Simulate multiple blocks with growing feature connections
    features = [input_tensor]
    current_input = input_tensor
    
    for block_idx in range(3):
        print(f"   Block {block_idx + 1}:")
        
        start_time = time.time()
        output = hybrid_block.forward(current_input, features[:-1])  # Exclude current input
        forward_time = time.time() - start_time
        
        print(f"     Input shape: {current_input.shape}")
        print(f"     Output shape: {output.shape}")
        print(f"     Forward time: {forward_time:.4f} seconds")
        
        features.append(output)
        current_input = output
    
    # Analyze hybrid architecture properties
    total_params = ArchitectureAnalyzer.count_parameters([hybrid_block])
    memory_usage = ArchitectureAnalyzer.analyze_memory_usage(input_tensor.shape, hybrid_block)
    
    print(f"\nHybrid Architecture Analysis:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Memory usage: {memory_usage['total_memory']/1024:.2f} KB")
    
    # Compare with individual paradigms
    paradigm_comparison = {
        'Pure CNN': {'params': 50000, 'efficiency': 0.6, 'flexibility': 0.4},
        'Pure Attention': {'params': 80000, 'efficiency': 0.4, 'flexibility': 0.9},
        'Hybrid Design': {'params': total_params, 'efficiency': 0.7, 'flexibility': 0.8}
    }
    
    plt.figure(figsize=(10, 6))
    
    paradigms = list(paradigm_comparison.keys())
    metrics = ['params', 'efficiency', 'flexibility']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        values = [paradigm_comparison[p][metric] for p in paradigms]
        
        if metric == 'params':
            plt.bar(paradigms, [v/1000 for v in values], alpha=0.8)
            plt.ylabel('Parameters (K)')
        else:
            plt.bar(paradigms, values, alpha=0.8)
            plt.ylabel(metric.capitalize())
        
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Hybrid vs Pure Paradigm Comparison')
    plt.tight_layout()
    plt.show()
    
    print("✅ Hybrid architecture design challenge completed!")
    print("\n🏆 Design Principles Demonstrated:")
    print("• Combine complementary strengths from different paradigms")
    print("• Balance efficiency with expressiveness")
    print("• Maintain gradient flow through residual connections")
    print("• Optimize for target deployment constraints")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🧠 Neural Odyssey - Week 29: Deep Learning Architectures")
    print("=" * 60)
    print("Choose an exercise to run:")
    print("1. Residual Connections (ResNet)")
    print("2. Dense Connectivity (DenseNet)")
    print("3. Attention Mechanisms")
    print("4. Efficient Architectures")
    print("5. Vision Transformers")
    print("6. Architecture Comparison")
    print("7. Comprehensive Demo (All Exercises)")
    print("8. Bonus: Architecture Design Challenge")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("👋 Continue your architectural journey! Keep innovating!")
                break
            elif choice == '1':
                exercise_1_residual_connections()
            elif choice == '2':
                exercise_2_dense_connectivity()
            elif choice == '3':
                exercise_3_attention_mechanisms()
            elif choice == '4':
                exercise_4_efficient_architectures()
            elif choice == '5':
                exercise_5_vision_transformers()
            elif choice == '6':
                exercise_6_architecture_comparison()
            elif choice == '7':
                comprehensive_architecture_demonstration()
            elif choice == '8':
                bonus_architecture_design_challenge()
            else:
                print("❌ Invalid choice. Please enter 0-8.")
                continue
                
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for exploring deep learning architectures! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or choose a different exercise.")


# ==========================================
# FINAL SUMMARY AND REFLECTION
# ==========================================

def print_week_summary():
    """Print comprehensive summary of Week 29 learning"""
    print("\n" + "🎯"*20)
    print("WEEK 29 COMPLETE: DEEP LEARNING ARCHITECTURES")
    print("🎯"*20)
    
    print("\n📈 Architectural Innovations Mastered:")
    print("✅ Residual connections and skip connections")
    print("✅ Dense connectivity and feature reuse")
    print("✅ Multi-head attention mechanisms")
    print("✅ Efficient depthwise separable convolutions")
    print("✅ Vision Transformer components")
    print("✅ Architectural design principles")
    
    print("\n🏆 Implementation Achievements:")
    print("🥇 Built ResNet blocks from scratch")
    print("🥈 Implemented DenseNet connectivity patterns")
    print("🥉 Created attention mechanisms")
    print("🏅 Developed efficient mobile architectures")
    print("🎖️ Constructed Vision Transformer blocks")
    print("🏆 Designed hybrid architectures")
    
    print("\n🚀 Ready for Advanced Topics:")
    print("📍 Generative Adversarial Networks (GANs)")
    print("📍 Variational Autoencoders (VAEs)")
    print("📍 Neural Architecture Search (NAS)")
    print("📍 Multimodal Learning")
    print("📍 Self-Supervised Learning")
    
    print(f"\n💡 Total Lines of Code: ~{3000}+")
    print("💡 Architectural Concepts: ResNet, DenseNet, Attention, Efficiency")
    print("💡 Deep Learning: Modern architectures, Design principles, Trade-offs")
    
    print("\n🌟 You've mastered the evolution of neural architectures!")
    print("🌟 Your understanding spans from CNNs to Transformers!")
    print("🌟 You can now design architectures for any application!")


# Execute summary at module level
if __name__ == "__main__":
    # Uncomment to see week summary
    # print_week_summary()
    pass