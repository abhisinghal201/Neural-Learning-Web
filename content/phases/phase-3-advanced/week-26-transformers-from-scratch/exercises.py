"""
Neural Odyssey - Week 26: Transformers from Scratch - The Complete Architecture
Phase 3: Advanced Topics and Modern AI (Week 2)

"Attention Is All You Need" - Building the Architecture That Changed Everything

Welcome to the most influential neural network architecture of the 21st century! This week, 
you'll build the complete transformer architecture from scratch, understanding every component 
that makes modern AI possible. From BERT to GPT to ChatGPT - they all trace back to the 
transformer you'll implement this week.

Building on last week's attention mechanism mastery, you'll now assemble the complete puzzle:
encoder-decoder architecture, layer normalization, positional encoding, feed-forward networks,
and the training dynamics that make it all work together.

Journey Overview:
1. Complete transformer architecture: From paper to working implementation
2. Encoder stack: Self-attention, normalization, and feed-forward layers
3. Decoder stack: Masked self-attention, cross-attention, and autoregressive generation
4. Training dynamics: How transformers learn language and patterns
5. Scaling laws: Understanding why bigger transformers work better
6. Modern variants: BERT, GPT, T5, and their architectural innovations
7. Implementation optimizations: Making transformers fast and memory-efficient
8. Real-world applications: Translation, text generation, and beyond

By week's end, you'll have built a complete transformer that can learn language patterns,
translate text, and generate coherent sequences - the foundation of modern AI systems.

Author: Neural Explorer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Union
import math
import copy
from dataclasses import dataclass
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("🎯 Neural Odyssey - Week 26: Transformers from Scratch")
print("=" * 60)
print("🏗️ 'Attention Is All You Need' - Building the Complete Architecture")
print("🚀 The foundation of BERT, GPT, ChatGPT, and modern AI!")
print("=" * 60)


# ==========================================
# SECTION 1: TRANSFORMER BUILDING BLOCKS
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism (refined from Week 25)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads if needed
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs
    """
    
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        # Create the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_length, d_model]
        
    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_length, :]
        
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # First linear transformation + ReLU
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization with learnable parameters
    """
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        # Compute mean and std along the feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize and apply learnable parameters
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def demonstrate_transformer_components():
    """
    Demonstrate each transformer component individually
    """
    print("\n🧱 Transformer Building Blocks Demonstration")
    print("=" * 50)
    
    # Configuration
    batch_size = 2
    seq_length = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # Test input
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, seq_length, d_model)
    
    print(f"📊 Test Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of heads: {num_heads}")
    print(f"   Feed-forward dimension: {d_ff}")
    print()
    
    # 1. Positional Encoding
    print("1️⃣ Positional Encoding:")
    pos_encoding = PositionalEncoding(d_model, max_length=1000)
    
    with torch.no_grad():
        encoded_input = pos_encoding(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {encoded_input.shape}")
    print(f"   ✅ Positional encoding applied successfully")
    
    # 2. Multi-Head Attention
    print("\n2️⃣ Multi-Head Attention:")
    mha = MultiHeadAttention(d_model, num_heads)
    
    with torch.no_grad():
        attn_output, attn_weights = mha(encoded_input, encoded_input, encoded_input)
    
    print(f"   Input shape: {encoded_input.shape}")
    print(f"   Output shape: {attn_output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   ✅ Multi-head attention working")
    
    # 3. Layer Normalization
    print("\n3️⃣ Layer Normalization:")
    layer_norm = LayerNorm(d_model)
    
    with torch.no_grad():
        norm_output = layer_norm(attn_output)
        
        # Show normalization effect
        original_mean = torch.mean(attn_output, dim=-1)
        original_std = torch.std(attn_output, dim=-1)
        norm_mean = torch.mean(norm_output, dim=-1)
        norm_std = torch.std(norm_output, dim=-1)
    
    print(f"   Original mean range: [{original_mean.min():.3f}, {original_mean.max():.3f}]")
    print(f"   Original std range: [{original_std.min():.3f}, {original_std.max():.3f}]")
    print(f"   Normalized mean range: [{norm_mean.min():.3f}, {norm_mean.max():.3f}]")
    print(f"   Normalized std range: [{norm_std.min():.3f}, {norm_std.max():.3f}]")
    print(f"   ✅ Layer normalization stabilizing activations")
    
    # 4. Feed-Forward Network
    print("\n4️⃣ Position-wise Feed-Forward:")
    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    with torch.no_grad():
        ffn_output = ffn(norm_output)
    
    print(f"   Input shape: {norm_output.shape}")
    print(f"   Hidden dimension: {d_ff}")
    print(f"   Output shape: {ffn_output.shape}")
    print(f"   ✅ Feed-forward network expanding and projecting")
    
    # Visualize component effects
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Positional encoding patterns
    plt.subplot(2, 4, 1)
    pe_sample = pos_encoding.pe[0, :50, :20].numpy()  # First 50 positions, first 20 dims
    plt.imshow(pe_sample.T, cmap='RdBu', aspect='auto')
    plt.title('Positional Encoding Patterns', fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    
    # Plot 2: Attention weights heatmap
    plt.subplot(2, 4, 2)
    attn_sample = attn_weights[0, 0].numpy()  # First batch, first head
    sns.heatmap(attn_sample, cmap='Blues', square=True, cbar=True)
    plt.title('Attention Weights (Head 1)', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Plot 3: Layer normalization effect
    plt.subplot(2, 4, 3)
    sample_pos = 0  # First sequence position
    original_features = attn_output[0, sample_pos, :50].numpy()
    normalized_features = norm_output[0, sample_pos, :50].numpy()
    
    x = np.arange(50)
    plt.plot(x, original_features, 'r-', label='Original', alpha=0.7, linewidth=2)
    plt.plot(x, normalized_features, 'b-', label='Normalized', alpha=0.7, linewidth=2)
    plt.title('Layer Normalization Effect', fontweight='bold')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Feed-forward transformation
    plt.subplot(2, 4, 4)
    ffn_input_sample = norm_output[0, sample_pos, :50].numpy()
    ffn_output_sample = ffn_output[0, sample_pos, :50].numpy()
    
    plt.plot(x, ffn_input_sample, 'g-', label='FFN Input', alpha=0.7, linewidth=2)
    plt.plot(x, ffn_output_sample, 'purple', label='FFN Output', alpha=0.7, linewidth=2)
    plt.title('Feed-Forward Transformation', fontweight='bold')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Multiple attention heads
    plt.subplot(2, 4, 5)
    head_entropies = []
    for head in range(min(8, num_heads)):
        head_attn = attn_weights[0, head].numpy()
        # Compute entropy for each query position
        entropies = []
        for i in range(seq_length):
            entropy = -np.sum(head_attn[i] * np.log(head_attn[i] + 1e-8))
            entropies.append(entropy)
        head_entropies.append(np.mean(entropies))
    
    plt.bar(range(len(head_entropies)), head_entropies, alpha=0.7, color='skyblue')
    plt.title('Attention Entropy by Head', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Average Entropy')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Residual connection importance
    plt.subplot(2, 4, 6)
    # Simulate residual connections
    residual_input = encoded_input[0, sample_pos, :50].numpy()
    attn_delta = (attn_output - encoded_input)[0, sample_pos, :50].numpy()
    final_after_residual = (encoded_input + attn_output)[0, sample_pos, :50].numpy()
    
    plt.plot(x, residual_input, 'b-', label='Input', alpha=0.7, linewidth=2)
    plt.plot(x, attn_delta, 'r-', label='Attention Δ', alpha=0.7, linewidth=2)
    plt.plot(x, final_after_residual, 'g-', label='After Residual', alpha=0.7, linewidth=2)
    plt.title('Residual Connection Effect', fontweight='bold')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Component output distributions
    plt.subplot(2, 4, 7)
    components = {
        'Input': test_input.flatten().numpy(),
        'After PE': encoded_input.flatten().numpy(),
        'After Attn': attn_output.flatten().numpy(),
        'After Norm': norm_output.flatten().numpy(),
        'After FFN': ffn_output.flatten().numpy()
    }
    
    for i, (name, values) in enumerate(components.items()):
        plt.hist(values, bins=30, alpha=0.6, label=name, density=True)
    
    plt.title('Output Distribution by Component', fontweight='bold')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Model capacity visualization
    plt.subplot(2, 4, 8)
    params_per_component = {
        'Positional Encoding': 0,  # No learnable parameters
        'Multi-Head Attention': 4 * d_model * d_model,  # Q, K, V, O projections
        'Layer Norm': 2 * d_model,  # gamma and beta
        'Feed-Forward': d_model * d_ff + d_ff * d_model,  # Two linear layers
    }
    
    labels = list(params_per_component.keys())
    sizes = list(params_per_component.values())
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Parameter Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎯 Component Analysis Summary:")
    print(f"   Total parameters per layer: {sum(params_per_component.values()):,}")
    print(f"   Attention parameters: {params_per_component['Multi-Head Attention']:,} ({params_per_component['Multi-Head Attention']/sum(params_per_component.values())*100:.1f}%)")
    print(f"   Feed-forward parameters: {params_per_component['Feed-Forward']:,} ({params_per_component['Feed-Forward']/sum(params_per_component.values())*100:.1f}%)")
    print(f"   Attention contributes most parameters but FFN does most computation")
    
    return {
        'positional_encoding': pos_encoding,
        'multi_head_attention': mha,
        'layer_norm': layer_norm,
        'feed_forward': ffn,
        'component_outputs': {
            'input': test_input,
            'pos_encoded': encoded_input,
            'attention': attn_output,
            'normalized': norm_output,
            'ffn': ffn_output
        },
        'attention_weights': attn_weights,
        'parameter_counts': params_per_component
    }


# ==========================================
# SECTION 2: TRANSFORMER ENCODER
# ==========================================

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm variant)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm self-attention with residual connection
        normalized_x = self.norm1(x)
        attn_output, attn_weights = self.self_attention(normalized_x, normalized_x, normalized_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward with residual connection
        normalized_x = self.norm2(x)
        ffn_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ffn_output)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Complete transformer encoder with multiple layers
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
        self.num_layers = num_layers
        
    def forward(self, x, mask=None):
        attention_weights = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Final normalization
        x = self.norm(x)
        
        return x, attention_weights


def demonstrate_transformer_encoder():
    """
    Demonstrate the complete transformer encoder
    """
    print("\n🏗️ Transformer Encoder Demonstration")
    print("=" * 50)
    
    # Configuration
    batch_size = 2
    seq_length = 16
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    print(f"📊 Encoder Configuration:")
    print(f"   Number of layers: {num_layers}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of heads: {num_heads}")
    print(f"   Feed-forward dimension: {d_ff}")
    print(f"   Dropout rate: {dropout}")
    print()
    
    # Create encoder
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
    
    # Add positional encoding
    pos_encoding = PositionalEncoding(d_model, max_length=1000, dropout=dropout)
    
    # Test input (simulate token embeddings)
    torch.manual_seed(42)
    token_embeddings = torch.randn(batch_size, seq_length, d_model)
    
    # Apply positional encoding
    encoder_input = pos_encoding(token_embeddings)
    
    # Forward pass through encoder
    with torch.no_grad():
        encoder_output, all_attention_weights = encoder(encoder_input)
    
    print(f"✅ Encoder forward pass successful:")
    print(f"   Input shape: {encoder_input.shape}")
    print(f"   Output shape: {encoder_output.shape}")
    print(f"   Attention weights per layer: {len(all_attention_weights)}")
    print(f"   Each attention tensor shape: {all_attention_weights[0].shape}")
    
    # Analyze encoder behavior
    print(f"\n🔍 Encoder Analysis:")
    
    # Track representation changes through layers
    layer_representations = []
    current_x = encoder_input
    
    with torch.no_grad():
        for i, layer in enumerate(encoder.layers):
            current_x, attn_weights = layer(current_x)
            layer_representations.append(current_x.clone())
    
    # Compute representation similarity between layers
    layer_similarities = []
    for i in range(len(layer_representations) - 1):
        repr1 = layer_representations[i][0].flatten()  # First batch item
        repr2 = layer_representations[i + 1][0].flatten()
        similarity = F.cosine_similarity(repr1, repr2, dim=0)
        layer_similarities.append(similarity.item())
    
    print(f"   Layer-to-layer similarity:")
    for i, sim in enumerate(layer_similarities):
        print(f"     Layer {i+1} → {i+2}: {sim:.4f}")
    
    # Analyze attention patterns across layers
    attention_entropies = []
    for layer_idx, attn_weights in enumerate(all_attention_weights):
        # Average entropy across heads and positions
        layer_entropy = []
        for head in range(num_heads):
            for pos in range(seq_length):
                attn_dist = attn_weights[0, head, pos]  # First batch item
                entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
                layer_entropy.append(entropy.item())
        attention_entropies.append(np.mean(layer_entropy))
    
    print(f"\n   Attention entropy by layer:")
    for i, entropy in enumerate(attention_entropies):
        print(f"     Layer {i+1}: {entropy:.3f} {'(more focused)' if entropy < 2.0 else '(more diffuse)'}")
    
    # Visualize encoder behavior
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Attention patterns across layers
    plt.subplot(2, 4, 1)
    for layer_idx in range(min(6, num_layers)):
        attn_matrix = all_attention_weights[layer_idx][0, 0].numpy()  # First batch, first head
        plt.subplot(2, 3, layer_idx + 1)
        sns.heatmap(attn_matrix, cmap='Blues', cbar=True, square=True)
        plt.title(f'Layer {layer_idx + 1} Attention', fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Layer-wise representation similarity
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(layer_similarities) + 1), layer_similarities, 'bo-', linewidth=2, markersize=8)
    plt.title('Layer-to-Layer Similarity', fontweight='bold')
    plt.xlabel('Layer Transition')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Attention entropy evolution
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(attention_entropies) + 1), attention_entropies, 'ro-', linewidth=2, markersize=8)
    plt.title('Attention Entropy by Layer', fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('Average Entropy')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Representation norm evolution
    plt.subplot(2, 3, 3)
    layer_norms = []
    for repr_tensor in layer_representations:
        norm = torch.norm(repr_tensor[0], dim=-1).mean().item()  # Average norm across positions
        layer_norms.append(norm)
    
    plt.plot(range(1, len(layer_norms) + 1), layer_norms, 'go-', linewidth=2, markersize=8)
    plt.title('Representation Norm by Layer', fontweight='bold')
    plt.xlabel('Layer')
    plt.ylabel('Average L2 Norm')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Head specialization analysis
    plt.subplot(2, 3, 4)
    head_entropies_by_layer = np.zeros((num_layers, num_heads))
    
    for layer_idx, attn_weights in enumerate(all_attention_weights):
        for head in range(num_heads):
            head_entropy = []
            for pos in range(seq_length):
                attn_dist = attn_weights[0, head, pos]
                entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
                head_entropy.append(entropy.item())
            head_entropies_by_layer[layer_idx, head] = np.mean(head_entropy)
    
    sns.heatmap(head_entropies_by_layer, cmap='viridis', cbar=True)
    plt.title('Head Entropy Across Layers', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    
    # Plot 5: Parameter distribution
    plt.subplot(2, 3, 5)
    total_params = sum(p.numel() for p in encoder.parameters())
    layer_params = sum(p.numel() for p in encoder.layers[0].parameters())
    
    components = ['Self-Attention', 'Feed-Forward', 'Layer Norm']
    # Approximate parameter distribution within a layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = d_model * d_ff + d_ff * d_model  # Two linear layers
    norm_params = 4 * d_model  # Two layer norms per layer
    
    sizes = [attn_params, ffn_params, norm_params]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    plt.pie(sizes, labels=components, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Parameter Distribution per Layer', fontweight='bold')
    
    # Plot 6: Computational complexity
    plt.subplot(2, 3, 6)
    seq_lengths = [16, 32, 64, 128, 256]
    
    
    # Attention complexity: O(n²d)
    attn_complexity = [n**2 * d_model for n in seq_lengths]
    # FFN complexity: O(nd²)
    ffn_complexity = [n * d_model * d_ff for n in seq_lengths]
    
    plt.loglog(seq_lengths, attn_complexity, 'b-', label='Attention O(n²d)', linewidth=2)
    plt.loglog(seq_lengths, ffn_complexity, 'r-', label='FFN O(nd²)', linewidth=2)
    plt.title('Computational Complexity', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Operations (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Encoder Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Parameters per layer: {layer_params:,}")
    print(f"   Memory usage (approx): {total_params * 4 / 1e6:.1f} MB")
    print(f"   Attention parameters: {attn_params * num_layers:,}")
    print(f"   Feed-forward parameters: {ffn_params * num_layers:,}")
    
    return {
        'encoder': encoder,
        'positional_encoding': pos_encoding,
        'encoder_output': encoder_output,
        'attention_weights': all_attention_weights,
        'layer_similarities': layer_similarities,
        'attention_entropies': attention_entropies,
        'total_parameters': total_params
    }


# ==========================================
# SECTION 3: TRANSFORMER DECODER
# ==========================================

class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with masked self-attention and cross-attention
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Three sub-layers in decoder
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm variant)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Masked self-attention with residual connection
        normalized_x = self.norm1(x)
        self_attn_output, self_attn_weights = self.self_attention(
            normalized_x, normalized_x, normalized_x, self_attn_mask
        )
        x = x + self.dropout(self_attn_output)
        
        # Cross-attention with encoder output
        normalized_x = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            normalized_x, encoder_output, encoder_output, cross_attn_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # Feed-forward with residual connection
        normalized_x = self.norm3(x)
        ffn_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ffn_output)
        
        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Complete transformer decoder with multiple layers
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNorm(d_model)
        
        self.num_layers = num_layers
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        self_attention_weights = []
        cross_attention_weights = []
        
        # Pass through each decoder layer
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, self_attn_mask, cross_attn_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)
        
        # Final normalization
        x = self.norm(x)
        
        return x, self_attention_weights, cross_attention_weights


def create_causal_mask(seq_length, device):
    """
    Create causal mask for autoregressive generation
    """
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    return mask  # 1 for allowed positions, 0 for masked positions


def demonstrate_transformer_decoder():
    """
    Demonstrate the complete transformer decoder
    """
    print("\n🎯 Transformer Decoder Demonstration")
    print("=" * 50)
    
    # Configuration
    batch_size = 2
    encoder_seq_length = 16
    decoder_seq_length = 12
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    print(f"📊 Decoder Configuration:")
    print(f"   Number of layers: {num_layers}")
    print(f"   Model dimension: {d_model}")
    print(f"   Encoder sequence length: {encoder_seq_length}")
    print(f"   Decoder sequence length: {decoder_seq_length}")
    print(f"   Causal masking: Enabled")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create decoder
    decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
    
    # Create test inputs
    torch.manual_seed(42)
    
    # Encoder output (from previous encoder demonstration)
    encoder_output = torch.randn(batch_size, encoder_seq_length, d_model)
    
    # Decoder input (target sequence with positional encoding)
    decoder_input = torch.randn(batch_size, decoder_seq_length, d_model)
    
    # Create causal mask for decoder self-attention
    causal_mask = create_causal_mask(decoder_seq_length, device)
    print(f"🎭 Causal Mask Created:")
    print(f"   Shape: {causal_mask.shape}")
    print(f"   Allows attention to previous positions only")
    
    # Visualize causal mask
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(causal_mask.numpy(), cmap='RdBu', aspect='auto')
    plt.title('Causal Mask Pattern', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Allowed (1) / Masked (0)')
    
    # Forward pass through decoder
    with torch.no_grad():
        decoder_output, self_attn_weights, cross_attn_weights = decoder(
            decoder_input, encoder_output, self_attn_mask=causal_mask
        )
    
    print(f"\n✅ Decoder forward pass successful:")
    print(f"   Decoder input shape: {decoder_input.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Decoder output shape: {decoder_output.shape}")
    print(f"   Self-attention weights per layer: {len(self_attn_weights)}")
    print(f"   Cross-attention weights per layer: {len(cross_attn_weights)}")
    
    # Analyze decoder attention patterns
    print(f"\n🔍 Decoder Attention Analysis:")
    
    # Self-attention analysis (should show causal pattern)
    plt.subplot(1, 3, 2)
    self_attn_sample = self_attn_weights[0][0, 0].numpy()  # First layer, first batch, first head
    sns.heatmap(self_attn_sample, cmap='Blues', cbar=True, square=True)
    plt.title('Decoder Self-Attention\n(Causal Pattern)', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Cross-attention analysis (decoder attending to encoder)
    plt.subplot(1, 3, 3)
    cross_attn_sample = cross_attn_weights[0][0, 0].numpy()  # First layer, first batch, first head
    sns.heatmap(cross_attn_sample, cmap='Greens', cbar=True, aspect='auto')
    plt.title('Decoder Cross-Attention\n(Decoder → Encoder)', fontweight='bold')
    plt.xlabel('Encoder Position')
    plt.ylabel('Decoder Position')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis
    print(f"   Self-attention causal compliance:")
    # Check if self-attention respects causal mask
    causal_violations = 0
    total_positions = 0
    
    for layer_idx in range(num_layers):
        layer_attn = self_attn_weights[layer_idx][0]  # First batch
        for head in range(num_heads):
            for i in range(decoder_seq_length):
                for j in range(decoder_seq_length):
                    if j > i:  # Future position
                        if layer_attn[head, i, j] > 1e-6:  # Should be ~0
                            causal_violations += 1
                        total_positions += 1
    
    print(f"     Causal violations: {causal_violations}/{total_positions}")
    print(f"     Causal compliance: {100 * (1 - causal_violations/total_positions):.2f}%")
    
    # Cross-attention analysis
    print(f"\n   Cross-attention patterns:")
    for layer_idx in range(min(3, num_layers)):  # Show first 3 layers
        layer_cross_attn = cross_attn_weights[layer_idx][0, 0]  # First batch, first head
        
        # Find most attended encoder positions for each decoder position
        max_attn_positions = torch.argmax(layer_cross_attn, dim=1)
        avg_attention_pos = torch.mean(max_attn_positions.float())
        
        print(f"     Layer {layer_idx + 1}: Avg max attention at encoder pos {avg_attention_pos:.1f}")
    
    # Analyze information flow
    decoder_layer_norms = []
    current_x = decoder_input
    
    with torch.no_grad():
        for layer in decoder.layers:
            current_x, _, _ = layer(current_x, encoder_output, causal_mask)
            layer_norm = torch.norm(current_x[0], dim=-1).mean().item()
            decoder_layer_norms.append(layer_norm)
    
    print(f"\n   Representation evolution:")
    for i, norm in enumerate(decoder_layer_norms):
        print(f"     Layer {i + 1}: Avg norm = {norm:.3f}")
    
    return {
        'decoder': decoder,
        'decoder_output': decoder_output,
        'self_attention_weights': self_attn_weights,
        'cross_attention_weights': cross_attn_weights,
        'causal_mask': causal_mask,
        'layer_norms': decoder_layer_norms
    }


# ==========================================
# SECTION 4: COMPLETE TRANSFORMER MODEL
# ==========================================

class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_length=5000,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Transformer components
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src_encoded = self.positional_encoding(src_embedded)
        tgt_encoded = self.positional_encoding(tgt_embedded)
        
        # Encoder
        encoder_output, encoder_attn_weights = self.encoder(src_encoded, src_mask)
        
        # Decoder
        decoder_output, decoder_self_attn, decoder_cross_attn = self.decoder(
            tgt_encoded, encoder_output, tgt_mask, src_mask
        )
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output, {
            'encoder_attention': encoder_attn_weights,
            'decoder_self_attention': decoder_self_attn,
            'decoder_cross_attention': decoder_cross_attn
        }
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_encoded = self.positional_encoding(src_embedded)
        encoder_output, attention_weights = self.encoder(src_encoded, src_mask)
        return encoder_output, attention_weights
    
    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """Decode target sequence given encoder output"""
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_encoded = self.positional_encoding(tgt_embedded)
        decoder_output, self_attn, cross_attn = self.decoder(
            tgt_encoded, encoder_output, tgt_mask, src_mask
        )
        output = self.output_projection(decoder_output)
        return output, self_attn, cross_attn


def create_padding_mask(seq, pad_token=0):
    """Create mask for padding tokens"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def demonstrate_complete_transformer():
    """
    Demonstrate the complete transformer model
    """
    print("\n🏗️ Complete Transformer Model Demonstration")
    print("=" * 50)
    
    # Configuration
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_length = 5000
    dropout = 0.1
    
    batch_size = 2
    src_seq_length = 20
    tgt_seq_length = 15
    
    print(f"📊 Complete Transformer Configuration:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Model dimension: {d_model}")
    print(f"   Encoder layers: {num_encoder_layers}")
    print(f"   Decoder layers: {num_decoder_layers}")
    print(f"   Attention heads: {num_heads}")
    print(f"   Feed-forward dimension: {d_ff}")
    print()
    
    # Create transformer model
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout
    )
    
    # Create test sequences
    torch.manual_seed(42)
    
    # Source sequence (e.g., source language in translation)
    src_seq = torch.randint(1, vocab_size, (batch_size, src_seq_length))
    src_seq[:, -2:] = 0  # Add some padding tokens
    
    # Target sequence (e.g., target language in translation)
    tgt_seq = torch.randint(1, vocab_size, (batch_size, tgt_seq_length))
    tgt_seq[:, -1:] = 0  # Add some padding tokens
    
    # Create masks
    src_mask = create_padding_mask(src_seq)
    tgt_mask = create_causal_mask(tgt_seq_length, tgt_seq.device) & create_padding_mask(tgt_seq)
    
    print(f"🎭 Mask Information:")
    print(f"   Source mask shape: {src_mask.shape}")
    print(f"   Target mask shape: {tgt_mask.shape}")
    print(f"   Target mask combines causal + padding masks")
    
    # Forward pass
    with torch.no_grad():
        output, attention_weights = transformer(src_seq, tgt_seq, src_mask, tgt_mask)
    
    print(f"\n✅ Complete Transformer Forward Pass:")
    print(f"   Source input shape: {src_seq.shape}")
    print(f"   Target input shape: {tgt_seq.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output vocabulary logits for each position")
    
    # Analyze model size and complexity
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size (approx): {total_params * 4 / 1e6:.1f} MB")
    
    # Break down parameters by component
    component_params = {}
    component_params['Embeddings'] = (
        transformer.src_embedding.weight.numel() + 
        transformer.tgt_embedding.weight.numel()
    )
    component_params['Encoder'] = sum(p.numel() for p in transformer.encoder.parameters())
    component_params['Decoder'] = sum(p.numel() for p in transformer.decoder.parameters())
    component_params['Output Projection'] = transformer.output_projection.weight.numel()
    
    print(f"\n   Parameter breakdown:")
    for component, count in component_params.items():
        percentage = 100 * count / total_params
        print(f"     {component}: {count:,} ({percentage:.1f}%)")
    
    # Visualize attention patterns
    plt.figure(figsize=(20, 12))
    
    # Encoder attention (last layer, first head)
    plt.subplot(2, 4, 1)
    encoder_attn = attention_weights['encoder_attention'][-1][0, 0].numpy()
    sns.heatmap(encoder_attn, cmap='Blues', cbar=True, square=True)
    plt.title('Encoder Self-Attention\n(Last Layer)', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Decoder self-attention (last layer, first head)
    plt.subplot(2, 4, 2)
    decoder_self_attn = attention_weights['decoder_self_attention'][-1][0, 0].numpy()
    sns.heatmap(decoder_self_attn, cmap='Reds', cbar=True, square=True)
    plt.title('Decoder Self-Attention\n(Last Layer)', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Decoder cross-attention (last layer, first head)
    plt.subplot(2, 4, 3)
    decoder_cross_attn = attention_weights['decoder_cross_attention'][-1][0, 0].numpy()
    sns.heatmap(decoder_cross_attn, cmap='Greens', cbar=True, aspect='auto')
    plt.title('Decoder Cross-Attention\n(Last Layer)', fontweight='bold')
    plt.xlabel('Encoder Position')
    plt.ylabel('Decoder Position')
    
    # Parameter distribution pie chart
    plt.subplot(2, 4, 4)
    labels = list(component_params.keys())
    sizes = list(component_params.values())
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Parameter Distribution', fontweight='bold')
    
    # Memory and computation analysis
    plt.subplot(2, 4, 5)
    seq_lengths = np.array([16, 32, 64, 128, 256, 512])
    
    # Memory usage (primarily attention matrices)
    encoder_memory = batch_size * num_heads * seq_lengths**2 * 4  # bytes
    decoder_self_memory = batch_size * num_heads * seq_lengths**2 * 4
    decoder_cross_memory = batch_size * num_heads * seq_lengths * src_seq_length * 4
    
    total_memory = (encoder_memory + decoder_self_memory + decoder_cross_memory) / 1e6  # MB
    
    plt.loglog(seq_lengths, total_memory, 'b-o', linewidth=2, markersize=8, label='Total Memory')
    plt.loglog(seq_lengths, encoder_memory / 1e6, 'r--', alpha=0.7, label='Encoder')
    plt.loglog(seq_lengths, decoder_self_memory / 1e6, 'g--', alpha=0.7, label='Decoder Self')
    plt.title('Memory Usage vs Sequence Length', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB, log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Attention head analysis across all layers
    plt.subplot(2, 4, 6)
    
    # Compute attention entropy for each head in each layer
    all_encoder_entropies = []
    for layer_attn in attention_weights['encoder_attention']:
        layer_entropies = []
        for head in range(num_heads):
            head_entropies = []
            for pos in range(src_seq_length):
                attn_dist = layer_attn[0, head, pos]  # First batch item
                entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
                head_entropies.append(entropy.item())
            layer_entropies.append(np.mean(head_entropies))
        all_encoder_entropies.append(layer_entropies)
    
    encoder_entropy_matrix = np.array(all_encoder_entropies)
    sns.heatmap(encoder_entropy_matrix, cmap='viridis', cbar=True)
    plt.title('Encoder Attention Entropy\n(Layer × Head)', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Encoder Layer')
    
    # Model scaling analysis
    plt.subplot(2, 4, 7)
    model_sizes = [d_model * i for i in [0.5, 1, 1.5, 2, 2.5, 3]]
    param_counts = []
    
    for size in model_sizes:
        # Approximate parameter count scaling
        embedding_params = 2 * vocab_size * size  # src + tgt embeddings
        encoder_params = num_encoder_layers * (4 * size**2 + 2 * size * d_ff + 6 * size)
        decoder_params = num_decoder_layers * (6 * size**2 + 2 * size * d_ff + 8 * size)
        output_params = vocab_size * size
        
        total = embedding_params + encoder_params + decoder_params + output_params
        param_counts.append(total / 1e6)  # Convert to millions
    
    plt.plot([s/d_model for s in model_sizes], param_counts, 'ro-', linewidth=2, markersize=8)
    plt.title('Parameter Scaling with Model Size', fontweight='bold')
    plt.xlabel('Model Size Multiplier')
    plt.ylabel('Parameters (Millions)')
    plt.grid(True, alpha=0.3)
    
    # Training complexity visualization
    plt.subplot(2, 4, 8)
    
    # Compare with other architectures (approximate)
    architectures = ['RNN', 'LSTM', 'Transformer', 'Large Transformer']
    params = [1, 4, 65, 175]  # Millions of parameters (approximate)
    performance = [0.6, 0.75, 0.85, 0.92]  # Approximate performance scores
    colors = ['red', 'orange', 'blue', 'green']
    
    plt.scatter(params, performance, s=[50, 100, 200, 400], c=colors, alpha=0.7)
    
    for i, arch in enumerate(architectures):
        plt.annotate(arch, (params[i], performance[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.title('Model Size vs Performance', fontweight='bold')
    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Performance Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    print(f"\n🔍 Transformer Analysis Summary:")
    print(f"   Architecture: Encoder-Decoder with attention")
    print(f"   Key innovation: Self-attention replaces recurrence")
    print(f"   Parallelization: All positions processed simultaneously")
    print(f"   Memory scaling: O(n²) with sequence length")
    print(f"   Compute scaling: O(n²d + nd²) per layer")
    
    print(f"\n🎯 Key Advantages over RNNs:")
    advantages = [
        "✅ Parallel processing enables faster training",
        "✅ Direct connections between all positions",
        "✅ No vanishing gradient problem",
        "✅ Better modeling of long-range dependencies",
        "✅ More interpretable attention patterns",
        "✅ Scalable to very large models"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    return {
        'transformer': transformer,
        'output': output,
        'attention_weights': attention_weights,
        'total_parameters': total_params,
        'component_params': component_params,
        'src_seq': src_seq,
        'tgt_seq': tgt_seq,
        'masks': {'src_mask': src_mask, 'tgt_mask': tgt_mask}
    }


# ==========================================
# SECTION 5: TRAINING DYNAMICS AND OPTIMIZATION
# ==========================================

class TransformerTrainingDynamics:
    """
    Analyze transformer training dynamics and optimization
    """
    
    def __init__(self, model):
        self.model = model
        self.training_history = []
        
    def demonstrate_training_setup(self):
        """
        Demonstrate proper transformer training setup
        """
        print("\n🎓 Transformer Training Dynamics")
        print("=" * 50)
        
        print("🔧 Essential Training Components:")
        print("   1. Label Smoothing: Prevents overconfident predictions")
        print("   2. Learning Rate Scheduling: Warmup + decay")
        print("   3. Gradient Clipping: Prevents gradient explosion")
        print("   4. Dropout: Regularization during training")
        print("   5. Layer Normalization: Stabilizes training")
        print()
        
        # Demonstrate learning rate scheduling
        self.demonstrate_lr_scheduling()
        
        # Demonstrate gradient flow
        self.analyze_gradient_flow()
        
        # Demonstrate training stability
        self.analyze_training_stability()
    
    def demonstrate_lr_scheduling(self):
        """
        Demonstrate transformer learning rate scheduling
        """
        print("📈 Learning Rate Scheduling:")
        
        # Transformer learning rate schedule: warmup + decay
        def transformer_lr_schedule(step, d_model=512, warmup_steps=4000):
            """Original transformer learning rate schedule"""
            arg1 = step ** (-0.5)
            arg2 = step * (warmup_steps ** (-1.5))
            return (d_model ** (-0.5)) * min(arg1, arg2)
        
        steps = np.arange(1, 50000, 100)
        lrs = [transformer_lr_schedule(step) for step in steps]
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(steps, lrs, 'b-', linewidth=2)
        plt.title('Transformer LR Schedule', fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        # Mark warmup phase
        warmup_steps = 4000
        plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='End of Warmup')
        plt.legend()
        
                # Compare with other schedules
        plt.subplot(2, 3, 2)
        
        # Different scheduling strategies
        constant_lr = [0.001] * len(steps)
        exponential_decay = [0.001 * (0.96 ** (step // 1000)) for step in steps]
        cosine_decay = [0.001 * (1 + np.cos(np.pi * step / 50000)) / 2 for step in steps]
        
        plt.plot(steps, lrs, 'b-', label='Transformer Schedule', linewidth=2)
        plt.plot(steps, constant_lr, 'r--', label='Constant LR', linewidth=2)
        plt.plot(steps, exponential_decay, 'g--', label='Exponential Decay', linewidth=2)
        plt.plot(steps, cosine_decay, 'm--', label='Cosine Decay', linewidth=2)
        
        plt.title('LR Schedule Comparison', fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Warmup effect visualization
        plt.subplot(2, 3, 3)
        warmup_steps_list = [1000, 2000, 4000, 8000]
        
        for warmup in warmup_steps_list:
            warmup_lrs = [transformer_lr_schedule(step, warmup_steps=warmup) for step in steps[:200]]
            plt.plot(steps[:200], warmup_lrs, label=f'Warmup: {warmup}', linewidth=2)
        
        plt.title('Effect of Warmup Duration', fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("   ✅ Warmup prevents early training instability")
        print("   ✅ Gradual decay maintains training stability")
        print("   ✅ Scale-invariant to model size")
    
    def analyze_gradient_flow(self):
        """
        Analyze gradient flow through transformer layers
        """
        print("\n🌊 Gradient Flow Analysis:")
        
        # Create simplified model for analysis
        small_transformer = Transformer(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=512
        )
        
        # Create dummy batch
        batch_size = 4
        src_seq = torch.randint(1, 1000, (batch_size, 10))
        tgt_seq = torch.randint(1, 1000, (batch_size, 8))
        
        # Forward pass
        output, _ = small_transformer(src_seq, tgt_seq)
        
        # Create dummy loss (sum of all outputs)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients by layer
        encoder_grad_norms = []
        decoder_grad_norms = []
        
        # Encoder gradients
        for i, layer in enumerate(small_transformer.encoder.layers):
            layer_grad_norm = 0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            if param_count > 0:
                encoder_grad_norms.append(math.sqrt(layer_grad_norm / param_count))
        
        # Decoder gradients
        for i, layer in enumerate(small_transformer.decoder.layers):
            layer_grad_norm = 0
            param_count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            if param_count > 0:
                decoder_grad_norms.append(math.sqrt(layer_grad_norm / param_count))
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(encoder_grad_norms) + 1), encoder_grad_norms, 
                'bo-', linewidth=2, markersize=8, label='Encoder')
        plt.plot(range(1, len(decoder_grad_norms) + 1), decoder_grad_norms, 
                'ro-', linewidth=2, markersize=8, label='Decoder')
        plt.title('Gradient Norms by Layer', fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Average Gradient Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Attention vs FFN gradient comparison
        plt.subplot(1, 3, 2)
        attn_grads = []
        ffn_grads = []
        
        for layer in small_transformer.encoder.layers:
            # Attention gradients
            attn_grad = 0
            attn_params = 0
            for name, param in layer.self_attention.named_parameters():
                if param.grad is not None:
                    attn_grad += param.grad.norm().item() ** 2
                    attn_params += 1
            
            # FFN gradients
            ffn_grad = 0
            ffn_params = 0
            for name, param in layer.feed_forward.named_parameters():
                if param.grad is not None:
                    ffn_grad += param.grad.norm().item() ** 2
                    ffn_params += 1
            
            if attn_params > 0:
                attn_grads.append(math.sqrt(attn_grad / attn_params))
            if ffn_params > 0:
                ffn_grads.append(math.sqrt(ffn_grad / ffn_params))
        
        layers = range(1, len(attn_grads) + 1)
        width = 0.35
        
        plt.bar([l - width/2 for l in layers], attn_grads, width, 
               label='Attention', alpha=0.8, color='skyblue')
        plt.bar([l + width/2 for l in layers], ffn_grads, width, 
               label='Feed-Forward', alpha=0.8, color='lightcoral')
        
        plt.title('Attention vs FFN Gradients', fontweight='bold')
        plt.xlabel('Layer')
        plt.ylabel('Average Gradient Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gradient distribution
        plt.subplot(1, 3, 3)
        all_grads = []
        for param in small_transformer.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().tolist())
        
        plt.hist(all_grads, bins=50, alpha=0.7, density=True, color='green')
        plt.title('Gradient Distribution', fontweight='bold')
        plt.xlabel('Gradient Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        grad_mean = np.mean(all_grads)
        grad_std = np.std(all_grads)
        plt.axvline(grad_mean, color='r', linestyle='--', alpha=0.8, label=f'Mean: {grad_mean:.4f}')
        plt.axvline(grad_mean + grad_std, color='r', linestyle=':', alpha=0.8, label=f'±1σ: {grad_std:.4f}')
        plt.axvline(grad_mean - grad_std, color='r', linestyle=':', alpha=0.8)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"   Average encoder gradient norm: {np.mean(encoder_grad_norms):.4f}")
        print(f"   Average decoder gradient norm: {np.mean(decoder_grad_norms):.4f}")
        print(f"   Gradient standard deviation: {grad_std:.4f}")
        print("   ✅ Pre-norm architecture helps gradient flow")
        print("   ✅ Residual connections prevent vanishing gradients")
    
    def analyze_training_stability(self):
        """
        Analyze factors affecting transformer training stability
        """
        print("\n⚖️ Training Stability Analysis:")
        
        stability_factors = {
            'Layer Normalization': {
                'effect': 'Stabilizes activations, enables deeper networks',
                'placement': 'Pre-norm (before sub-layers) vs Post-norm (after)',
                'benefit': 'Reduces internal covariate shift'
            },
            'Residual Connections': {
                'effect': 'Enables gradient flow through deep networks',
                'placement': 'Around each sub-layer (attention, FFN)',
                'benefit': 'Prevents vanishing gradients'
            },
            'Dropout': {
                'effect': 'Regularization, prevents overfitting',
                'placement': 'After attention weights, in FFN, after embeddings',
                'benefit': 'Improves generalization'
            },
            'Label Smoothing': {
                'effect': 'Prevents overconfident predictions',
                'placement': 'In loss function',
                'benefit': 'Better calibrated probabilities'
            },
            'Gradient Clipping': {
                'effect': 'Prevents gradient explosion',
                'placement': 'Before optimizer step',
                'benefit': 'Training stability'
            }
        }
        
        print("   Key stability mechanisms:")
        for mechanism, details in stability_factors.items():
            print(f"   🔧 {mechanism}:")
            print(f"      Effect: {details['effect']}")
            print(f"      Placement: {details['placement']}")
            print(f"      Benefit: {details['benefit']}")
            print()
        
        # Demonstrate label smoothing effect
        print("📊 Label Smoothing Demonstration:")
        
        # Simulate predictions
        vocab_size = 1000
        true_class = 42
        
        # Without label smoothing (one-hot)
        hard_target = torch.zeros(vocab_size)
        hard_target[true_class] = 1.0
        
        # With label smoothing
        smoothing = 0.1
        smooth_target = torch.full((vocab_size,), smoothing / (vocab_size - 1))
        smooth_target[true_class] = 1.0 - smoothing
        
        # Model predictions (over-confident)
        confident_pred = torch.zeros(vocab_size)
        confident_pred[true_class] = 0.95
        confident_pred[true_class + 1] = 0.03
        confident_pred[true_class + 2] = 0.02
        
        # Compute losses
        def cross_entropy_loss(pred, target):
            return -torch.sum(target * torch.log(pred + 1e-8))
        
        hard_loss = cross_entropy_loss(confident_pred, hard_target)
        smooth_loss = cross_entropy_loss(confident_pred, smooth_target)
        
        print(f"   Hard target loss: {hard_loss:.4f}")
        print(f"   Smooth target loss: {smooth_loss:.4f}")
        print(f"   Label smoothing encourages less confident predictions")
        
        return stability_factors


# ==========================================
# SECTION 6: MODERN TRANSFORMER VARIANTS
# ==========================================

def explore_transformer_variants():
    """
    Explore modern transformer variants and their innovations
    """
    print("\n🚀 Modern Transformer Variants")
    print("=" * 50)
    
    variants = {
        'BERT': {
            'year': 2018,
            'architecture': 'Encoder-only',
            'key_innovation': 'Bidirectional training with masked language modeling',
            'use_case': 'Language understanding, classification, NER',
            'training': 'Pre-training on masked LM + NSP, fine-tuning on downstream tasks',
            'impact': 'Revolutionized NLP benchmarks, established pre-training paradigm'
        },
        'GPT': {
            'year': 2018,
            'architecture': 'Decoder-only',
            'key_innovation': 'Autoregressive language modeling at scale',
            'use_case': 'Text generation, completion, few-shot learning',
            'training': 'Unsupervised pre-training on next token prediction',
            'impact': 'Showed emergence of capabilities with scale'
        },
        'T5': {
            'year': 2019,
            'architecture': 'Encoder-decoder',
            'key_innovation': 'Text-to-text unified framework',
            'use_case': 'All NLP tasks as text generation',
            'training': 'Span corruption + supervised multi-task learning',
            'impact': 'Unified approach to diverse NLP tasks'
        },
        'Vision Transformer (ViT)': {
            'year': 2020,
            'architecture': 'Encoder-only',
            'key_innovation': 'Transformers for computer vision',
            'use_case': 'Image classification, object detection',
            'training': 'Images as sequences of patches',
            'impact': 'Brought transformers to computer vision'
        },
        'DALL-E': {
            'year': 2021,
            'architecture': 'Decoder-only',
            'key_innovation': 'Text-to-image generation',
            'use_case': 'Multimodal generation',
            'training': 'Autoregressive modeling over text+image tokens',
            'impact': 'Demonstrated multimodal capabilities'
        },
        'ChatGPT/GPT-4': {
            'year': 2022,
            'architecture': 'Decoder-only',
            'key_innovation': 'Reinforcement Learning from Human Feedback (RLHF)',
            'use_case': 'Conversational AI, instruction following',
            'training': 'Pre-training + supervised fine-tuning + RLHF',
            'impact': 'Brought AI to mainstream, conversational abilities'
        }
    }
    
    print("🏛️ Transformer Evolution Timeline:")
    for variant, details in variants.items():
        print(f"\n📅 {details['year']}: {variant}")
        print(f"   Architecture: {details['architecture']}")
        print(f"   Innovation: {details['key_innovation']}")
        print(f"   Use Case: {details['use_case']}")
        print(f"   Training: {details['training']}")
        print(f"   Impact: {details['impact']}")
    
    # Visualize the evolution
    plt.figure(figsize=(20, 12))
    
    # Timeline visualization
    plt.subplot(2, 3, 1)
    years = [details['year'] for details in variants.values()]
    names = list(variants.keys())
    
    plt.scatter(years, range(len(years)), s=200, alpha=0.7, c='skyblue')
    
    for i, (year, name) in enumerate(zip(years, names)):
        plt.annotate(name, (year, i), xytext=(10, 0), textcoords='offset points',
                    fontsize=10, ha='left')
    
    plt.title('Transformer Variants Timeline', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Variant')
    plt.yticks(range(len(names)), [''] * len(names))  # Hide y-tick labels
    plt.grid(True, alpha=0.3)
    
    # Architecture types
    plt.subplot(2, 3, 2)
    arch_types = {}
    for variant, details in variants.items():
        arch = details['architecture']
        if arch not in arch_types:
            arch_types[arch] = []
        arch_types[arch].append(variant)
    
    labels = list(arch_types.keys())
    sizes = [len(variants) for variants in arch_types.values()]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Architecture Distribution', fontweight='bold')
    
    # Use case categories
    plt.subplot(2, 3, 3)
    use_cases = {}
    for variant, details in variants.items():
        case = details['use_case'].split(',')[0].strip()  # Take first use case
        if 'generation' in case.lower():
            category = 'Generation'
        elif 'understanding' in case.lower() or 'classification' in case.lower():
            category = 'Understanding'
        elif 'vision' in case.lower() or 'image' in case.lower():
            category = 'Vision'
        else:
            category = 'Other'
        
        if category not in use_cases:
            use_cases[category] = 0
        use_cases[category] += 1
    
    categories = list(use_cases.keys())
    counts = list(use_cases.values())
    
    plt.bar(categories, counts, alpha=0.7, color=['gold', 'lightcoral', 'lightgreen', 'lightblue'])
    plt.title('Use Case Categories', fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Number of Variants')
    plt.grid(True, alpha=0.3)
    
    # Model size evolution (approximate)
    plt.subplot(2, 3, 4)
    model_sizes = {
        'BERT': 110,  # BERT-Base
        'GPT': 117,   # GPT-1
        'T5': 220,    # T5-Base
        'Vision Transformer (ViT)': 86,  # ViT-Base
        'DALL-E': 12000,  # DALL-E 1
        'ChatGPT/GPT-4': 175000  # GPT-3 size (GPT-4 size unknown)
    }
    
    model_names = list(model_sizes.keys())
    sizes = list(model_sizes.values())
    
    plt.bar(range(len(model_names)), sizes, alpha=0.7, color='lightcoral')
    plt.title('Model Size Evolution (Millions of Parameters)', fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Parameters (Millions)')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Scaling laws demonstration
    plt.subplot(2, 3, 5)
    
    # Approximate scaling law: Performance ∝ (Parameters)^α
    alpha = 0.2  # Empirical scaling exponent
    params = np.logspace(6, 11, 50)  # 1M to 100B parameters
    performance = 50 + 40 * (params / 1e9) ** alpha  # Arbitrary performance metric
    
    plt.semilogx(params, performance, 'b-', linewidth=3, label='Scaling Law')
    
    # Mark actual models
    for name, size in model_sizes.items():
        if size < 1000:  # Convert to consistent units
            size_scaled = size * 1e6
        else:
            size_scaled = size * 1e6
        
        perf = 50 + 40 * (size_scaled / 1e9) ** alpha
        plt.semilogx(size_scaled, perf, 'ro', markersize=8)
        
        if name in ['GPT', 'ChatGPT/GPT-4']:  # Annotate key models
            plt.annotate(name, (size_scaled, perf), xytext=(10, 10), 
                        textcoords='offset points', fontsize=9)
    
    plt.title('Transformer Scaling Laws', fontweight='bold')
    plt.xlabel('Parameters')
    plt.ylabel('Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Future directions
    plt.subplot(2, 3, 6)
    
    future_challenges = [
        'Efficiency',
        'Long Context',
        'Multimodal',
        'Reasoning',
        'Alignment'
    ]
    
    importance_scores = [9, 8, 9, 10, 10]  # Subjective importance ratings
    colors = ['lightblue', 'lightgreen', 'gold', 'lightcoral', 'plum']
    
    plt.barh(future_challenges, importance_scores, color=colors, alpha=0.7)
    plt.title('Future Research Directions', fontweight='bold')
    plt.xlabel('Importance/Urgency (1-10)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔮 Future Directions:")
    future_directions = {
        'Efficiency': 'Sparse attention, efficient architectures, model compression',
        'Long Context': 'Linear attention, hierarchical processing, memory mechanisms',
        'Multimodal': 'Vision-language, audio-text, unified multimodal architectures',
        'Reasoning': 'Chain-of-thought, tool use, symbolic reasoning integration',
        'Alignment': 'RLHF improvements, constitutional AI, interpretability'
    }
    
    for direction, description in future_directions.items():
        print(f"   🎯 {direction}: {description}")
    
    return variants


# ==========================================
# SECTION 7: COMPREHENSIVE INTEGRATION
# ==========================================

def transformer_mastery_assessment():
    """
    Comprehensive assessment of transformer architecture mastery
    """
    print("\n🎓 Week 26 Mastery Assessment: Complete Transformer Architecture")
    print("=" * 70)
    
    print("🧠 Architecture Understanding Assessment:")
    
    assessment_results = {}
    
    # 1. Building Blocks Mastery
    print("\n1. Building Blocks Implementation")
    components_demo = demonstrate_transformer_components()
    assessment_results['building_blocks'] = True
    
    # 2. Encoder Architecture
    print("\n2. Encoder Architecture")
    encoder_demo = demonstrate_transformer_encoder()
    assessment_results['encoder_mastery'] = True
    
    # 3. Decoder Architecture  
    print("\n3. Decoder Architecture")
    decoder_demo = demonstrate_transformer_decoder()
    assessment_results['decoder_mastery'] = True
    
    # 4. Complete Model
    print("\n4. Complete Transformer Model")
    complete_demo = demonstrate_complete_transformer()
    assessment_results['complete_model'] = True
    
    # 5. Training Dynamics
    print("\n5. Training Dynamics")
    training_dynamics = TransformerTrainingDynamics(complete_demo['transformer'])
    training_dynamics.demonstrate_training_setup()
    assessment_results['training_understanding'] = True
    
    # 6. Modern Variants
    print("\n6. Modern Transformer Variants")
    variants = explore_transformer_variants()
    assessment_results['modern_variants'] = True
    
    # Calculate overall mastery
    overall_score = sum(assessment_results.values()) / len(assessment_results)
    
    # Comprehensive visualization
    plt.figure(figsize=(20, 15))
    
    # Mastery radar chart
    categories = [
        'Building Blocks',
        'Encoder Architecture', 
        'Decoder Architecture',
        'Complete Model',
        'Training Dynamics',
        'Modern Variants'
    ]
    
    scores = [1.0 if assessment_results[key] else 0.5 for key in assessment_results.keys()]
    
    # Close the radar chart
    categories_plot = categories + [categories[0]]
    scores_plot = scores + [scores[0]]
    
    angles = np.linspace(0, 2*np.pi, len(categories_plot), endpoint=True)
    
    plt.subplot(2, 4, 1)
    plt.polar(angles, scores_plot, 'o-', linewidth=3, markersize=8, color='blue')
    plt.fill(angles, scores_plot, alpha=0.25, color='blue')
    plt.thetagrids(angles[:-1] * 180/np.pi, categories, fontsize=10)
    plt.ylim(0, 1)
    plt.title('Transformer Mastery Assessment', fontweight='bold', pad=20)
    
    # Architecture complexity evolution
    plt.subplot(2, 4, 2)
    models = ['RNN', 'LSTM', 'Attention', 'Transformer', 'BERT', 'GPT-3']
    complexity = [1, 2, 3, 5, 7, 10]
    performance = [3, 5, 6, 8, 9, 9.5]
    
    plt.scatter(complexity, performance, s=[50*c for c in complexity], alpha=0.7, c='red')
    
    for i, model in enumerate(models):
        plt.annotate(model, (complexity[i], performance[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Architecture Evolution', fontweight='bold')
    plt.xlabel('Model Complexity')
    plt.ylabel('Performance')
    plt.grid(True, alpha=0.3)
    
    # Parameter distribution in modern transformers
    plt.subplot(2, 4, 3)
    components = ['Embeddings', 'Attention', 'Feed-Forward', 'Output Layer']
    percentages = [20, 25, 50, 5]  # Typical distribution
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
    
    plt.pie(percentages, labels=components, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Parameter Distribution', fontweight='bold')
    
    # Scaling trends
    plt.subplot(2, 4, 4)
    years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023])
    params = np.array([65, 340, 1500, 11000, 175000, 540000, 1000000])  # Millions
    
    plt.semilogy(years, params, 'go-', linewidth=3, markersize=8)
    plt.title('Model Size Growth', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Parameters (Millions, log scale)')
    plt.grid(True, alpha=0.3)
    
    # Attention pattern examples
    plt.subplot(2, 4, 5)
    # Create example attention pattern
    seq_len = 12
    attention_pattern = np.random.rand(seq_len, seq_len)
    
    # Make it more realistic (diagonal + some global attention)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= 2:  # Local attention
                attention_pattern[i, j] *= 2
            if j == 0 or j == seq_len - 1:  # Boundary attention
                attention_pattern[i, j] *= 1.5
    
    # Normalize
    attention_pattern = attention_pattern / attention_pattern.sum(axis=1, keepdims=True)
    
    sns.heatmap(attention_pattern, cmap='Blues', cbar=True, square=True)
    plt.title('Typical Attention Pattern', fontweight='bold')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Training efficiency comparison
    plt.subplot(2, 4, 6)
    architectures = ['RNN', 'LSTM', 'Transformer']
    training_time = [100, 80, 20]  # Relative training time
    parallelization = [1, 1, 10]  # Parallelization factor
    
    x = np.arange(len(architectures))
    width = 0.35
    
    plt.bar(x - width/2, training_time, width, label='Training Time', alpha=0.8, color='red')
    plt.bar(x + width/2, [p*5 for p in parallelization], width, label='Parallelization×5', alpha=0.8, color='green')
    
    plt.title('Training Efficiency', fontweight='bold')
    plt.xlabel('Architecture')
    plt.ylabel('Relative Score')
    plt.xticks(x, architectures)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Application domains
    plt.subplot(2, 4, 7)
    domains = ['NLP', 'Vision', 'Speech', 'Multimodal', 'Code', 'Science']
    adoption = [10, 8, 6, 9, 7, 5]  # Transformer adoption score
    
    plt.bar(domains, adoption, alpha=0.7, color='purple')
    plt.title('Transformer Adoption by Domain', fontweight='bold')
    plt.xlabel('Domain')
    plt.ylabel('Adoption Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Memory usage scaling
    plt.subplot(2, 4, 8)
    seq_lengths = [128, 256, 512, 1024, 2048]
    memory_usage = [s**2 * 0.001 for s in seq_lengths]  # O(n²) scaling
    
    plt.loglog(seq_lengths, memory_usage, 'b-o', linewidth=3, markersize=8)
    plt.title('Memory Scaling (O(n²))', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (relative)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final assessment summary
    print(f"\n🏆 Overall Mastery Assessment:")
    print(f"   Score: {overall_score:.1%}")
    
    if overall_score >= 0.9:
        mastery_level = "Expert"
        next_steps = "Ready for advanced applications and research"
    elif overall_score >= 0.7:
        mastery_level = "Advanced"
        next_steps = "Continue with NLP and vision applications"
    else:
        mastery_level = "Intermediate"
        next_steps = "Review implementation details and practice"
    
    print(f"   Mastery Level: {mastery_level}")
    print(f"   Next Steps: {next_steps}")
    
    # Detailed breakdown
    print(f"\n📊 Detailed Assessment Breakdown:")
    assessment_details = {
        'building_blocks': 'Building Blocks Mastery',
        'encoder_mastery': 'Encoder Architecture',
        'decoder_mastery': 'Decoder Architecture', 
        'complete_model': 'Complete Model Integration',
        'training_understanding': 'Training Dynamics',
        'modern_variants': 'Modern Variants Knowledge' 
        }
    
    for key, description in assessment_details.items():
        status = "✅ MASTERED" if assessment_results[key] else "❌ NEEDS WORK"
        print(f"   {description}: {status}")
    
    # Week 27 preparation
    print(f"\n🚀 Week 27 Preparation Checklist:")
    week27_prep = [
        "Understand complete transformer architecture ✅",
        "Can implement encoder and decoder from scratch ✅", 
        "Familiar with attention mechanisms deeply ✅",
        "Know training dynamics and optimization ✅",
        "Understand modern variants (BERT, GPT, T5) ✅",
        "Ready for NLP applications and fine-tuning 🎯"
    ]
    
    for item in week27_prep:
        print(f"   {item}")
    
    return {
        'assessment_results': assessment_results,
        'overall_score': overall_score,
        'mastery_level': mastery_level,
        'components_demo': components_demo,
        'encoder_demo': encoder_demo,
        'decoder_demo': decoder_demo,
        'complete_demo': complete_demo,
        'variants': variants
    }


def transformer_implementation_showcase():
    """
    Final showcase demonstrating complete transformer mastery
    """
    print("\n🎭 TRANSFORMER ARCHITECTURE MASTERY SHOWCASE")
    print("=" * 70)
    
    print("🌟 Journey Completed: From Attention to Complete Transformers")
    print()
    
    # Showcase 1: End-to-end translation example
    print("🔥 Showcase 1: End-to-End Translation Pipeline")
    print("-" * 50)
    
    # Create a complete translation-ready transformer
    translation_transformer = Transformer(
        vocab_size=32000,  # Typical BPE vocabulary
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_length=512,
        dropout=0.1
    )
    
    # Simulate a translation task
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 18
    
    # Create realistic sequences (simulating tokenized sentences)
    torch.manual_seed(42)
    src_sequences = torch.randint(1, 32000, (batch_size, src_seq_len))
    tgt_sequences = torch.randint(1, 32000, (batch_size, tgt_seq_len))
    
    # Add special tokens
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    PAD_TOKEN = 0
    
    # Add BOS to target sequences
    tgt_input = torch.cat([
        torch.full((batch_size, 1), BOS_TOKEN),
        tgt_sequences[:, :-1]
    ], dim=1)
    
    # Create masks
    src_mask = create_padding_mask(src_sequences, PAD_TOKEN)
    tgt_mask = create_causal_mask(tgt_seq_len, src_sequences.device) & create_padding_mask(tgt_input, PAD_TOKEN)
    
    # Forward pass
    with torch.no_grad():
        translation_output, attention_weights = translation_transformer(
            src_sequences, tgt_input, src_mask, tgt_mask
        )
    
    print(f"✅ Translation Pipeline Complete:")
    print(f"   Source shape: {src_sequences.shape}")
    print(f"   Target input shape: {tgt_input.shape}")
    print(f"   Translation output shape: {translation_output.shape}")
    print(f"   Ready for beam search decoding!")
    
    # Showcase 2: Attention visualization
    print(f"\n🔥 Showcase 2: Multi-Level Attention Analysis")
    print("-" * 50)
    
    # Analyze attention patterns at different levels
    encoder_attn = attention_weights['encoder_attention']
    decoder_self_attn = attention_weights['decoder_self_attention'] 
    decoder_cross_attn = attention_weights['decoder_cross_attention']
    
    print(f"   Encoder layers: {len(encoder_attn)}")
    print(f"   Decoder layers: {len(decoder_self_attn)}")
    print(f"   Cross-attention patterns: {len(decoder_cross_attn)}")
    
    # Attention head analysis
    num_heads = 8
    head_specializations = []
    
    for layer_idx in range(3):  # Analyze first 3 layers
        layer_attn = encoder_attn[layer_idx][0]  # First batch item
        
        for head in range(num_heads):
            head_attn = layer_attn[head]
            
            # Analyze attention pattern characteristics
            diagonal_strength = torch.diag(head_attn).mean().item()
            entropy = -(head_attn * torch.log(head_attn + 1e-8)).sum(dim=1).mean().item()
            max_attention = head_attn.max().item()
            
            pattern_type = "focused" if entropy < 2.0 else "diffuse"
            specialization = f"Layer {layer_idx+1}, Head {head+1}: {pattern_type} (entropy={entropy:.2f})"
            head_specializations.append(specialization)
    
    print(f"   Head specialization examples:")
    for spec in head_specializations[:6]:  # Show first 6
        print(f"     {spec}")
    
    # Showcase 3: Scaling analysis
    print(f"\n🔥 Showcase 3: Transformer Scaling Analysis")
    print("-" * 50)
    
    # Analyze how transformers scale
    model_configs = [
        {'name': 'Small', 'd_model': 256, 'layers': 4, 'heads': 4},
        {'name': 'Base', 'd_model': 512, 'layers': 6, 'heads': 8},
        {'name': 'Large', 'd_model': 768, 'layers': 12, 'heads': 12},
        {'name': 'XL', 'd_model': 1024, 'layers': 24, 'heads': 16}
    ]
    
    for config in model_configs:
        # Calculate approximate parameters
        d_model = config['d_model']
        layers = config['layers']
        heads = config['heads']
        d_ff = 4 * d_model  # Standard ratio
        vocab_size = 32000
        
        # Parameter calculation
        embedding_params = 2 * vocab_size * d_model  # src + tgt embeddings
        encoder_params = layers * (4 * d_model**2 + 2 * d_model * d_ff + 6 * d_model)
        decoder_params = layers * (6 * d_model**2 + 2 * d_model * d_ff + 8 * d_model)
        output_params = vocab_size * d_model
        
        total_params = embedding_params + encoder_params + decoder_params + output_params
        
        print(f"   {config['name']}: {total_params/1e6:.1f}M parameters")
        print(f"     d_model={d_model}, layers={layers}, heads={heads}")
    
    # Showcase 4: Modern applications
    print(f"\n🔥 Showcase 4: Real-World Applications")
    print("-" * 50)
    
    applications = {
        'Machine Translation': {
            'architecture': 'Encoder-Decoder',
            'example': 'Google Translate, DeepL',
            'key_feature': 'Cross-attention between languages'
        },
        'Text Generation': {
            'architecture': 'Decoder-only', 
            'example': 'GPT-3, ChatGPT',
            'key_feature': 'Autoregressive generation with causal attention'
        },
        'Language Understanding': {
            'architecture': 'Encoder-only',
            'example': 'BERT, RoBERTa',
            'key_feature': 'Bidirectional context encoding'
        },
        'Code Generation': {
            'architecture': 'Decoder-only',
            'example': 'GitHub Copilot, CodeT5',
            'key_feature': 'Programming language patterns'
        },
        'Multimodal AI': {
            'architecture': 'Encoder-Decoder + Vision',
            'example': 'DALL-E, GPT-4V',
            'key_feature': 'Cross-modal attention'
        }
    }
    
    for app, details in applications.items():
        print(f"   📱 {app}:")
        print(f"      Architecture: {details['architecture']}")
        print(f"      Example: {details['example']}")
        print(f"      Key Feature: {details['key_feature']}")
    
    # Final mastery demonstration
    print(f"\n🎯 Complete Mastery Demonstrations:")
    demonstrations = [
        "✅ Architectural understanding: Encoder-decoder, attention, normalization",
        "✅ Implementation mastery: Built complete transformer from scratch",
        "✅ Training dynamics: Learning rates, gradients, stability",
        "✅ Modern variants: BERT, GPT, T5, Vision Transformers",
        "✅ Scaling insights: Parameter counts, memory usage, computational complexity",
        "✅ Real-world applications: Translation, generation, understanding",
        "✅ Optimization techniques: Efficient attention, training strategies",
        "✅ Research readiness: Can implement papers and novel architectures"
    ]
    
    for demo in demonstrations:
        print(f"   {demo}")
    
    print(f"\n🚀 Ready for Week 27: Advanced NLP Applications!")
    
    return {
        'translation_transformer': translation_transformer,
        'attention_analysis': {
            'encoder_attention': encoder_attn,
            'decoder_self_attention': decoder_self_attn,
            'decoder_cross_attention': decoder_cross_attn
        },
        'scaling_analysis': model_configs,
        'applications': applications
    }


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

def main():
    """
    Main execution flow for Week 26: Complete Transformer Architecture
    """
    print("🎯 Starting Week 26: Complete Transformer Architecture")
    print("=" * 70)
    
    # Section 1: Building Blocks
    print("\n🧱 SECTION 1: TRANSFORMER BUILDING BLOCKS")
    components_results = demonstrate_transformer_components()
    
    # Section 2: Encoder Architecture
    print("\n🏗️ SECTION 2: TRANSFORMER ENCODER")
    encoder_results = demonstrate_transformer_encoder()
    
    # Section 3: Decoder Architecture
    print("\n🎯 SECTION 3: TRANSFORMER DECODER")
    decoder_results = demonstrate_transformer_decoder()
    
    # Section 4: Complete Model
    print("\n🏛️ SECTION 4: COMPLETE TRANSFORMER MODEL")
    complete_results = demonstrate_complete_transformer()
    
    # Section 5: Training Dynamics
    print("\n🎓 SECTION 5: TRAINING DYNAMICS")
    training_dynamics = TransformerTrainingDynamics(complete_results['transformer'])
    training_results = training_dynamics.demonstrate_training_setup()
    
    # Section 6: Modern Variants
    print("\n🚀 SECTION 6: MODERN TRANSFORMER VARIANTS")
    variants_results = explore_transformer_variants()
    
    # Section 7: Mastery Assessment
    print("\n🎓 SECTION 7: MASTERY ASSESSMENT")
    assessment_results = transformer_mastery_assessment()
    
    # Final Showcase
    print("\n🎭 FINAL SHOWCASE")
    showcase_results = transformer_implementation_showcase()
    
    # Week Summary
    print("\n🎉 WEEK 26 COMPLETE: TRANSFORMER ARCHITECTURE MASTERED!")
    print("=" * 70)
    
    key_achievements = [
        "🏗️ Built complete transformer architecture from scratch",
        "🧠 Mastered encoder-decoder design patterns",
        "⚡ Implemented multi-head attention mechanisms",
        "🎯 Understood training dynamics and optimization",
        "🚀 Explored modern variants (BERT, GPT, T5)",
        "📊 Analyzed scaling laws and computational complexity",
        "🌟 Ready for advanced NLP applications",
        "🔬 Prepared for transformer research and innovation"
    ]
    
    print("🏆 Key Achievements:")
    for achievement in key_achievements:
        print(f"   {achievement}")
    
    print(f"\n🔮 Next Week Preview: Natural Language Processing Applications")
    print("   Applying transformers to real NLP tasks")
    print("   Pre-training, fine-tuning, and task-specific adaptations")
    print("   BERT for understanding, GPT for generation, T5 for seq2seq")
    
    print(f"\n💡 Key Insights Gained:")
    insights = [
        "Transformers enable fully parallel sequence processing",
        "Self-attention captures long-range dependencies efficiently",
        "Encoder-decoder architecture provides flexible input-output mapping",
        "Layer normalization and residual connections enable deep networks",
        "Pre-training + fine-tuning is the dominant paradigm",
        "Scaling transformers consistently improves performance",
        "Modern AI is built on transformer foundations"
    ]
    
    for insight in insights:
        print(f"   • {insight}")
    
    print(f"\n🧭 The Path Forward:")
    print("   Week 25: ✅ Attention Mechanisms (COMPLETED)")
    print("   Week 26: ✅ Complete Transformer Architecture (COMPLETED)")
    print("   Week 27: 🎯 Natural Language Processing Applications")
    print("   Week 28: 👁️ Computer Vision with Transformers")
    print("   Week 29: 🏗️ Advanced Architectures and Scaling")
    
    return {
        'components': components_results,
        'encoder': encoder_results,
        'decoder': decoder_results,
        'complete_model': complete_results,
        'training_dynamics': training_results,
        'variants': variants_results,
        'assessment': assessment_results,
        'showcase': showcase_results
    }


# Execute the main function when script is run
if __name__ == "__main__":
    print("🚀 Neural Odyssey Week 26: Complete Transformer Architecture")
    print("   'Attention Is All You Need' - From Paper to Implementation")
    print("   Building the Foundation of Modern AI")
    print()
    
    # Run the complete transformer architecture journey
    results = main()
    
    print("\n🎊 Congratulations! You have mastered the complete transformer architecture!")
    print("   You now understand the technology that powers:")
    print("   • ChatGPT and all large language models")
    print("   • Google Translate and modern translation systems")
    print("   • BERT and language understanding models")
    print("   • Vision transformers in computer vision")
    print("   • Multimodal AI systems like DALL-E")
    print("   • Code generation tools like GitHub Copilot")
    
    print(f"\n📚 Ready for advanced applications:")
    print("   Your transformer mastery enables you to:")
    print("   • Implement any transformer variant from research papers")
    print("   • Design custom architectures for specific tasks")
    print("   • Optimize transformers for production deployment")
    print("   • Contribute to cutting-edge AI research")
    print("   • Build the next generation of AI systems")
    
    print(f"\n🌟 The transformer revolution continues with your expertise!")
    # Attention complexity: O(n²d)
    attn_complexity = [n**2 * d_model for n in seq_lengths]
    # FFN complexity: O(nd²)
    ffn_complexity = [n * d_model * d_ff for n in seq_lengths]
    
    plt.loglog(seq_lengths, attn_complexity, 'b-', label='Attention O(n²d)', linewidth=2)
    plt.loglog(seq_lengths, ffn_complexity, 'r-', label='FFN O(nd²)', linewidth=2)
    plt.title('Computational Complexity', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Operations (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Encoder Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Parameters per layer: {layer_params:,}")
    print(f"   Memory usage (approx): {total_params * 4 / 1e6:.1f} MB")
    print(f"   Attention parameters: {attn_params * num_layers:,}")
    print(f"   Feed-forward parameters: {ffn_params * num_layers:,}")
    
    return {
        'encoder': encoder,
        'positional_encoding': pos_encoding,
        'encoder_output': encoder_output,
        'attention_weights': all_attention_weights,
        'layer_similarities': layer_similarities,
        'attention_entropies': attention_entropies,
        'total_parameters': total_params
    }