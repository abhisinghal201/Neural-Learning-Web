"""
Neural Odyssey - Week 25: Attention Mechanisms - The Foundation of Modern AI
Phase 3: Advanced Topics and Modern AI (Week 1)

The Attention Revolution

Welcome to the most revolutionary concept in modern AI! This week, you'll master attention 
mechanisms - the breakthrough that eliminated the need for recurrent processing and enabled 
the transformer revolution that powers ChatGPT, BERT, and virtually every state-of-the-art 
AI system today.

From the bottleneck problems of RNNs to the parallel processing power of attention, you'll 
understand why "Attention Is All You Need" became the most influential AI paper of the 2010s.

Journey Overview:
1. Historical context: From RNNs to attention mechanisms
2. Mathematical foundations: Query, Key, Value paradigm
3. Implementation mastery: Build attention from scratch
4. Visualization and interpretation: See what models attend to
5. Advanced variants: Multi-head, sparse, and linear attention
6. Real-world applications: Translation, summarization, and beyond
7. Cognitive connections: How attention mirrors human cognition
8. Transformer preparation: Building blocks for modern architectures

By week's end, you'll possess the foundational knowledge that underlies virtually every 
modern AI breakthrough, from GPT to DALL-E to autonomous driving systems.

Author: Neural Explorer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
import math
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("🎯 Neural Odyssey - Week 25: Attention Mechanisms")
print("=" * 60)
print("🧠 'Attention Is All You Need' - The Foundation of Modern AI")
print("🚀 Building the mechanisms that power ChatGPT, BERT, and beyond!")
print("=" * 60)


# ==========================================
# SECTION 1: HISTORICAL CONTEXT AND MOTIVATION
# ==========================================

class RNNBottleneckDemo:
    """
    Demonstrate the fundamental limitations of RNNs that motivated attention mechanisms
    """
    
    def __init__(self, hidden_size=128, max_length=50):
        self.hidden_size = hidden_size
        self.max_length = max_length
        
    def demonstrate_information_bottleneck(self):
        """
        Show how RNN hidden states create information bottlenecks
        """
        print("\n🏛️  Historical Context: The RNN Bottleneck Problem")
        print("=" * 50)
        
        print("📚 The Problem with Sequence-to-Sequence Models:")
        print("   In 2014, encoder-decoder RNNs were state-of-the-art for translation")
        print("   But they had a fundamental flaw: the fixed-size hidden state bottleneck")
        print()
        
        # Simulate information compression in RNN encoder
        sequence_lengths = [5, 10, 20, 30, 40, 50]
        information_retained = []
        
        for length in sequence_lengths:
            # Simulate information loss as sequence gets longer
            # This is a simplified model of the bottleneck effect
            information = 1.0 * np.exp(-length / 30.0)  # Exponential decay
            information_retained.append(information)
        
        # Visualize the bottleneck problem
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(sequence_lengths, information_retained, 'ro-', linewidth=2, markersize=8)
        plt.title('RNN Information Bottleneck Problem', fontsize=14, fontweight='bold')
        plt.xlabel('Input Sequence Length')
        plt.ylabel('Information Retained')
        plt.grid(True, alpha=0.3)
        plt.text(25, 0.7, 'Information lost\nfor long sequences!', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Show translation quality degradation
        plt.subplot(2, 2, 2)
        translation_quality = [0.95, 0.92, 0.85, 0.75, 0.65, 0.55]
        plt.plot(sequence_lengths, translation_quality, 'bo-', linewidth=2, markersize=8)
        plt.title('Translation Quality vs Sequence Length', fontsize=14, fontweight='bold')
        plt.xlabel('Source Sentence Length')
        plt.ylabel('BLEU Score')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Acceptable Quality')
        plt.legend()
        
        # Show attention solution preview
        plt.subplot(2, 2, 3)
        attention_quality = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]  # Much more stable
        plt.plot(sequence_lengths, translation_quality, 'bo-', label='RNN Encoder-Decoder', 
                linewidth=2, markersize=8)
        plt.plot(sequence_lengths, attention_quality, 'go-', label='With Attention', 
                linewidth=2, markersize=8)
        plt.title('Attention Mechanism Solution', fontsize=14, fontweight='bold')
        plt.xlabel('Source Sentence Length')
        plt.ylabel('Translation Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show parallel processing advantage
        plt.subplot(2, 2, 4)
        rnn_time = [t**2 for t in sequence_lengths]  # Sequential processing
        attention_time = [t * 1.2 for t in sequence_lengths]  # Parallel processing
        plt.plot(sequence_lengths, rnn_time, 'ro-', label='RNN (Sequential)', 
                linewidth=2, markersize=8)
        plt.plot(sequence_lengths, attention_time, 'go-', label='Attention (Parallel)', 
                linewidth=2, markersize=8)
        plt.title('Processing Time Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Sequence Length')
        plt.ylabel('Relative Processing Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("🔍 Key Insights from the Bottleneck Analysis:")
        print("   1. RNNs compress entire sequences into fixed-size hidden states")
        print("   2. Information is lost as sequences get longer (especially early tokens)")
        print("   3. Translation quality degrades significantly for long sentences")
        print("   4. Sequential processing prevents parallelization")
        print("   5. Attention mechanisms solve all these problems!")
        
        return {
            'sequence_lengths': sequence_lengths,
            'information_retained': information_retained,
            'translation_quality': translation_quality,
            'attention_quality': attention_quality
        }
    
    def demonstrate_alignment_problem(self):
        """
        Show how RNNs struggle with word alignment in translation
        """
        print("\n🎯 The Alignment Problem in Neural Machine Translation")
        print("=" * 50)
        
        # Example: English to French translation
        english = ["The", "quick", "brown", "fox", "jumps"]
        french = ["Le", "rapide", "renard", "brun", "saute"]
        
        print(f"📝 Example Translation:")
        print(f"   English: {' '.join(english)}")
        print(f"   French:  {' '.join(french)}")
        print()
        
        # Show correct alignment
        correct_alignment = [
            ("The", "Le"),
            ("quick", "rapide"), 
            ("brown", "brun"),
            ("fox", "renard"),
            ("jumps", "saute")
        ]
        
        print("✅ Correct Word Alignment:")
        for eng, fr in correct_alignment:
            print(f"   {eng:6} ↔ {fr}")
        print()
        
        # Simulate RNN encoder-decoder confusion
        print("❌ RNN Encoder-Decoder Problems:")
        print("   • Fixed encoding loses positional information")
        print("   • Decoder can't directly access specific source words")
        print("   • Alignment is implicit and often incorrect")
        print("   • No way to visualize what the model is focusing on")
        print()
        
        print("🎯 How Attention Solves This:")
        print("   • Dynamic alignment: decoder can attend to any source position")
        print("   • Explicit attention weights show word correspondences")  
        print("   • Different target words can focus on different source words")
        print("   • Attention weights are interpretable and visualizable")
        
        return correct_alignment


def historical_attention_timeline():
    """
    Comprehensive timeline of attention mechanism development
    """
    print("\n📅 The Attention Revolution Timeline")
    print("=" * 50)
    
    timeline = [
        {
            'year': 2014,
            'paper': 'Neural Machine Translation by Jointly Learning to Align and Translate',
            'authors': 'Bahdanau, Cho, Bengio',
            'breakthrough': 'First successful attention mechanism',
            'impact': 'Solved the bottleneck problem in seq2seq models',
            'key_idea': 'Let decoder attend to different encoder states'
        },
        {
            'year': 2015,
            'paper': 'Effective Approaches to Attention-based Neural Machine Translation',
            'authors': 'Luong, Pham, Manning',
            'breakthrough': 'Simplified attention computation',
            'impact': 'Made attention more practical and efficient',
            'key_idea': 'Global vs local attention, different scoring functions'
        },
        {
            'year': 2016,
            'paper': 'Show, Attend and Tell',
            'authors': 'Xu et al.',
            'breakthrough': 'Visual attention for image captioning',
            'impact': 'Extended attention to computer vision',
            'key_idea': 'Attend to different image regions when generating words'
        },
        {
            'year': 2017,
            'paper': 'Attention Is All You Need',
            'authors': 'Vaswani et al.',
            'breakthrough': 'Transformer architecture with self-attention',
            'impact': 'Eliminated recurrence entirely, enabled modern LLMs',
            'key_idea': 'Self-attention can model all sequence dependencies'
        },
        {
            'year': 2018,
            'paper': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'authors': 'Devlin et al.',
            'breakthrough': 'Bidirectional self-attention pretraining',
            'impact': 'Revolutionary NLP representations',
            'key_idea': 'Contextual embeddings from bidirectional attention'
        },
        {
            'year': 2019,
            'paper': 'Language Models are Unsupervised Multitask Learners (GPT-2)',
            'authors': 'Radford et al.',
            'breakthrough': 'Scaled transformer language modeling',
            'impact': 'Demonstrated emergence of capabilities at scale',
            'key_idea': 'Causal self-attention for autoregressive generation'
        },
        {
            'year': 2020,
            'paper': 'An Image is Worth 16x16 Words: Transformers for Image Recognition',
            'authors': 'Dosovitskiy et al.',
            'breakthrough': 'Vision Transformer (ViT)',
            'impact': 'Brought transformers to computer vision',
            'key_idea': 'Treat image patches as sequence tokens'
        }
    ]
    
    for milestone in timeline:
        print(f"🏆 {milestone['year']}: {milestone['breakthrough']}")
        print(f"   📄 Paper: {milestone['paper']}")
        print(f"   👥 Authors: {milestone['authors']}")  
        print(f"   💡 Key Idea: {milestone['key_idea']}")
        print(f"   🌟 Impact: {milestone['impact']}")
        print()
    
    print("🚀 The Path Forward:")
    print("   2021-2024: GPT-3, ChatGPT, GPT-4, Multimodal Transformers")
    print("   Today: Attention mechanisms power virtually every SOTA AI system")
    print("   Future: Efficient attention, long-context models, multimodal fusion")
    
    return timeline


# ==========================================
# SECTION 2: MATHEMATICAL FOUNDATIONS
# ==========================================

class AttentionMathematicalFoundations:
    """
    Deep dive into the mathematical principles underlying attention mechanisms
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
    
    def explain_query_key_value_paradigm(self):
        """
        Explain the Query-Key-Value paradigm with intuitive examples
        """
        print("\n🗝️  The Query-Key-Value Paradigm")
        print("=" * 50)
        
        print("🧠 Cognitive Analogy - Library Search:")
        print("   Query (Q): What you're looking for ('machine learning books')")
        print("   Keys (K):  Book catalog entries/keywords")  
        print("   Values (V): The actual books on the shelf")
        print("   Process:   Match query against keys, retrieve relevant values")
        print()
        
        print("🔍 Database Lookup Analogy:")
        print("   Query (Q): SQL query 'SELECT * FROM users WHERE age > 25'")
        print("   Keys (K):  Index keys (age column values)")
        print("   Values (V): Complete user records")
        print("   Process:   Index lookup followed by value retrieval")
        print()
        
        print("🧮 Mathematical Formulation:")
        print("   Attention(Q,K,V) = softmax(score(Q,K))V")
        print("   Where:")
        print("   • Q ∈ ℝ^(n×d_k): Query matrix")
        print("   • K ∈ ℝ^(m×d_k): Key matrix") 
        print("   • V ∈ ℝ^(m×d_v): Value matrix")
        print("   • score(Q,K): Compatibility function")
        print()
        
        # Demonstrate with concrete example
        print("🎯 Concrete Example:")
        
        # Simple sequence: "The cat sat"
        vocab = ["The", "cat", "sat", "<pad>"]
        sequence = [0, 1, 2, 3]  # Token indices
        
        # Create simple embeddings (normally learned)
        d_model = 4
        embeddings = torch.tensor([
            [1.0, 0.5, 0.2, 0.1],  # "The"
            [0.3, 1.0, 0.8, 0.4],  # "cat"  
            [0.6, 0.2, 1.0, 0.7],  # "sat"
            [0.0, 0.0, 0.0, 0.0],  # "<pad>"
        ])
        
        print(f"   Input sequence: {[vocab[i] for i in sequence[:3]]}")
        print(f"   Embeddings shape: {embeddings.shape}")
        print()
        
        # Create Q, K, V matrices (simplified - normally via learned projections)
        Q = embeddings  # In self-attention, Q=K=V=input
        K = embeddings
        V = embeddings
        
        print("   Query-Key-Value Setup (Self-Attention):")
        print(f"   Q = K = V = input embeddings")
        print(f"   Each token queries all tokens (including itself)")
        print()
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        print("   Attention Scores (Q @ K^T / √d_k):")
        print(f"   Shape: {scores.shape}")
        
        # Show score matrix with labels
        score_matrix = scores[:3, :3].numpy()  # Exclude padding
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(score_matrix, 
                   xticklabels=vocab[:3], 
                   yticklabels=vocab[:3],
                   annot=True, fmt='.2f', cmap='Blues')
        plt.title('Raw Attention Scores (Q @ K^T / √d_k)', fontweight='bold')
        plt.ylabel('Query (What we\'re looking for)')
        plt.xlabel('Key (What we\'re comparing against)')
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        weight_matrix = attention_weights[:3, :3].numpy()
        
        plt.subplot(2, 2, 2)
        sns.heatmap(weight_matrix,
                   xticklabels=vocab[:3],
                   yticklabels=vocab[:3], 
                   annot=True, fmt='.3f', cmap='Greens')
        plt.title('Attention Weights (Softmax Normalized)', fontweight='bold')
        plt.ylabel('Query Position')
        plt.xlabel('Key Position')
        
        # Compute final output
        output = torch.matmul(attention_weights, V)
        
        plt.subplot(2, 2, 3)
        original_emb = embeddings[:3].numpy()
        attended_emb = output[:3].numpy()
        
        positions = np.arange(len(vocab[:3]))
        width = 0.35
        
        for i in range(d_model):
            plt.bar(positions - width/2, original_emb[:, i], width, 
                   alpha=0.7, label=f'Original Dim {i}' if i < 2 else "")
            plt.bar(positions + width/2, attended_emb[:, i], width,
                   alpha=0.7, label=f'Attended Dim {i}' if i < 2 else "")
        
        plt.title('Original vs Attended Embeddings', fontweight='bold')
        plt.xlabel('Token Position')
        plt.ylabel('Embedding Value')
        plt.xticks(positions, vocab[:3])
        if len(vocab[:3]) == 3:  # Only show legend for small examples
            plt.legend()
        
        # Show attention interpretation
        plt.subplot(2, 2, 4)
        # Create attention flow diagram
        for i, query_word in enumerate(vocab[:3]):
            weights = weight_matrix[i]
            max_attention_idx = np.argmax(weights)
            
            plt.text(0.1, 0.8 - i*0.25, f"'{query_word}' attends most to:", 
                    fontsize=12, fontweight='bold')
            plt.text(0.1, 0.75 - i*0.25, f"'{vocab[max_attention_idx]}' ({weights[max_attention_idx]:.3f})",
                    fontsize=10, color='darkgreen')
            
            # Show top 2 attention targets
            sorted_indices = np.argsort(weights)[::-1]
            for j, idx in enumerate(sorted_indices[:2]):
                color = plt.cm.Greens(weights[idx])
                plt.text(0.5 + j*0.2, 0.8 - i*0.25, f"{vocab[idx]}", 
                        fontsize=10, color=color, fontweight='bold')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Attention Interpretation', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("🎯 Key Observations:")
        print("   1. Each query attends to all keys (including itself)")
        print("   2. Attention weights are normalized probabilities (sum to 1)")
        print("   3. Output is weighted combination of values")
        print("   4. Different queries can have different attention patterns")
        print("   5. Self-attention allows tokens to gather context from sequence")
        
        return {
            'embeddings': embeddings,
            'scores': scores,
            'attention_weights': attention_weights,
            'output': output,
            'vocab': vocab
        }
    
    def derive_scaled_dot_product_attention(self):
        """
        Mathematical derivation of scaled dot-product attention
        """
        print("\n📊 Scaled Dot-Product Attention Derivation")
        print("=" * 50)
        
        print("🎯 Goal: Efficient attention computation for transformer models")
        print()
        
        print("📐 Step 1: Basic Dot-Product Attention")
        print("   score(q_i, k_j) = q_i · k_j")
        print("   • Fast computation (single matrix multiplication)")
        print("   • But: values can become very large for high dimensions")
        print()
        
        print("📐 Step 2: The Scaling Problem")
        print("   For d_k-dimensional vectors with unit variance:")
        print("   Var(q_i · k_j) = d_k")
        print("   • Dot products grow with dimension")
        print("   • Large values push softmax into saturation regions")
        print("   • Gradients become very small")
        print()
        
        # Demonstrate the scaling problem
        dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        dot_product_vars = []
        gradient_magnitudes = []
        
        for d_k in dimensions:
            # Generate random unit vectors
            q = torch.randn(d_k) / math.sqrt(d_k)
            k = torch.randn(100, d_k) / math.sqrt(d_k)  # 100 key vectors
            
            # Compute dot products
            dots = torch.matmul(q, k.T)
            dot_product_vars.append(torch.var(dots).item())
            
            # Compute softmax and gradients
            attention_weights = F.softmax(dots, dim=0)
            
            # Simulate gradient (derivative of softmax)
            # For softmax, gradient magnitude decreases when max is much larger than others
            max_weight = torch.max(attention_weights)
            gradient_magnitude = (max_weight * (1 - max_weight)).item()
            gradient_magnitudes.append(gradient_magnitude)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.semilogx(dimensions, dot_product_vars, 'ro-', linewidth=2, markersize=8)
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Expected (d_k)')
        plt.title('Dot Product Variance vs Dimension', fontweight='bold')
        plt.xlabel('Dimension (d_k)')
        plt.ylabel('Variance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.semilogx(dimensions, gradient_magnitudes, 'bo-', linewidth=2, markersize=8)
        plt.title('Gradient Magnitude vs Dimension', fontweight='bold')
        plt.xlabel('Dimension (d_k)')
        plt.ylabel('Gradient Magnitude')
        plt.grid(True, alpha=0.3)
        
        # Show scaled version
        scaled_vars = [v / d_k for v, d_k in zip(dot_product_vars, dimensions)]
        plt.subplot(1, 3, 3)
        plt.semilogx(dimensions, dot_product_vars, 'ro-', label='Unscaled', linewidth=2, markersize=8)
        plt.semilogx(dimensions, scaled_vars, 'go-', label='Scaled by √d_k', linewidth=2, markersize=8)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Target Variance')
        plt.title('Scaling Solution', fontweight='bold')
        plt.xlabel('Dimension (d_k)')
        plt.ylabel('Variance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("📐 Step 3: The Scaling Solution")
        print("   score(q_i, k_j) = (q_i · k_j) / √d_k")
        print("   • Keeps variance constant regardless of dimension")
        print("   • Prevents softmax saturation")
        print("   • Maintains healthy gradients")
        print()
        
        print("📐 Step 4: Complete Scaled Dot-Product Attention")
        print("   Attention(Q,K,V) = softmax(QK^T / √d_k)V")
        print()
        print("   Where:")
        print("   • Q ∈ ℝ^(n×d_k): n queries of dimension d_k")
        print("   • K ∈ ℝ^(m×d_k): m keys of dimension d_k")
        print("   • V ∈ ℝ^(m×d_v): m values of dimension d_v")
        print("   • QK^T ∈ ℝ^(n×m): attention score matrix")
        print("   • softmax normalizes each row to probability distribution")
        print("   • Final output ∈ ℝ^(n×d_v)")
        print()
        
        print("🎯 Computational Advantages:")
        print("   1. Highly parallelizable (matrix operations)")
        print("   2. No sequential dependencies")
        print("   3. Stable training dynamics")
        print("   4. Scales efficiently with sequence length")
        
        return {
            'dimensions': dimensions,
            'dot_product_vars': dot_product_vars,
            'gradient_magnitudes': gradient_magnitudes,
            'scaled_vars': scaled_vars
        }
    
    def explain_positional_encoding(self):
        """
        Explain why and how positional encoding works with attention
        """
        print("\n📍 Positional Encoding: Giving Attention a Sense of Order")
        print("=" * 50)
        
        print("🤔 The Problem: Attention is Permutation Invariant")
        print("   Without positional information:")
        print("   'The cat sat on the mat' ≡ 'mat the on sat cat The'")
        print("   Attention produces identical results for any permutation!")
        print()
        
        # Demonstrate permutation invariance
        sentence1 = ["The", "cat", "sat", "on", "mat"]
        sentence2 = ["mat", "The", "on", "sat", "cat"]  # Scrambled
        
        print(f"   Original: {' '.join(sentence1)}")
        print(f"   Scrambled: {' '.join(sentence2)}")
        print()
        
        # Simple embeddings
        d_model = 64
        vocab_size = 10
        embedding = nn.Embedding(vocab_size, d_model)
        
        # Convert to indices (simplified)
        indices1 = torch.tensor([0, 1, 2, 3, 4])
        indices2 = torch.tensor([4, 0, 3, 2, 1])  # Scrambled order
        
        emb1 = embedding(indices1)
        emb2 = embedding(indices2)
        
        # Apply self-attention (without positional encoding)
        attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        
        with torch.no_grad():
            out1, attn1 = attention(emb1.unsqueeze(0), emb1.unsqueeze(0), emb1.unsqueeze(0))
            # For scrambled: need to reorder to match original vocab
            emb2_reordered = emb2[[1, 4, 3, 2, 0]]  # Reorder to match original
            out2, attn2 = attention(emb2_reordered.unsqueeze(0), emb2_reordered.unsqueeze(0), emb2_reordered.unsqueeze(0))
        
        print("🔍 Sinusoidal Positional Encoding Solution")
        print("   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
        print("   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
        print()
        
        def get_positional_encoding(max_len, d_model):
            """Generate sinusoidal positional encodings"""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return pe
        
        # Generate positional encodings
        max_len = 50
        pos_encoding = get_positional_encoding(max_len, d_model)
        
        # Visualize positional encodings
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 3)
        # Show odd dimensions (cosine)
        for dim in [1, 11, 21, 31]:
            if dim < d_model:
                plt.plot(positions, pos_encoding[:50, dim], label=f'Dim {dim}', linewidth=2)
        
        plt.title('Positional Encoding - Odd Dimensions (cos)', fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        # Show relative position encoding properties
        pos_diff = []
        encoding_similarity = []
        
        for delta in range(1, 20):
            similarities = []
            for pos in range(20):
                if pos + delta < max_len:
                    sim = F.cosine_similarity(pos_encoding[pos:pos+1], 
                                            pos_encoding[pos+delta:pos+delta+1])
                    similarities.append(sim.item())
            pos_diff.append(delta)
            encoding_similarity.append(np.mean(similarities))
        
        plt.plot(pos_diff, encoding_similarity, 'ro-', linewidth=2, markersize=8)
        plt.title('Relative Position Similarity', fontweight='bold')
        plt.xlabel('Position Difference')
        plt.ylabel('Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        # Compare with and without positional encoding
        seq_len = 10
        base_emb = torch.randn(seq_len, d_model)
        pos_emb = base_emb + pos_encoding[:seq_len]
        
        # Compute pairwise similarities
        base_sim = torch.matmul(base_emb, base_emb.T)
        pos_sim = torch.matmul(pos_emb, pos_emb.T)
        
        # Show difference
        sim_diff = pos_sim - base_sim
        sns.heatmap(sim_diff.numpy(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        plt.title('Position Encoding Effect on Similarity', fontweight='bold')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        
        plt.subplot(2, 3, 6)
        # Show frequency spectrum
        freqs = []
        for i in range(0, d_model, 2):
            freq = 1.0 / (10000 ** (i / d_model))
            freqs.append(freq)
        
        plt.semilogy(range(len(freqs)), freqs, 'bo-', linewidth=2, markersize=8)
        plt.title('Positional Encoding Frequencies', fontweight='bold')
        plt.xlabel('Dimension Pair')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("🎯 Key Properties of Sinusoidal Positional Encoding:")
        print("   1. Deterministic: Same encoding for same position across sequences")
        print("   2. Unique: Each position has a unique encoding")
        print("   3. Relative: Model can learn to attend by relative position")
        print("   4. Extrapolation: Can handle longer sequences than training")
        print("   5. Smooth: Similar positions have similar encodings")
        print()
        
        print("🔍 Alternative Approaches:")
        print("   1. Learned Positional Embeddings: Parameters learned during training")
        print("   2. Relative Positional Encoding: Encode relative distances")
        print("   3. Rotary Position Embedding (RoPE): Used in modern LLMs")
        print("   4. ALiBi: Attention with Linear Biases")
        
        return {
            'pos_encoding': pos_encoding,
            'encoding_similarity': encoding_similarity,
            'freqs': freqs
        }


# ==========================================
# SECTION 3: CORE ATTENTION IMPLEMENTATIONS
# ==========================================

class BasicAttentionMechanisms:
    """
    Implementation of core attention mechanisms from scratch
    """
    
    def __init__(self, d_model=64, device='cpu'):
        self.d_model = d_model
        self.device = torch.device(device)
    
    def additive_attention(self, query, key, value):
        """
        Bahdanau attention mechanism (additive attention)
        score(q,k) = v^T * tanh(W_q*q + W_k*k)
        """
        print("\n🧮 Implementing Additive Attention (Bahdanau et al., 2014)")
        print("=" * 50)
        
        batch_size, seq_len_q, d_k = query.shape
        _, seq_len_k, _ = key.shape
        _, _, d_v = value.shape
        
        # Learned parameters
        W_q = nn.Linear(d_k, self.d_model, bias=False)
        W_k = nn.Linear(d_k, self.d_model, bias=False)
        v = nn.Linear(self.d_model, 1, bias=False)
        
        print(f"📊 Input shapes:")
        print(f"   Query: {query.shape}")
        print(f"   Key: {key.shape}")
        print(f"   Value: {value.shape}")
        print()
        
        # Compute attention scores
        q_transformed = W_q(query)  # [batch, seq_len_q, d_model]
        k_transformed = W_k(key)    # [batch, seq_len_k, d_model]
        
        # Broadcast and combine
        q_expanded = q_transformed.unsqueeze(2)  # [batch, seq_len_q, 1, d_model]
        k_expanded = k_transformed.unsqueeze(1)  # [batch, 1, seq_len_k, d_model]
        
        # Add and apply tanh
        combined = torch.tanh(q_expanded + k_expanded)  # [batch, seq_len_q, seq_len_k, d_model]
        
        # Apply final linear layer
        scores = v(combined).squeeze(-1)  # [batch, seq_len_q, seq_len_k]
        
        print(f"✅ Attention scores shape: {scores.shape}")
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape}")
        
        return output, attention_weights
    
    def multiplicative_attention(self, query, key, value, scale=True):
        """
        Luong attention mechanism (multiplicative attention)
        score(q,k) = q^T * k (optionally scaled)
        """
        print("\n🧮 Implementing Multiplicative Attention (Luong et al., 2015)")
        print("=" * 50)
        
        print(f"📊 Input shapes:")
        print(f"   Query: {query.shape}")
        print(f"   Key: {key.shape}")
        print(f"   Value: {value.shape}")
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        if scale:
            d_k = query.size(-1)
            scores = scores / math.sqrt(d_k)
            print(f"🔧 Applied scaling by √{d_k} = {math.sqrt(d_k):.2f}")
        
        print(f"✅ Raw scores shape: {scores.shape}")
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape}")
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, query, key, value, mask=None, dropout_p=0.0):
        """
        Scaled dot-product attention from "Attention Is All You Need"
        """
        print("\n🧮 Implementing Scaled Dot-Product Attention (Vaswani et al., 2017)")
        print("=" * 50)
        
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        print(f"📊 Attention computation:")
        print(f"   QK^T shape: {scores.shape}")
        print(f"   Scaling factor: 1/√{d_k} = {1/math.sqrt(d_k):.4f}")
        
        # Apply mask if provided
        if mask is not None:
            print(f"🎭 Applying mask: {mask.shape}")
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Dropout (if training)
        if dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        print(f"✅ Final output shape: {output.shape}")
        
        return output, attention_weights
    
    def compare_attention_mechanisms(self):
        """
        Compare different attention mechanisms on the same input
        """
        print("\n🔍 Comparing Attention Mechanisms")
        print("=" * 50)
        
        # Create test input
        batch_size, seq_len, d_model = 2, 8, 64
        
        torch.manual_seed(42)  # For reproducible comparison
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        print(f"🎯 Test Setup:")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Model dimension: {d_model}")
        print()
        
        # Test all mechanisms
        mechanisms = {}
        
        # Additive attention
        with torch.no_grad():
            try:
                add_out, add_attn = self.additive_attention(query, key, value)
                mechanisms['Additive (Bahdanau)'] = {
                    'output': add_out,
                    'attention': add_attn,
                    'complexity': 'O(T*d*h) where h is hidden size'
                }
            except Exception as e:
                print(f"❌ Additive attention failed: {e}")
        
        # Multiplicative attention
        with torch.no_grad():
            mult_out, mult_attn = self.multiplicative_attention(query, key, value, scale=False)
            mechanisms['Multiplicative (Unscaled)'] = {
                'output': mult_out,
                'attention': mult_attn,
                'complexity': 'O(T*d)'
            }
        
        # Scaled dot-product attention
        with torch.no_grad():
            scaled_out, scaled_attn = self.scaled_dot_product_attention(query, key, value)
            mechanisms['Scaled Dot-Product'] = {
                'output': scaled_out,
                'attention': scaled_attn,
                'complexity': 'O(T*d)'
            }
        
        # Visualize attention patterns
        fig, axes = plt.subplots(2, len(mechanisms), figsize=(15, 8))
        
        for i, (name, data) in enumerate(mechanisms.items()):
            # Plot attention weights for first batch item
            attn_matrix = data['attention'][0].numpy()
            
            # First row: attention heatmap
            sns.heatmap(attn_matrix, ax=axes[0, i], cmap='Blues', 
                       cbar=True, square=True)
            axes[0, i].set_title(f'{name}\nAttention Weights', fontweight='bold')
            axes[0, i].set_xlabel('Key Position')
            axes[0, i].set_ylabel('Query Position')
            
            # Second row: attention statistics
            axes[1, i].hist(attn_matrix.flatten(), bins=20, alpha=0.7, density=True)
            axes[1, i].set_title(f'Attention Distribution', fontweight='bold')
            axes[1, i].set_xlabel('Attention Weight')
            axes[1, i].set_ylabel('Density')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Compare computational complexity
        print("⚡ Computational Complexity Comparison:")
        for name, data in mechanisms.items():
            print(f"   {name}: {data['complexity']}")
        print()
        
        # Compare attention entropy (measure of focus vs. diffusion)
        print("🎯 Attention Analysis:")
        for name, data in mechanisms.items():
            attn_weights = data['attention'][0]  # First batch item
            
            # Compute entropy for each query position
            entropies = []
            for i in range(attn_weights.size(0)):
                entropy = -torch.sum(attn_weights[i] * torch.log(attn_weights[i] + 1e-8))
                entropies.append(entropy.item())
            
            avg_entropy = np.mean(entropies)
            max_entropy = math.log(seq_len)  # Maximum possible entropy
            normalized_entropy = avg_entropy / max_entropy
            
            print(f"   {name}:")
            print(f"     Average entropy: {avg_entropy:.3f}")
            print(f"     Normalized entropy: {normalized_entropy:.3f}")
            print(f"     Focus level: {'High' if normalized_entropy < 0.5 else 'Low'}")
        
        return mechanisms


# ==========================================
# SECTION 4: MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation from scratch
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
        
        print(f"🔧 MultiHeadAttention initialized:")
        print(f"   Model dimension: {d_model}")
        print(f"   Number of heads: {num_heads}")
        print(f"   Dimension per head: {self.d_k}")
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. Linear projections
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)    # [batch, seq_len, d_model]
        V = self.W_v(value)  # [batch, seq_len, d_model]
        
        # 2. Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, d_k]
        
        # 3. Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask=mask
        )
        
        # 4. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. Final projection
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention for multi-head"""
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


def demonstrate_multi_head_attention():
    """
    Comprehensive demonstration of multi-head attention
    """
    print("\n🧠 Multi-Head Attention Deep Dive")
    print("=" * 50)
    
    print("💡 Key Insight: Multiple attention heads can focus on different aspects")
    print("   Head 1: Syntactic relationships (subject-verb agreement)")
    print("   Head 2: Semantic relationships (word meanings)")
    print("   Head 3: Positional relationships (nearby words)")
    print("   Head 4: Long-range dependencies (discourse connections)")
    print()
    
    # Setup
    d_model = 128
    num_heads = 8
    seq_len = 12
    batch_size = 2
    
    # Create multi-head attention module
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)
    
    # Generate test input (simulate token embeddings)
    torch.manual_seed(42)
    input_embeddings = torch.randn(batch_size, seq_len, d_model)
    
    print(f"🎯 Demonstration Setup:")
    print(f"   Input shape: {input_embeddings.shape}")
    print(f"   Model dimension: {d_model}")
    print(f"   Number of heads: {num_heads}")
    print(f"   Dimension per head: {d_model // num_heads}")
    print()
    
    # Forward pass
    with torch.no_grad():
        output, attention_weights = mha(input_embeddings, input_embeddings, input_embeddings)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    print(f"   [batch_size, num_heads, seq_len, seq_len]")
    print()
    
    # Analyze different attention heads
    plt.figure(figsize=(20, 12))
    
    # First batch item, all heads
    attn_matrices = attention_weights[0]  # [num_heads, seq_len, seq_len]
    
    # Plot individual heads
    for head in range(min(8, num_heads)):
        plt.subplot(2, 4, head + 1)
        
        attn_matrix = attn_matrices[head].numpy()
        sns.heatmap(attn_matrix, cmap='Blues', cbar=True, square=True,
                   xticklabels=range(seq_len), yticklabels=range(seq_len))
        plt.title(f'Head {head + 1}', fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
    
    plt.suptitle('Multi-Head Attention Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Analyze head specialization
    print("🔍 Attention Head Analysis:")
    
    # Compute attention statistics for each head
    head_stats = {}
    
    for head in range(num_heads):
        attn_matrix = attn_matrices[head]
        
        # Entropy (measure of attention spread)
        entropies = []
        for i in range(seq_len):
            entropy = -torch.sum(attn_matrix[i] * torch.log(attn_matrix[i] + 1e-8))
            entropies.append(entropy.item())
        avg_entropy = np.mean(entropies)
        
        # Diagonal attention (self-attention strength)
        diagonal_strength = torch.mean(torch.diag(attn_matrix)).item()
        
        # Local vs global attention
        # Local: attention to nearby positions
        local_attention = 0
        global_attention = 0
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= 2:  # Local window
                    local_attention += attn_matrix[i, j].item()
                else:
                    global_attention += attn_matrix[i, j].item()
        
        local_attention /= seq_len * seq_len
        global_attention /= seq_len * seq_len
        
        head_stats[head] = {
            'entropy': avg_entropy,
            'diagonal_strength': diagonal_strength,
            'local_attention': local_attention,
            'global_attention': global_attention
        }
        
        print(f"   Head {head + 1}:")
        print(f"     Entropy: {avg_entropy:.3f} (higher = more spread out)")
        print(f"     Self-attention: {diagonal_strength:.3f}")
        print(f"     Local focus: {local_attention:.3f}")
        print(f"     Global focus: {global_attention:.3f}")
        
        # Characterize head behavior
        if diagonal_strength > 0.3:
            behavior = "Self-focused"
        elif local_attention > global_attention:
            behavior = "Local pattern detector"
        elif avg_entropy > 2.0:
            behavior = "Global context aggregator"
        else:
            behavior = "Selective attention"
        
        print(f"     Behavior: {behavior}")
        print()
    
    # Visualize head statistics
    plt.figure(figsize=(15, 10))
    
    heads = list(range(1, num_heads + 1))
    entropies = [head_stats[h]['entropy'] for h in range(num_heads)]
    diagonal_strengths = [head_stats[h]['diagonal_strength'] for h in range(num_heads)]
    local_attentions = [head_stats[h]['local_attention'] for h in range(num_heads)]
    global_attentions = [head_stats[h]['global_attention'] for h in range(num_heads)]
    
    plt.subplot(2, 2, 1)
    plt.bar(heads, entropies, color='skyblue', alpha=0.7)
    plt.title('Attention Entropy by Head', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Average Entropy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(heads, diagonal_strengths, color='lightcoral', alpha=0.7)
    plt.title('Self-Attention Strength by Head', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Diagonal Attention Weight')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    x = np.arange(len(heads))
    width = 0.35
    
    plt.bar(x - width/2, local_attentions, width, label='Local', alpha=0.7, color='green')
    plt.bar(x + width/2, global_attentions, width, label='Global', alpha=0.7, color='orange')
    plt.title('Local vs Global Attention', fontweight='bold')
    plt.xlabel('Attention Head')
    plt.ylabel('Attention Weight')
    plt.xticks(x, heads)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Scatter plot: entropy vs self-attention
    plt.scatter(diagonal_strengths, entropies, c=heads, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Head Number')
    plt.title('Head Specialization Patterns', fontweight='bold')
    plt.xlabel('Self-Attention Strength')
    plt.ylabel('Attention Entropy')
    plt.grid(True, alpha=0.3)
    
    # Add quadrant labels
    plt.axhline(y=np.mean(entropies), color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=np.mean(diagonal_strengths), color='k', linestyle='--', alpha=0.5)
    
    plt.text(0.1, max(entropies) * 0.9, 'High Entropy\nLow Self-Attention', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.text(max(diagonal_strengths) * 0.7, max(entropies) * 0.9, 'High Entropy\nHigh Self-Attention',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("🎯 Multi-Head Attention Benefits:")
    print("   1. Parallel processing of different relationship types")
    print("   2. Increased model capacity without increasing depth")
    print("   3. Different heads can specialize in different patterns")
    print("   4. Redundancy provides robustness")
    print("   5. Ensemble effect improves overall performance")
    
    return {
        'mha_module': mha,
        'output': output,
        'attention_weights': attention_weights,
        'head_stats': head_stats
    }


# ==========================================
# SECTION 5: ADVANCED ATTENTION VARIANTS
# ==========================================

class SparseAttention(nn.Module):
    """
    Sparse attention patterns for long sequences
    """
    
    def __init__(self, d_model, num_heads, pattern_type='local', window_size=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pattern_type = pattern_type
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        print(f"🔧 SparseAttention initialized:")
        print(f"   Pattern: {pattern_type}")
        print(f"   Window size: {window_size}")
    
    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask based on pattern type"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        if self.pattern_type == 'local':
            # Local attention: each position attends to nearby positions
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = 1
                
        elif self.pattern_type == 'strided':
            # Strided attention: attend to positions at regular intervals
            stride = self.window_size // 4
            for i in range(seq_len):
                # Local window
                start = max(0, i - stride)
                end = min(seq_len, i + stride + 1)
                mask[i, start:end] = 1
                
                # Strided positions
                for j in range(0, seq_len, stride):
                    mask[i, j] = 1
                    
        elif self.pattern_type == 'random':
            # Random sparse attention
            prob = self.window_size / seq_len
            mask = torch.bernoulli(torch.full((seq_len, seq_len), prob, device=device))
            # Ensure diagonal is always attended
            mask.fill_diagonal_(1)
        
        return mask
    
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        device = query.device
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, device)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse mask
        scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        
        # Softmax and attention
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights, sparse_mask


def demonstrate_sparse_attention():
    """
    Demonstrate different sparse attention patterns
    """
    print("\n⚡ Sparse Attention for Long Sequences")
    print("=" * 50)
    
    print("🎯 Motivation: Standard attention has O(n²) complexity")
    print("   For sequence length 1000: 1M attention weights")
    print("   For sequence length 10000: 100M attention weights")
    print("   Sparse patterns reduce this significantly")
    print()
    
    # Setup
    d_model = 64
    num_heads = 4
    seq_len = 64  # Reduced for visualization
    
    patterns = ['local', 'strided', 'random']
    sparse_models = {}
    
    for pattern in patterns:
        sparse_models[pattern] = SparseAttention(
            d_model, num_heads, pattern_type=pattern, window_size=16
        )
    
    # Test input
    torch.manual_seed(42)
    test_input = torch.randn(1, seq_len, d_model)
    
    # Compare patterns
    plt.figure(figsize=(18, 12))
    
    for i, (pattern, model) in enumerate(sparse_models.items()):
        with torch.no_grad():
            output, attn_weights, sparse_mask = model(test_input, test_input, test_input)
        
        # Plot sparse mask
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(sparse_mask.cpu().numpy(), cmap='Blues', aspect='auto')
        plt.title(f'{pattern.capitalize()} Attention Mask', fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Plot actual attention weights (first head)
        plt.subplot(3, 3, i*3 + 2)
        attn_matrix = attn_weights[0, 0].cpu().numpy()  # First batch, first head
        plt.imshow(attn_matrix, cmap='Blues', aspect='auto')
        plt.title(f'{pattern.capitalize()} Attention Weights', fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Plot sparsity statistics
        plt.subplot(3, 3, i*3 + 3)
        
        # Compute sparsity metrics
        total_positions = seq_len * seq_len
        active_positions = torch.sum(sparse_mask).item()
        sparsity_ratio = active_positions / total_positions
        
        # Attention entropy
        entropies = []
        for pos in range(seq_len):
            entropy = -torch.sum(attn_weights[0, 0, pos] * torch.log(attn_weights[0, 0, pos] + 1e-8))
            entropies.append(entropy.item())
        
        plt.hist(entropies, bins=20, alpha=0.7, density=True)
        plt.title(f'{pattern.capitalize()} Attention Entropy\nSparsity: {sparsity_ratio:.2f}', fontweight='bold')
        plt.xlabel('Entropy')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Complexity comparison
    print("📊 Complexity Analysis:")
    
    full_attention_ops = seq_len ** 2
    
    for pattern, model in sparse_models.items():
        with torch.no_grad():
            _, _, sparse_mask = model(test_input, test_input, test_input)
        
        sparse_ops = torch.sum(sparse_mask).item()
        reduction_factor = full_attention_ops / sparse_ops
        
        print(f"   {pattern.capitalize()} Attention:")
        print(f"     Operations: {sparse_ops:,} vs {full_attention_ops:,} (full)")
        print(f"     Reduction factor: {reduction_factor:.1f}x")
        print(f"     Memory savings: {(1 - sparse_ops/full_attention_ops)*100:.1f}%")
        print()
    
    return sparse_models


class LinearAttention(nn.Module):
    """
    Linear attention approximation using kernel methods
    """
    
    def __init__(self, d_model, num_heads=8, feature_dim=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_dim = feature_dim
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        print(f"🔧 LinearAttention initialized:")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Complexity: O(n*d) instead of O(n²*d)")
    
    def feature_map(self, x):
        """Apply feature map to approximate softmax kernel"""
        # Use random Fourier features to approximate RBF kernel
        # This is a simplified version - in practice, more sophisticated methods exist
        
        # Positive features to ensure non-negativity (required for softmax approximation)
        return F.relu(x) + 1e-6  # Small epsilon for numerical stability
    
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply feature maps
        Q_feat = self.feature_map(Q)  # [batch, heads, seq_len, d_k]
        K_feat = self.feature_map(K)  # [batch, heads, seq_len, d_k]
        
        # Linear attention computation
        # Instead of computing QK^T explicitly, we use the identity:
        # softmax(QK^T)V ≈ φ(Q)(φ(K)^T V) / (φ(Q) φ(K)^T 1)
        
        # Compute K^T V (numerator)
        KV = torch.matmul(K_feat.transpose(-2, -1), V)  # [batch, heads, d_k, d_k]
        
        # Compute normalization (denominator)
        K_sum = torch.sum(K_feat, dim=-2, keepdim=True)  # [batch, heads, 1, d_k]
        
        # Apply to queries
        numerator = torch.matmul(Q_feat, KV)  # [batch, heads, seq_len, d_k]
        denominator = torch.matmul(Q_feat, K_sum.transpose(-2, -1))  # [batch, heads, seq_len, 1]
        
        # Normalize
        output = numerator / (denominator + 1e-8)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output


def compare_attention_efficiency():
    """
    Compare computational efficiency of different attention mechanisms
    """
    print("\n⚡ Attention Efficiency Comparison")
    print("=" * 50)
    
    import time
    
    # Test different sequence lengths
    sequence_lengths = [64, 128, 256, 512, 1024]
    d_model = 128
    num_heads = 8
    batch_size = 4
    
    # Initialize models
    standard_attention = MultiHeadAttention(d_model, num_heads)
    linear_attention = LinearAttention(d_model, num_heads)
    sparse_attention = SparseAttention(d_model, num_heads, pattern_type='local', window_size=64)
    
    timing_results = {
        'Standard': [],
        'Linear': [],
        'Sparse': []
    }
    
    memory_usage = {
        'Standard': [],
        'Linear': [],
        'Sparse': []
    }
    
    print("🔬 Running efficiency benchmarks...")
    
    for seq_len in sequence_lengths:
        print(f"   Testing sequence length: {seq_len}")
        
        # Generate test data
        test_input = torch.randn(batch_size, seq_len, d_model)
        
        # Standard attention
        with torch.no_grad():
            start_time = time.time()
            std_output, _ = standard_attention(test_input, test_input, test_input)
            std_time = time.time() - start_time
            timing_results['Standard'].append(std_time)
            
            # Approximate memory usage (attention matrix)
            std_memory = batch_size * num_heads * seq_len * seq_len * 4  # 4 bytes per float32
            memory_usage['Standard'].append(std_memory / 1e6)  # Convert to MB
        
        # Linear attention
        with torch.no_grad():
            start_time = time.time()
            lin_output = linear_attention(test_input, test_input, test_input)
            lin_time = time.time() - start_time
            timing_results['Linear'].append(lin_time)
            
            # Linear attention memory (no explicit attention matrix)
            lin_memory = batch_size * num_heads * d_model * d_model * 4
            memory_usage['Linear'].append(lin_memory / 1e6)
        
        # Sparse attention
        with torch.no_grad():
            start_time = time.time()
            sparse_output, _, _ = sparse_attention(test_input, test_input, test_input)
            sparse_time = time.time() - start_time
            timing_results['Sparse'].append(sparse_time)
            
            # Sparse attention memory (reduced by sparsity factor)
            sparsity_factor = min(64, seq_len) / seq_len  # Local window
            sparse_memory = std_memory * sparsity_factor
            memory_usage['Sparse'].append(sparse_memory / 1e6)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for method, times in timing_results.items():
        plt.plot(sequence_lengths, times, 'o-', label=method, linewidth=2, markersize=8)
    
    plt.title('Computational Time vs Sequence Length', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    for method, memory in memory_usage.items():
        plt.plot(sequence_lengths, memory, 'o-', label=method, linewidth=2, markersize=8)
    
    plt.title('Memory Usage vs Sequence Length', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    # Theoretical complexity curves
    n_values = np.array(sequence_lengths)
    plt.plot(n_values, n_values**2, 'r--', label='O(n²) - Standard', linewidth=2)
    plt.plot(n_values, n_values, 'g--', label='O(n) - Linear', linewidth=2)
    plt.plot(n_values, n_values * np.log(n_values), 'b--', label='O(n log n) - Sparse', linewidth=2)
    
    plt.title('Theoretical Complexity Comparison', fontweight='bold')
    plt.xlabel('Sequence Length (n)')
    plt.ylabel('Operations (relative)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.subplot(2, 2, 4)
    # Speedup comparison
    for seq_idx, seq_len in enumerate(sequence_lengths):
        std_time = timing_results['Standard'][seq_idx]
        lin_speedup = std_time / timing_results['Linear'][seq_idx]
        sparse_speedup = std_time / timing_results['Sparse'][seq_idx]
        
        plt.bar([f'{seq_len}\nLinear', f'{seq_len}\nSparse'], 
               [lin_speedup, sparse_speedup], 
               alpha=0.7, color=['green', 'blue'])
    
    plt.title('Speedup vs Standard Attention', fontweight='bold')
    plt.ylabel('Speedup Factor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n📊 Efficiency Summary:")
    print("   Method      | Time Complexity | Space Complexity | Best Use Case")
    print("   ------------|-----------------|------------------|------------------")
    print("   Standard    | O(n²d)         | O(n²)           | Short sequences")
    print("   Linear      | O(nd²)         | O(d²)           | Long sequences, approximation OK")
    print("   Sparse      | O(nwd)         | O(nw)           | Long sequences, local patterns")
    print("   Where: n=seq_len, d=model_dim, w=window_size")
    
    return timing_results, memory_usage


# ==========================================
# SECTION 6: ATTENTION VISUALIZATION & INTERPRETATION
# ==========================================

class AttentionVisualizer:
    """
    Tools for visualizing and interpreting attention patterns
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_interpretable_example(self):
        """
        Create an example with interpretable attention patterns
        """
        print("\n🔍 Attention Pattern Analysis on Interpretable Example")
        print("=" * 50)
        
        # Create a sentence with clear syntactic structure
        sentence = "The quick brown fox jumps over the lazy dog"
        tokens = sentence.split()
        
        print(f"📝 Example sentence: '{sentence}'")
        print(f"   Tokens: {tokens}")
        print(f"   Length: {len(tokens)} tokens")
        print()
        
        # Create simple embeddings (in practice, these would be learned)
        vocab_size = len(tokens)
        d_model = 64
        
        # Simulate different types of relationships
        embeddings = torch.randn(vocab_size, d_model)
        
        # Add some structure to embeddings to make patterns more interpretable
        # Nouns get similar embeddings
        noun_indices = [0, 2, 3, 6, 8]  # "The", "brown", "fox", "the", "dog"
        for i in noun_indices:
            embeddings[i, :16] += 2.0  # Boost first 16 dimensions for nouns
        
        # Verbs get different pattern
        verb_indices = [4]  # "jumps"
        for i in verb_indices:
            embeddings[i, 16:32] += 2.0  # Boost different dimensions for verbs
        
        # Adjectives
        adj_indices = [1, 2, 7]  # "quick", "brown", "lazy"
        for i in adj_indices:
            embeddings[i, 32:48] += 2.0
        
        # Create multi-head attention
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)
        
        # Forward pass
        with torch.no_grad():
            input_emb = embeddings.unsqueeze(0)  # Add batch dimension
            output, attention_weights = mha(input_emb, input_emb, input_emb)
        
        # Analyze attention patterns
        attn_matrix = attention_weights[0]  # Remove batch dimension
        
        print("🧠 Attention Pattern Analysis:")
        
        # Analyze each head
        head_behaviors = []
        
        plt.figure(figsize=(20, 15))
        
        for head in range(num_heads):
            head_attn = attn_matrix[head].numpy()
            
            plt.subplot(3, num_heads, head + 1)
            sns.heatmap(head_attn, 
                       xticklabels=tokens, 
                       yticklabels=tokens,
                       annot=True, fmt='.2f', cmap='Blues', cbar=True)
            plt.title(f'Head {head + 1}', fontweight='bold')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Analyze patterns
            behavior = self.analyze_head_behavior(head_attn, tokens)
            head_behaviors.append(behavior)
            
            print(f"   Head {head + 1}: {behavior}")
        
        # Aggregate attention patterns
        plt.subplot(3, 1, 2)
        avg_attention = torch.mean(attn_matrix, dim=0).numpy()
        sns.heatmap(avg_attention,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   annot=True, fmt='.2f', cmap='Greens', cbar=True)
        plt.title('Average Attention Across All Heads', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Attention flow diagram
        plt.subplot(3, 1, 3)
        self.create_attention_flow_diagram(avg_attention, tokens)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'tokens': tokens,
            'embeddings': embeddings,
            'attention_weights': attention_weights,
            'head_behaviors': head_behaviors
        }
    
    def analyze_head_behavior(self, attention_matrix, tokens):
        """
        Analyze what pattern a specific attention head has learned
        """
        n_tokens = len(tokens)
        
        # Check for self-attention (diagonal elements)
        diagonal_strength = np.mean(np.diag(attention_matrix))
        
        # Check for local attention (nearby tokens)
        local_strength = 0
        for i in range(n_tokens):
            for j in range(max(0, i-1), min(n_tokens, i+2)):
                if i != j:
                    local_strength += attention_matrix[i, j]
        local_strength /= (n_tokens * 2 - 2)  # Normalize
        
        # Check for specific relationships
        # Article-noun relationships
        article_noun_strength = 0
        article_indices = [i for i, token in enumerate(tokens) if token.lower() in ['the', 'a', 'an']]
        noun_indices = [i for i, token in enumerate(tokens) if token.lower() in ['fox', 'dog']]
        
        count = 0
        for art_idx in article_indices:
            for noun_idx in noun_indices:
                if abs(art_idx - noun_idx) <= 2:  # Nearby article-noun pairs
                    article_noun_strength += attention_matrix[art_idx, noun_idx]
                    count += 1
        
        if count > 0:
            article_noun_strength /= count
        
        # Determine behavior
        if diagonal_strength > 0.4:
            return "Self-focused (high diagonal attention)"
        elif local_strength > 0.3:
            return "Local context (neighboring tokens)"
        elif article_noun_strength > 0.3:
            return "Syntactic relationships (articles→nouns)"
        else:
            return "Global context aggregation"
    
    def create_attention_flow_diagram(self, attention_matrix, tokens):
        """
        Create a flow diagram showing strongest attention connections
        """
        n_tokens = len(tokens)
        
        # Find strongest attention connections
        threshold = np.percentile(attention_matrix.flatten(), 80)  # Top 20%
        
        # Create a simple flow visualization
        pos_y = np.arange(n_tokens)
        
        # Draw tokens
        for i, token in enumerate(tokens):
            plt.text(0, i, token, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Draw attention flows
        for i in range(n_tokens):
            for j in range(n_tokens):
                if attention_matrix[i, j] > threshold and i != j:
                    # Draw arrow from token i to token j
                    alpha = min(1.0, attention_matrix[i, j] / 0.5)
                    plt.arrow(0.1, i, 0, j-i, head_width=0.1, head_length=0.1,
                             fc='red', ec='red', alpha=alpha, length_includes_head=True)
                    
                    # Add attention weight label
                    mid_y = (i + j) / 2
                    plt.text(0.05, mid_y, f'{attention_matrix[i, j]:.2f}',
                            fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7))
        
        plt.xlim(-0.2, 0.3)
        plt.ylim(-0.5, n_tokens - 0.5)
        plt.title('Attention Flow (Strongest Connections)', fontweight='bold')
        plt.axis('off')


def attention_interpretation_deep_dive():
    """
    Deep dive into attention interpretation techniques
    """
    print("\n🔬 Advanced Attention Interpretation Techniques")
    print("=" * 50)
    
    visualizer = AttentionVisualizer()
    
    # Create interpretable example
    results = visualizer.create_interpretable_example()
    
    print("\n🎯 Key Interpretation Insights:")
    print("   1. Different heads often specialize in different linguistic patterns")
    print("   2. Self-attention (diagonal) indicates token-specific processing")
    print("   3. Local attention suggests syntactic relationships")
    print("   4. Global attention indicates semantic/discourse connections")
    print("   5. Attention weights are not always explanations (ongoing debate)")
    print()
    
    print("🔍 Advanced Analysis Techniques:")
    techniques = [
        {
            'name': 'Attention Rollout',
            'description': 'Trace attention through multiple layers',
            'use_case': 'Understanding deep attention patterns'
        },
        {
            'name': 'Gradient × Attention',
            'description': 'Weight attention by gradient importance',
            'use_case': 'Finding causally important attention'
        },
        {
            'name': 'Attention Perturbation',
            'description': 'Modify attention weights and observe output changes',
            'use_case': 'Testing attention causality'
        },
        {
            'name': 'Attention Head Analysis',
            'description': 'Systematic analysis of head specialization',
            'use_case': 'Understanding model architecture'
        },
        {
            'name': 'Attention Entropy Analysis',
            'description': 'Measure attention concentration vs diffusion',
            'use_case': 'Understanding attention dynamics'
        }
    ]
    
    for technique in techniques:
        print(f"   • {technique['name']}: {technique['description']}")
        print(f"     Use case: {technique['use_case']}")
    
    print()
    print("⚠️  Important Caveats:")
    print("   • Attention weights ≠ feature importance")
    print("   • Multiple attention patterns can produce same output")
    print("   • Gradient-based importance may differ from attention")
    print("   • Attention is one lens among many for understanding models")
    
    return results


# ==========================================
# SECTION 7: COMPREHENSIVE WEEK INTEGRATION
# ==========================================

def comprehensive_attention_mastery_assessment():
    """
    Comprehensive assessment of attention mechanism mastery
    """
    print("\n🎓 Week 25 Mastery Assessment: Attention Mechanisms")
    print("=" * 60)
    
    print("🧠 Theoretical Understanding Assessment:")
    
    assessment_results = {}
    
    # 1. Historical Context
    print("\n1. Historical Context & Motivation")
    timeline = historical_attention_timeline()
    assessment_results['historical_knowledge'] = len(timeline) >= 5
    
    # 2. Mathematical Foundations
    print("\n2. Mathematical Foundations")
    foundations = AttentionMathematicalFoundations()
    qkv_demo = foundations.explain_query_key_value_paradigm()
    scaled_demo = foundations.derive_scaled_dot_product_attention()
    pos_demo = foundations.explain_positional_encoding()
    
    assessment_results['mathematical_understanding'] = True
    
    # 3. Implementation Mastery
    print("\n3. Implementation Skills")
    basic_attn = BasicAttentionMechanisms()
    comparison_results = basic_attn.compare_attention_mechanisms()
    
    mha_results = demonstrate_multi_head_attention()
    assessment_results['implementation_skills'] = len(comparison_results) >= 3
    
    # 4. Advanced Variants
    print("\n4. Advanced Attention Variants")
    sparse_results = demonstrate_sparse_attention()
    efficiency_results = compare_attention_efficiency()
    
    assessment_results['advanced_techniques'] = len(sparse_results) >= 2
    
    # 5. Interpretation & Analysis
    print("\n5. Attention Interpretation")
    interpretation_results = attention_interpretation_deep_dive()
    assessment_results['interpretation_skills'] = True
    
    # 6. RNN Bottleneck Understanding
    print("\n6. Historical Problem Understanding")
    rnn_demo = RNNBottleneckDemo()
    bottleneck_results = rnn_demo.demonstrate_information_bottleneck()
    alignment_results = rnn_demo.demonstrate_alignment_problem()
    
    assessment_results['problem_understanding'] = True
    
    # Final Assessment Visualization
    plt.figure(figsize=(15, 10))
    
    # Mastery radar chart
    categories = list(assessment_results.keys())
    scores = [1.0 if v else 0.5 for v in assessment_results.values()]
    
    # Add first point to close the radar chart
    categories.append(categories[0])
    scores.append(scores[0])
    
    # Create angles for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    plt.subplot(2, 2, 1)
    plt.polar(angles, scores, 'o-', linewidth=2, markersize=8)
    plt.fill(angles, scores, alpha=0.25)
    plt.thetagrids(angles[:-1] * 180/np.pi, categories[:-1], fontsize=10)
    plt.ylim(0, 1)
    plt.title('Attention Mastery Assessment', fontweight='bold', pad=20)
    
    # Timeline visualization
    plt.subplot(2, 2, 2)
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    breakthroughs = [1, 2, 3, 10, 15, 25, 40]  # Relative impact scores
    
    plt.plot(years, breakthroughs, 'ro-', linewidth=3, markersize=10)
    plt.title('Attention Revolution Timeline', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Impact')
    plt.grid(True, alpha=0.3)
    
    # Key milestones
    milestones = [
        (2014, 'First Attention'),
        (2017, 'Transformers'),
        (2018, 'BERT'),
        (2020, 'GPT-3')
    ]
    
    for year, milestone in milestones:
        plt.annotate(milestone, (year, breakthroughs[years.index(year)]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Complexity comparison
    plt.subplot(2, 2, 3)
    seq_lengths = np.array([64, 128, 256, 512, 1024])
    
    plt.loglog(seq_lengths, seq_lengths**2, 'r-', linewidth=3, label='Standard O(n²)')
    plt.loglog(seq_lengths, seq_lengths, 'g-', linewidth=3, label='Linear O(n)')
    plt.loglog(seq_lengths, seq_lengths * np.log(seq_lengths), 'b-', linewidth=3, label='Sparse O(n log n)')
    
    plt.title('Attention Complexity Scaling', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Operations (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Applications overview
    plt.subplot(2, 2, 4)
    applications = ['Translation', 'Summarization', 'QA', 'Image Captioning', 'Speech Recognition']
    scores = [0.95, 0.88, 0.92, 0.85, 0.78]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    bars = plt.bar(applications, scores, color=colors, alpha=0.7)
    plt.title('Attention Success in Applications', fontweight='bold')
    plt.ylabel('Performance Improvement')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar. 1)
        plt.imshow(pos_encoding[:20, :20].numpy(), cmap='RdBu', aspect='auto')
        plt.title('Positional Encoding Patterns', fontweight='bold')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.colorbar()
        
        plt.subplot(2, 3, 2)
        # Show how different dimensions have different frequencies
        positions = torch.arange(0, 50)
        dim_examples = [0, 1, 10, 20, 30, 40]
        
        for dim in dim_examples[:3]:
            plt.plot(positions, pos_encoding[:50, dim], label=f'Dim {dim}', linewidth=2)
        
        plt.title('Positional Encoding - Even Dimensions (sin)', fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3,1)
    
    # Calculate overall mastery score
    overall_score = sum(assessment_results.values()) / len(assessment_results)
    
    print(f"\n🏆 Overall Mastery Assessment:")
    print(f"   Score: {overall_score:.1%}")
    
    if overall_score >= 0.9:
        mastery_level = "Expert"
        next_steps = "Ready for advanced transformer architectures and research"
    elif overall_score >= 0.7:
        mastery_level = "Advanced"
        next_steps = "Continue with transformer implementation next week"
    elif overall_score >= 0.5:
        mastery_level = "Intermediate"
        next_steps = "Review weak areas, practice implementation"
    else:
        mastery_level = "Beginner"
        next_steps = "Revisit fundamental concepts and mathematical foundations"
    
    print(f"   Mastery Level: {mastery_level}")
    print(f"   Next Steps: {next_steps}")
    
    # Detailed breakdown
    print(f"\n📊 Detailed Assessment Breakdown:")
    for category, passed in assessment_results.items():
        status = "✅ PASSED" if passed else "❌ NEEDS WORK"
        category_formatted = category.replace('_', ' ').title()
        print(f"   {category_formatted}: {status}")
    
    # Week 26 preparation
    print(f"\n🚀 Week 26 Preparation Checklist:")
    week26_prep = [
        "Understand attention mechanisms deeply ✅",
        "Can implement multi-head attention from scratch ✅",
        "Familiar with positional encoding ✅",
        "Know different attention variants ✅",
        "Can interpret attention patterns ✅",
        "Understand computational trade-offs ✅",
        "Ready to build complete transformer architecture 🎯"
    ]
    
    for item in week26_prep:
        print(f"   {item}")
    
    return {
        'assessment_results': assessment_results,
        'overall_score': overall_score,
        'mastery_level': mastery_level,
        'next_steps': next_steps
    }


def attention_mechanisms_final_showcase():
    """
    Final showcase demonstrating complete attention mechanism mastery
    """
    print("\n🎭 ATTENTION MECHANISMS MASTERY SHOWCASE")
    print("=" * 60)
    
    print("🌟 Journey Completed: From RNN Bottlenecks to Attention Revolution")
    print()
    
    # Showcase 1: Complete attention pipeline
    print("🔥 Showcase 1: Complete Attention Pipeline")
    print("-" * 40)
    
    # Create a mini-transformer block
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
        
        def forward(self, x):
            # Self-attention with residual connection
            attn_out, attn_weights = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # Feed-forward with residual connection
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            return x, attn_weights
    
    # Test the complete pipeline
    d_model = 128
    num_heads = 8
    seq_len = 16
    batch_size = 2
    
    transformer_block = TransformerBlock(d_model, num_heads)
    
    # Add positional encoding
    def get_positional_encoding(seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    # Complete pipeline test
    torch.manual_seed(42)
    input_embeddings = torch.randn(batch_size, seq_len, d_model)
    pos_encoding = get_positional_encoding(seq_len, d_model)
    
    # Add positional encoding
    input_with_pos = input_embeddings + pos_encoding.unsqueeze(0)
    
    with torch.no_grad():
        output, attention_weights = transformer_block(input_with_pos)
    
    print(f"✅ Input shape: {input_embeddings.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    print(f"✅ Complete transformer block working!")
    
    # Showcase 2: Multi-modal attention
    print(f"\n🔥 Showcase 2: Cross-Modal Attention")
    print("-" * 40)
    
    # Simulate vision-language attention
    vision_features = torch.randn(batch_size, 49, d_model)  # 7x7 image patches
    text_features = torch.randn(batch_size, seq_len, d_model)  # Text tokens
    
    # Cross-attention: text queries attend to visual keys/values
    cross_attention = MultiHeadAttention(d_model, num_heads)
    
    with torch.no_grad():
        cross_output, cross_attn = cross_attention(text_features, vision_features, vision_features)
    
    print(f"✅ Text features: {text_features.shape}")
    print(f"✅ Vision features: {vision_features.shape}")
    print(f"✅ Cross-attention output: {cross_output.shape}")
    print(f"✅ Cross-attention weights: {cross_attn.shape}")
    print(f"✅ Multi-modal attention working!")
    
    # Showcase 3: Attention efficiency comparison
    print(f"\n🔥 Showcase 3: Efficiency Mastery")
    print("-" * 40)
    
    seq_lengths = [128, 256, 512]
    
    for seq_len in seq_lengths:
        # Standard attention memory
        standard_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1e6
        
        # Sparse attention memory (assuming 25% sparsity)
        sparse_memory = standard_memory * 0.25
        
        # Linear attention memory
        linear_memory = batch_size * num_heads * d_model * d_model * 4 / 1e6
        
        print(f"   Sequence Length {seq_len}:")
        print(f"     Standard: {standard_memory:.1f} MB")
        print(f"     Sparse:   {sparse_memory:.1f} MB ({sparse_memory/standard_memory:.1%})")
        print(f"     Linear:   {linear_memory:.1f} MB ({linear_memory/standard_memory:.1%})")
    
    # Final mastery demonstration
    print(f"\n🎯 Mastery Demonstrations Completed:")
    demonstrations = [
        "✅ Historical understanding: RNN bottlenecks → Attention solutions",
        "✅ Mathematical mastery: Query-Key-Value, scaling, positional encoding",
        "✅ Implementation skills: Multiple attention variants from scratch",
        "✅ Multi-head attention: Parallel processing, head specialization",
        "✅ Advanced variants: Sparse, linear, cross-modal attention",
        "✅ Efficiency analysis: Complexity trade-offs and optimizations",
        "✅ Interpretation skills: Attention pattern analysis and visualization",
        "✅ Transformer readiness: Complete building blocks implemented"
    ]
    
    for demo in demonstrations:
        print(f"   {demo}")
    
    print(f"\n🚀 Ready for Week 26: Complete Transformer Architecture!")
    
    return {
        'transformer_block': transformer_block,
        'attention_outputs': {
            'self_attention': (output, attention_weights),
            'cross_attention': (cross_output, cross_attn)
        },
        'efficiency_analysis': {
            'seq_lengths': seq_lengths,
            'memory_usage': {
                'standard': [batch_size * num_heads * sl * sl * 4 / 1e6 for sl in seq_lengths],
                'sparse': [batch_size * num_heads * sl * sl * 4 / 1e6 * 0.25 for sl in seq_lengths],
                'linear': [batch_size * num_heads * d_model * d_model * 4 / 1e6 for _ in seq_lengths]
            }
        }
    }


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

def main():
    """
    Main execution flow for Week 25: Attention Mechanisms
    """
    print("🎯 Starting Week 25: Attention Mechanisms Journey")
    print("=" * 60)
    
    # Section 1: Historical Context
    print("\n🏛️  SECTION 1: HISTORICAL CONTEXT AND MOTIVATION")
    rnn_demo = RNNBottleneckDemo()
    bottleneck_results = rnn_demo.demonstrate_information_bottleneck()
    alignment_results = rnn_demo.demonstrate_alignment_problem()
    timeline = historical_attention_timeline()
    
    # Section 2: Mathematical Foundations
    print("\n📊 SECTION 2: MATHEMATICAL FOUNDATIONS")
    foundations = AttentionMathematicalFoundations()
    qkv_results = foundations.explain_query_key_value_paradigm()
    scaling_results = foundations.derive_scaled_dot_product_attention()
    pos_results = foundations.explain_positional_encoding()
    
    # Section 3: Core Implementations
    print("\n🧮 SECTION 3: CORE ATTENTION IMPLEMENTATIONS")
    basic_attention = BasicAttentionMechanisms()
    comparison_results = basic_attention.compare_attention_mechanisms()
    
    # Section 4: Multi-Head Attention
    print("\n🧠 SECTION 4: MULTI-HEAD ATTENTION")
    mha_results = demonstrate_multi_head_attention()
    
    # Section 5: Advanced Variants
    print("\n⚡ SECTION 5: ADVANCED ATTENTION VARIANTS")
    sparse_results = demonstrate_sparse_attention()
    efficiency_results = compare_attention_efficiency()
    
    # Section 6: Visualization & Interpretation
    print("\n🔍 SECTION 6: ATTENTION VISUALIZATION & INTERPRETATION")
    interpretation_results = attention_interpretation_deep_dive()
    
    # Section 7: Comprehensive Assessment
    print("\n🎓 SECTION 7: MASTERY ASSESSMENT")
    assessment_results = comprehensive_attention_mastery_assessment()
    
    # Final Showcase
    print("\n🎭 FINAL SHOWCASE")
    showcase_results = attention_mechanisms_final_showcase()
    
    # Week Summary
    print("\n🎉 WEEK 25 COMPLETE: ATTENTION MECHANISMS MASTERED!")
    print("=" * 60)
    
    key_achievements = [
        "🧠 Mastered the mathematical foundations of attention mechanisms",
        "⚡ Implemented multiple attention variants from scratch",
        "🔍 Developed skills in attention pattern interpretation",
        "🚀 Built complete transformer building blocks",
        "📊 Understood computational trade-offs and efficiency",
        "🎯 Ready for advanced transformer architectures",
        "🌟 Grasped the revolutionary impact on modern AI"
    ]
    
    print("🏆 Key Achievements:")
    for achievement in key_achievements:
        print(f"   {achievement}")
    
    print(f"\n🔮 Next Week Preview: Complete Transformer Architecture")
    print("   Building on attention mastery to create the full transformer")
    print("   Encoder-decoder architecture, training dynamics, applications")
    print("   Path to understanding BERT, GPT, and modern language models")
    
    print(f"\n💡 Key Insights Gained:")
    insights = [
        "Attention solves the fundamental bottleneck problem in sequence modeling",
        "Query-Key-Value paradigm enables flexible information routing",
        "Multi-head attention allows parallel processing of different relationships",
        "Positional encoding is crucial for sequence understanding",
        "Sparse and linear attention enable scaling to long sequences",
        "Attention patterns are interpretable but not always explanatory",
        "The transformer revolution started with attention mechanisms"
    ]
    
    for insight in insights:
        print(f"   • {insight}")
    
    print(f"\n🧭 The Path Forward:")
    print("   Week 25: ✅ Attention Mechanisms (COMPLETED)")
    print("   Week 26: 🎯 Complete Transformer Architecture")
    print("   Week 27: 📝 Natural Language Processing Applications")
    print("   Week 28: 👁️ Computer Vision with Transformers")
    print("   Week 29: 🏗️ Advanced Architectures and Scaling")
    
    return {
        'bottleneck_demo': bottleneck_results,
        'mathematical_foundations': {
            'qkv': qkv_results,
            'scaling': scaling_results,
            'positional': pos_results
        },
        'implementations': {
            'basic': comparison_results,
            'multi_head': mha_results,
            'sparse': sparse_results,
            'efficiency': efficiency_results
        },
        'interpretation': interpretation_results,
        'assessment': assessment_results,
        'showcase': showcase_results
    }


# Execute the main function when script is run
if __name__ == "__main__":
    print("🚀 Neural Odyssey Week 25: Attention Mechanisms")
    print("   The Foundation of Modern AI Revolution")
    print("   From RNN Bottlenecks to Transformer Breakthroughs")
    print()
    
    # Run the complete attention mechanisms journey
    results = main()
    
    print("\n🎊 Congratulations! You have mastered attention mechanisms!")
    print("   You now understand the core technology behind:")
    print("   • ChatGPT and GPT models")
    print("   • BERT and language understanding")
    print("   • Vision transformers")
    print("   • Multimodal AI systems")
    print("   • Modern neural machine translation")
    print("   • And virtually every state-of-the-art AI system!")
    
    print(f"\n📚 Ready for the next chapter: Complete Transformer Architecture")
    print("   Your attention mastery is the foundation for everything that follows!")
 