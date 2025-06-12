
"""
Neural Odyssey - Week 10: Historical Context of AI and ML
Phase 1: Mathematical Foundations

Understanding the Journey: From Logic to Learning

This week explores the rich history of artificial intelligence and machine learning,
implementing key historical algorithms and understanding how we arrived at modern deep learning.
You'll build classical AI systems and see how mathematical breakthroughs shaped the field.

Learning Objectives:
- Understand the evolution from symbolic AI to statistical learning
- Implement historical algorithms (perceptron, k-means, decision trees)
- Explore key breakthroughs and their mathematical foundations
- Connect historical context to modern deep learning
- Appreciate the interdisciplinary nature of AI development

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import random
import math
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd


# ==========================================
# HISTORICAL TIMELINE AND MILESTONES
# ==========================================

class AIHistoryTimeline:
    """
    Interactive timeline of AI history with key milestones and implementations
    """
    
    def __init__(self):
        self.milestones = self._create_timeline()
        self.current_era = None
    
    def _create_timeline(self) -> Dict[str, Dict]:
        """Create comprehensive AI history timeline"""
        return {
            "1943": {
                "era": "Birth of AI",
                "milestone": "McCulloch-Pitts Neuron",
                "description": "First mathematical model of a neuron",
                "significance": "Established neurons as logical units",
                "key_people": ["Warren McCulloch", "Walter Pitts"],
                "mathematical_contribution": "Boolean logic in neural networks",
                "implementation": "mcculloch_pitts_neuron"
            },
            "1949": {
                "era": "Learning Theory",
                "milestone": "Hebbian Learning",
                "description": "First learning rule: 'Neurons that fire together, wire together'",
                "significance": "Foundation of synaptic plasticity",
                "key_people": ["Donald Hebb"],
                "mathematical_contribution": "Δw = η * x_i * x_j",
                "implementation": "hebbian_learning"
            },
            "1950": {
                "era": "AI Philosophy",
                "milestone": "Turing Test",
                "description": "Operational definition of machine intelligence",
                "significance": "Established AI as a scientific discipline",
                "key_people": ["Alan Turing"],
                "mathematical_contribution": "Computational theory of mind",
                "implementation": "turing_test_simulator"
            },
            "1957": {
                "era": "First Learning Algorithm",
                "milestone": "Perceptron Algorithm",
                "description": "First algorithm guaranteed to find linear separator",
                "significance": "Proved machines could learn from data",
                "key_people": ["Frank Rosenblatt"],
                "mathematical_contribution": "Linear classification with guaranteed convergence",
                "implementation": "perceptron_algorithm"
            },
            "1969": {
                "era": "AI Winter Begins",
                "milestone": "Perceptrons Book",
                "description": "Showed limitations of linear classifiers",
                "significance": "Led to first AI winter",
                "key_people": ["Marvin Minsky", "Seymour Papert"],
                "mathematical_contribution": "XOR problem and linear separability",
                "implementation": "xor_limitation_demo"
            },
            "1975": {
                "era": "Unsupervised Learning",
                "milestone": "K-Means Algorithm",
                "description": "Clustering algorithm for data exploration",
                "significance": "Enabled pattern discovery without labels",
                "key_people": ["Stuart Lloyd"],
                "mathematical_contribution": "Iterative centroid optimization",
                "implementation": "kmeans_algorithm"
            },
            "1986": {
                "era": "Neural Renaissance",
                "milestone": "Backpropagation Popularized",
                "description": "Multilayer networks could learn non-linear functions",
                "significance": "Overcame XOR limitation, revived neural networks",
                "key_people": ["Rumelhart", "Hinton", "Williams"],
                "mathematical_contribution": "Chain rule for gradient computation",
                "implementation": "backprop_xor_solution"
            },
            "1997": {
                "era": "AI Triumphant",
                "milestone": "Deep Blue Defeats Kasparov",
                "description": "First computer to beat world chess champion",
                "significance": "Demonstrated AI superiority in complex domains",
                "key_people": ["IBM Deep Blue Team"],
                "mathematical_contribution": "Advanced search algorithms and evaluation functions",
                "implementation": "minimax_chess_demo"
            },
            "2006": {
                "era": "Deep Learning Revolution",
                "milestone": "Deep Belief Networks",
                "description": "Showed how to train deep networks effectively",
                "significance": "Launched modern deep learning era",
                "key_people": ["Geoffrey Hinton"],
                "mathematical_contribution": "Layer-wise pretraining and RBMs",
                "implementation": "deep_belief_network"
            },
            "2012": {
                "era": "Computer Vision Breakthrough",
                "milestone": "AlexNet ImageNet Victory",
                "description": "CNN dramatically outperformed traditional methods",
                "significance": "Proved deep learning's practical superiority",
                "key_people": ["Alex Krizhevsky", "Geoffrey Hinton"],
                "mathematical_contribution": "Convolutional neural networks with GPU training",
                "implementation": "simple_cnn_demo"
            }
        }
    
    def display_timeline(self):
        """Display interactive timeline visualization"""
        years = sorted(self.milestones.keys())
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create timeline
        y_positions = range(len(years))
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, year in enumerate(years):
            milestone = self.milestones[year]
            
            # Plot point
            ax.scatter(int(year), i, s=200, c=colors[i % len(colors)], alpha=0.7, zorder=3)
            
            # Add milestone name
            ax.annotate(milestone['milestone'], 
                       xy=(int(year), i), 
                       xytext=(20, 0), 
                       textcoords='offset points',
                       fontsize=10, 
                       fontweight='bold',
                       ha='left')
            
            # Add era and key people
            ax.annotate(f"{milestone['era']}\n{', '.join(milestone['key_people'])}", 
                       xy=(int(year), i), 
                       xytext=(20, -15), 
                       textcoords='offset points',
                       fontsize=8, 
                       alpha=0.7,
                       ha='left')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Historical Milestones', fontsize=12, fontweight='bold')
        ax.set_title('AI/ML Historical Timeline: From Logic to Learning', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, len(years) - 0.5)
        
        # Remove y-axis ticks
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print("\n🕒 AI/ML Historical Timeline")
        print("=" * 60)
        for year in years:
            milestone = self.milestones[year]
            print(f"\n📅 {year}: {milestone['milestone']}")
            print(f"   Era: {milestone['era']}")
            print(f"   Description: {milestone['description']}")
            print(f"   Key People: {', '.join(milestone['key_people'])}")
            print(f"   Mathematical Contribution: {milestone['mathematical_contribution']}")
    
    def get_milestone(self, year: str) -> Dict:
        """Get specific milestone information"""
        return self.milestones.get(year, {})


# ==========================================
# HISTORICAL ALGORITHM IMPLEMENTATIONS
# ==========================================

class McCullochPittsNeuron:
    """
    Implementation of the first artificial neuron (1943)
    """
    
    def __init__(self, weights: List[float], threshold: float):
        """
        Initialize McCulloch-Pitts neuron
        
        Args:
            weights: Connection weights
            threshold: Activation threshold
        """
        self.weights = np.array(weights)
        self.threshold = threshold
    
    def activate(self, inputs: List[float]) -> int:
        """
        Compute neuron activation using step function
        
        Args:
            inputs: Input values
            
        Returns:
            Binary output (0 or 1)
        """
        weighted_sum = np.dot(self.weights, inputs)
        return 1 if weighted_sum >= self.threshold else 0
    
    def demonstrate_logic_gates(self):
        """Demonstrate that M-P neurons can implement logic gates"""
        print("🧠 McCulloch-Pitts Neuron: Logic Gate Implementation")
        print("=" * 55)
        
        # AND gate: both inputs must be 1
        and_neuron = McCullochPittsNeuron([1, 1], threshold=2)
        print("\n🔗 AND Gate (threshold=2, weights=[1,1]):")
        for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            output = and_neuron.activate([x1, x2])
            print(f"   {x1} AND {x2} = {output}")
        
        # OR gate: at least one input must be 1
        or_neuron = McCullochPittsNeuron([1, 1], threshold=1)
        print("\n🔗 OR Gate (threshold=1, weights=[1,1]):")
        for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            output = or_neuron.activate([x1, x2])
            print(f"   {x1} OR {x2} = {output}")
        
        # NOT gate: invert input
        not_neuron = McCullochPittsNeuron([-1], threshold=0)
        print("\n🔗 NOT Gate (threshold=0, weights=[-1]):")
        for x in [0, 1]:
            output = not_neuron.activate([x])
            print(f"   NOT {x} = {output}")
        
        print("\n💡 Key Insight: Any Boolean function can be computed by a network of M-P neurons!")


class HebbianLearning:
    """
    Implementation of Hebbian learning rule (1949)
    """
    
    def __init__(self, input_size: int):
        """
        Initialize Hebbian learning system
        
        Args:
            input_size: Number of input neurons
        """
        self.weights = np.zeros((input_size, input_size))
        self.learning_rate = 0.1
    
    def update_weights(self, pattern: np.ndarray):
        """
        Update weights using Hebbian rule: Δw_ij = η * x_i * x_j
        
        Args:
            pattern: Input pattern
        """
        # Hebbian learning: strengthen connections between co-active neurons
        outer_product = np.outer(pattern, pattern)
        self.weights += self.learning_rate * outer_product
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, pattern: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Recall pattern using Hopfield-like dynamics
        
        Args:
            pattern: Input pattern (potentially noisy)
            iterations: Number of update iterations
            
        Returns:
            Recalled pattern
        """
        current_pattern = pattern.copy()
        
        for _ in range(iterations):
            # Update each neuron based on weighted inputs
            new_pattern = np.sign(np.dot(self.weights, current_pattern))
            new_pattern[new_pattern == 0] = 1  # Handle zero case
            current_pattern = new_pattern
        
        return current_pattern
    
    def demonstrate_pattern_learning(self):
        """Demonstrate Hebbian learning on simple patterns"""
        print("🧠 Hebbian Learning: Pattern Association")
        print("=" * 45)
        
        # Define simple patterns
        patterns = [
            np.array([1, 1, -1, -1]),   # Pattern A
            np.array([1, -1, 1, -1]),   # Pattern B
            np.array([-1, 1, 1, -1])    # Pattern C
        ]
        
        pattern_names = ['A', 'B', 'C']
        
        print("\nOriginal Patterns:")
        for i, pattern in enumerate(patterns):
            print(f"   Pattern {pattern_names[i]}: {pattern}")
        
        # Learn patterns
        print(f"\n🎯 Learning patterns with Hebbian rule...")
        for pattern in patterns:
            self.update_weights(pattern)
        
        print(f"Weight matrix after learning:")
        print(self.weights)
        
        # Test recall with noisy patterns
        print(f"\n🔍 Testing recall with noisy patterns:")
        for i, original in enumerate(patterns):
            # Add noise by flipping one bit
            noisy = original.copy()
            flip_idx = np.random.randint(0, len(noisy))
            noisy[flip_idx] *= -1
            
            recalled = self.recall(noisy)
            
            print(f"\n   Pattern {pattern_names[i]}:")
            print(f"     Original: {original}")
            print(f"     Noisy:    {noisy}")
            print(f"     Recalled: {recalled}")
            print(f"     Match: {'✓' if np.array_equal(original, recalled) else '✗'}")
        
        print("\n💡 Hebbian learning enables associative memory!")


class PerceptronAlgorithm:
    """
    Implementation of Rosenblatt's Perceptron (1957)
    """
    
    def __init__(self, input_size: int):
        """
        Initialize perceptron
        
        Args:
            input_size: Number of input features
        """
        self.weights = np.random.randn(input_size + 1) * 0.1  # +1 for bias
        self.learning_rate = 0.1
        self.training_history = []
    
    def predict(self, x: np.ndarray) -> int:
        """
        Make prediction using step function
        
        Args:
            x: Input features
            
        Returns:
            Prediction (0 or 1)
        """
        # Add bias term
        x_with_bias = np.concatenate([[1], x])
        activation = np.dot(self.weights, x_with_bias)
        return 1 if activation >= 0 else 0
    
    def train(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 100) -> bool:
        """
        Train perceptron using the perceptron learning rule
        
        Args:
            X: Training features
            y: Training labels
            max_epochs: Maximum number of epochs
            
        Returns:
            True if converged, False otherwise
        """
        self.training_history = []
        
        for epoch in range(max_epochs):
            errors = 0
            
            for i in range(len(X)):
                # Make prediction
                prediction = self.predict(X[i])
                error = y[i] - prediction
                
                if error != 0:
                    errors += 1
                    # Update weights: w = w + η * (target - prediction) * x
                    x_with_bias = np.concatenate([[1], X[i]])
                    self.weights += self.learning_rate * error * x_with_bias
            
            accuracy = 1 - (errors / len(X))
            self.training_history.append(accuracy)
            
            # Check for convergence
            if errors == 0:
                print(f"✅ Perceptron converged after {epoch + 1} epochs!")
                return True
        
        print(f"❌ Perceptron did not converge after {max_epochs} epochs")
        return False
    
    def demonstrate_linear_separation(self):
        """Demonstrate perceptron on linearly separable data"""
        print("🎯 Perceptron Algorithm: Linear Classification")
        print("=" * 48)
        
        # Generate linearly separable data
        np.random.seed(42)
        n_samples = 100
        
        # Class 0: points around (1, 1)
        X0 = np.random.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], n_samples // 2)
        y0 = np.zeros(n_samples // 2)
        
        # Class 1: points around (3, 3)
        X1 = np.random.multivariate_normal([3, 3], [[0.3, 0], [0, 0.3]], n_samples // 2)
        y1 = np.ones(n_samples // 2)
        
        # Combine data
        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])
        
        print(f"Generated {n_samples} linearly separable samples")
        
        # Train perceptron
        print("\n🚀 Training perceptron...")
        converged = self.train(X, y)
        
        # Plot results
        self._plot_decision_boundary(X, y, converged)
        
        # Plot training history
        plt.figure(figsize=(8, 4))
        plt.plot(self.training_history)
        plt.title('Perceptron Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
        
        return converged
    
    def demonstrate_xor_limitation(self):
        """Demonstrate perceptron's limitation on XOR problem"""
        print("\n❌ Perceptron Limitation: XOR Problem")
        print("=" * 40)
        
        # XOR dataset
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])  # XOR truth table
        
        print("XOR Truth Table:")
        for i in range(len(X_xor)):
            print(f"   {X_xor[i]} -> {y_xor[i]}")
        
        # Try to train perceptron on XOR
        print(f"\n🚀 Attempting to train perceptron on XOR...")
        perceptron_xor = PerceptronAlgorithm(2)
        converged = perceptron_xor.train(X_xor, y_xor, max_epochs=1000)
        
        # Show final predictions
        print(f"\nFinal predictions:")
        for i in range(len(X_xor)):
            pred = perceptron_xor.predict(X_xor[i])
            print(f"   {X_xor[i]} -> {pred} (target: {y_xor[i]})")
        
        # Visualize the limitation
        self._plot_xor_limitation(X_xor, y_xor, perceptron_xor)
        
        print("\n💡 Key Insight: Single perceptron cannot solve XOR!")
        print("   This limitation led to the first AI winter (1970s)")
        print("   Solution: Multi-layer networks with non-linear activations")
        
        return converged
    
    def _plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, converged: bool):
        """Plot perceptron decision boundary"""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        colors = ['red', 'blue']
        for class_val in [0, 1]:
            mask = y == class_val
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                       label=f'Class {class_val}', alpha=0.7, s=50)
        
        if converged:
            # Plot decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            
            # Decision boundary: w0 + w1*x1 + w2*x2 = 0
            # Solving for x2: x2 = -(w0 + w1*x1) / w2
            if abs(self.weights[2]) > 1e-6:  # Avoid division by zero
                x_boundary = np.array([x_min, x_max])
                y_boundary = -(self.weights[0] + self.weights[1] * x_boundary) / self.weights[2]
                plt.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Perceptron Classification {"(Converged)" if converged else "(Not Converged)"}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _plot_xor_limitation(self, X: np.ndarray, y: np.ndarray, perceptron):
        """Visualize XOR limitation"""
        plt.figure(figsize=(8, 6))
        
        # Plot XOR data points
        colors = ['red', 'blue']
        for i in range(len(X)):
            plt.scatter(X[i, 0], X[i, 1], c=colors[y[i]], s=200, 
                       edgecolors='black', linewidth=2)
            plt.annotate(f'({X[i, 0]}, {X[i, 1]}) -> {y[i]}', 
                        (X[i, 0], X[i, 1]), xytext=(10, 10), 
                        textcoords='offset points', fontsize=10)
        
        # Try to plot decision boundary (will be inadequate)
        x_range = np.linspace(-0.5, 1.5, 100)
        if abs(perceptron.weights[2]) > 1e-6:
            y_boundary = -(perceptron.weights[0] + perceptron.weights[1] * x_range) / perceptron.weights[2]
            plt.plot(x_range, y_boundary, 'k--', linewidth=2, 
                    label='Attempted Decision Boundary', alpha=0.7)
        
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('XOR Problem: No Linear Solution Exists')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class KMeansHistorical:
    """
    Implementation of Lloyd's K-Means Algorithm (1975)
    """
    
    def __init__(self, k: int, max_iters: int = 100):
        """
        Initialize K-Means algorithm
        
        Args:
            k: Number of clusters
            max_iters: Maximum number of iterations
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.history = []
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-means to data
        
        Args:
            X: Data points
            
        Returns:
            Cluster labels
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        self.history = [self.centroids.copy()]
        
        for iteration in range(self.max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"✅ K-means converged after {iteration + 1} iterations")
                break
            
            self.centroids = new_centroids
            self.history.append(self.centroids.copy())
        
        return self.labels
    
    def demonstrate_clustering(self):
        """Demonstrate K-means clustering on synthetic data"""
        print("🎯 K-Means Algorithm: Unsupervised Learning")
        print("=" * 45)
        
        # Generate synthetic clustered data
        np.random.seed(42)
        
        # Create three clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
        cluster2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 0.5]], 50)
        cluster3 = np.random.multivariate_normal([2, 6], [[0.5, 0], [0, 0.5]], 50)
        
        X = np.vstack([cluster1, cluster2, cluster3])
        
        print(f"Generated {len(X)} data points in 3 natural clusters")
        
        # Apply K-means
        print(f"\n🚀 Applying K-means with k=3...")
        labels = self.fit(X)
        
        # Visualize results
        self._plot_clustering_result(X, labels)
        self._animate_convergence(X)
        
        # Calculate within-cluster sum of squares
        wcss = self._calculate_wcss(X, labels)
        print(f"\nWithin-Cluster Sum of Squares: {wcss:.2f}")
        
        return labels
    
    def demonstrate_elbow_method(self):
        """Demonstrate elbow method for choosing k"""
        print("\n📊 Elbow Method: Choosing Optimal k")
        print("=" * 40)
        
        # Generate data
        np.random.seed(42)
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
        cluster2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 0.5]], 50)
        cluster3 = np.random.multivariate_normal([2, 6], [[0.5, 0], [0, 0.5]], 50)
        X = np.vstack([cluster1, cluster2, cluster3])
        
        # Test different values of k
        k_values = range(1, 8)
        wcss_values = []
        
        for k in k_values:
            kmeans = KMeansHistorical(k)
            labels = kmeans.fit(X)
            wcss = kmeans._calculate_wcss(X, labels)
            wcss_values.append(wcss)
            print(f"k={k}: WCSS = {wcss:.2f}")
        
        # Plot elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)
        
        # Highlight the elbow
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Elbow at k=3')
        plt.legend()
        plt.show()
        
        print("\n💡 The elbow at k=3 suggests 3 is the optimal number of clusters!")
    
    def _calculate_wcss(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares"""
        wcss = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[i])**2)
        return wcss
    
    def _plot_clustering_result(self, X: np.ndarray, labels: np.ndarray):
        """Plot clustering results"""
        plt.figure(figsize=(10, 6))
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Plot data points
        for i in range(self.k):
            cluster_points = X[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=50)
        
        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _animate_convergence(self, X: np.ndarray):
        """Animate K-means convergence"""
        if len(self.history) <= 1:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Plot final clustering
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[i], alpha=0.7, s=50)
        
        # Plot centroid history
        for i in range(self.k):
            centroid_history = np.array([h[i] for h in self.history])
            ax.plot(centroid_history[:, 0], centroid_history[:, 1], 
                   'o-', color=colors[i], linewidth=2, markersize=8,
                   label=f'Centroid {i+1} Path')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('K-Means Convergence Animation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()


class DecisionTreeHistorical:
    """
    Implementation of basic decision tree (1970s-1980s)
    """
    
    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.prediction = None
            self.is_leaf = False
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        """
        Initialize decision tree
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of labels"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float) -> float:
        """Calculate information gain for a split"""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Weighted entropy of children
        n = len(y)
        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask) / n) * left_entropy + (np.sum(right_mask) / n) * right_entropy
        
        return parent_entropy - weighted_entropy
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find best feature and threshold for splitting"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build decision tree"""
        node = self.Node()
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            node.is_leaf = True
            node.prediction = np.bincount(y.astype(int)).argmax()
            return node
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            node.is_leaf = True
            node.prediction = np.bincount(y.astype(int)).argmax()
            return node
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train decision tree"""
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """Predict single sample"""
        if node.is_leaf:
            return node.prediction
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def demonstrate_decision_tree(self):
        """Demonstrate decision tree on Iris-like dataset"""
        print("🌳 Decision Tree: Symbolic AI Approach")
        print("=" * 40)
        
        # Generate synthetic iris-like data
        np.random.seed(42)
        
        # Class 0: Small flowers
        X0 = np.random.multivariate_normal([1.5, 0.5], [[0.1, 0], [0, 0.05]], 50)
        y0 = np.zeros(50)
        
        # Class 1: Medium flowers
        X1 = np.random.multivariate_normal([3.0, 1.5], [[0.2, 0], [0, 0.1]], 50)
        y1 = np.ones(50)
        
        # Class 2: Large flowers
        X2 = np.random.multivariate_normal([4.5, 2.5], [[0.2, 0], [0, 0.1]], 50)
        y2 = np.full(50, 2)
        
        X = np.vstack([X0, X1, X2])
        y = np.concatenate([y0, y1, y2])
        
        feature_names = ['Petal Length', 'Petal Width']
        class_names = ['Small', 'Medium', 'Large']
        
        print(f"Generated {len(X)} samples with 2 features, 3 classes")
        
        # Train decision tree
        print(f"\n🚀 Training decision tree...")
        self.fit(X, y)
        
        # Make predictions
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Training accuracy: {accuracy:.3f}")
        
        # Visualize results
        self._plot_decision_tree_results(X, y, predictions, feature_names, class_names)
        self._print_tree_rules(self.root, feature_names, class_names)
        
        return accuracy
    
    def _plot_decision_tree_results(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray, 
                                   feature_names: List[str], class_names: List[str]):
        """Plot decision tree classification results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['red', 'blue', 'green']
        
        # Plot actual classes
        for class_val in np.unique(y):
            mask = y == class_val
            ax1.scatter(X[mask, 0], X[mask, 1], c=colors[int(class_val)], 
                       label=f'{class_names[int(class_val)]}', alpha=0.7, s=50)
        
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_title('Actual Classes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predicted classes
        for class_val in np.unique(predictions):
            mask = predictions == class_val
            ax2.scatter(X[mask, 0], X[mask, 1], c=colors[int(class_val)], 
                       label=f'{class_names[int(class_val)]}', alpha=0.7, s=50)
        
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_title('Decision Tree Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _print_tree_rules(self, node: Node, feature_names: List[str], class_names: List[str], depth: int = 0):
        """Print decision tree rules in human-readable format"""
        if depth == 0:
            print(f"\n📋 Decision Tree Rules:")
            print("=" * 30)
        
        indent = "  " * depth
        
        if node.is_leaf:
            print(f"{indent}→ Predict: {class_names[node.prediction]}")
        else:
            print(f"{indent}If {feature_names[node.feature]} <= {node.threshold:.2f}:")
            self._print_tree_rules(node.left, feature_names, class_names, depth + 1)
            print(f"{indent}Else:")
            self._print_tree_rules(node.right, feature_names, class_names, depth + 1)


class BackpropXORSolution:
    """
    Demonstrate how backpropagation solved the XOR limitation (1986)
    """
    
    def __init__(self):
        # Simple 2-2-1 network for XOR
        self.W1 = np.random.randn(2, 2) * 0.5  # Input to hidden
        self.b1 = np.random.randn(1, 2) * 0.5  # Hidden biases
        self.W2 = np.random.randn(2, 1) * 0.5  # Hidden to output
        self.b2 = np.random.randn(1, 1) * 0.5  # Output bias
        self.learning_rate = 1.0
        self.training_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """Backward pass"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        """Train the network"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y)**2)
            self.training_history.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return self.training_history
    
    def demonstrate_xor_solution(self):
        """Demonstrate how multilayer network solves XOR"""
        print("🧠 Backpropagation: Solving the XOR Problem")
        print("=" * 45)
        
        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        print("XOR Truth Table:")
        for i in range(len(X)):
            print(f"   {X[i]} -> {y[i][0]}")
        
        print(f"\n🚀 Training multilayer network...")
        history = self.train(X, y, epochs=2000)
        
        # Test final predictions
        final_predictions = self.forward(X)
        print(f"\nFinal Predictions:")
        for i in range(len(X)):
            pred = final_predictions[i][0]
            target = y[i][0]
            print(f"   {X[i]} -> {pred:.4f} (target: {target})")
        
        # Calculate accuracy
        binary_pred = (final_predictions > 0.5).astype(int)
        accuracy = np.mean(binary_pred == y)
        print(f"\nAccuracy: {accuracy:.3f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history)
        plt.title('XOR Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.yscale('log')
        plt.grid(True)
        
        # Visualize learned representations
        plt.subplot(1, 2, 2)
        hidden_reps = self.sigmoid(np.dot(X, self.W1) + self.b1)
        
        colors = ['red', 'blue']
        for i in range(len(X)):
            color = colors[y[i][0]]
            plt.scatter(hidden_reps[i, 0], hidden_reps[i, 1], c=color, s=200, alpha=0.7)
            plt.annotate(f'{X[i]}→{y[i][0]}', (hidden_reps[i, 0], hidden_reps[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Hidden Layer Representations')
        plt.xlabel('Hidden Unit 1')
        plt.ylabel('Hidden Unit 2')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Key Insight: Hidden layer creates linearly separable representations!")
        print("   This breakthrough ended the first AI winter and launched the neural renaissance.")
        
        return accuracy


# ==========================================
# HISTORICAL PARADIGM COMPARISONS
# ==========================================

class AIParadigmComparison:
    """
    Compare different AI paradigms through history
    """
    
    def __init__(self):
        self.paradigms = {
            "symbolic": {
                "period": "1950s-1980s",
                "approach": "Logic-based reasoning",
                "strengths": ["Interpretable", "Exact reasoning", "Knowledge representation"],
                "weaknesses": ["Brittle", "Hand-crafted rules", "Poor with uncertainty"],
                "examples": ["Expert systems", "Logic programming", "Rule-based AI"]
            },
            "connectionist": {
                "period": "1980s-1990s, 2010s-present",
                "approach": "Neural networks and learning",
                "strengths": ["Learning from data", "Pattern recognition", "Robust to noise"],
                "weaknesses": ["Black box", "Requires large data", "Computational intensive"],
                "examples": ["Perceptrons", "Backpropagation", "Deep learning"]
            },
            "statistical": {
                "period": "1990s-2010s",
                "approach": "Probabilistic and statistical methods",
                "strengths": ["Handles uncertainty", "Principled inference", "Well-founded theory"],
                "weaknesses": ["Model assumptions", "Computational complexity", "Feature engineering"],
                "examples": ["Bayesian networks", "SVM", "Random forests"]
            }
        }
    
    def demonstrate_paradigm_comparison(self):
        """Demonstrate different AI paradigms on the same problem"""
        print("🔄 AI Paradigm Evolution: Three Approaches")
        print("=" * 50)
        
        # Simple classification problem: predict if person gets loan
        # Features: [income, credit_score, age]
        # Target: loan_approved (0/1)
        
        np.random.seed(42)
        n_samples = 200
        
        # Generate synthetic loan data
        income = np.random.normal(50000, 20000, n_samples)
        credit_score = np.random.normal(650, 100, n_samples)
        age = np.random.normal(35, 10, n_samples)
        
        # Simple rule: approve if income > 40k AND credit_score > 600
        loan_approved = ((income > 40000) & (credit_score > 600)).astype(int)
        
        X = np.column_stack([income, credit_score, age])
        y = loan_approved
        
        print(f"Loan Approval Dataset: {n_samples} samples, 3 features")
        print(f"Approval rate: {np.mean(y):.2%}")
        
        # 1. Symbolic AI Approach (Rule-based)
        print(f"\n📋 1. SYMBOLIC AI: Rule-Based System")
        print("-" * 40)
        symbolic_predictions = self._symbolic_approach(X, y)
        
        # 2. Statistical AI Approach (Logistic Regression)
        print(f"\n📊 2. STATISTICAL AI: Logistic Regression")
        print("-" * 40)
        statistical_predictions = self._statistical_approach(X, y)
        
        # 3. Connectionist AI Approach (Neural Network)
        print(f"\n🧠 3. CONNECTIONIST AI: Neural Network")
        print("-" * 40)
        connectionist_predictions = self._connectionist_approach(X, y)
        
        # Compare results
        self._compare_paradigm_results(y, symbolic_predictions, statistical_predictions, connectionist_predictions)
        
        return {
            'symbolic': symbolic_predictions,
            'statistical': statistical_predictions,
            'connectionist': connectionist_predictions
        }
    
    def _symbolic_approach(self, X, y):
        """Implement rule-based symbolic AI"""
        print("Rules:")
        print("  IF income > 45000 AND credit_score > 620 THEN approve")
        print("  ELSE reject")
        
        income, credit_score, age = X[:, 0], X[:, 1], X[:, 2]
        predictions = ((income > 45000) & (credit_score > 620)).astype(int)
        
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy:.3f}")
        
        return predictions
    
    def _statistical_approach(self, X, y):
        """Implement statistical approach (simplified logistic regression)"""
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Add bias term
        X_bias = np.column_stack([np.ones(len(X)), X_norm])
        
        # Initialize weights
        weights = np.random.randn(X_bias.shape[1]) * 0.01
        learning_rate = 0.01
        
        # Gradient descent
        for epoch in range(1000):
            # Sigmoid predictions
            z = np.dot(X_bias, weights)
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
            # Gradient
            gradient = np.dot(X_bias.T, (predictions - y)) / len(y)
            weights -= learning_rate * gradient
        
        final_predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(final_predictions == y)
        print(f"Learned weights: {weights}")
        print(f"Accuracy: {accuracy:.3f}")
        
        return final_predictions
    
    def _connectionist_approach(self, X, y):
        """Implement neural network approach"""
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Simple 3-5-1 network
        W1 = np.random.randn(3, 5) * 0.5
        b1 = np.random.randn(1, 5) * 0.5
        W2 = np.random.randn(5, 1) * 0.5
        b2 = np.random.randn(1, 1) * 0.5
        
        learning_rate = 0.1
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        # Training
        for epoch in range(1000):
            # Forward pass
            z1 = np.dot(X_norm, W1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, W2) + b2
            a2 = sigmoid(z2)
            
            # Backward pass
            m = len(X)
            dz2 = a2.reshape(-1, 1) - y.reshape(-1, 1)
            dW2 = (1/m) * np.dot(a1.T, dz2)
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, W2.T)
            dz1 = da1 * a1 * (1 - a1)
            dW1 = (1/m) * np.dot(X_norm.T, dz1)
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
        
        # Final predictions
        z1 = np.dot(X_norm, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        predictions = (a2.flatten() > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"Hidden layer weights shape: {W1.shape}")
        print(f"Output layer weights shape: {W2.shape}")
        print(f"Accuracy: {accuracy:.3f}")
        
        return predictions
    
    def _compare_paradigm_results(self, y_true, symbolic, statistical, connectionist):
        """Compare results from different paradigms"""
        print(f"\n📊 Paradigm Comparison Results")
        print("=" * 35)
        
        approaches = ['Symbolic', 'Statistical', 'Connectionist']
        predictions = [symbolic, statistical, connectionist]
        
        accuracies = []
        for i, (name, pred) in enumerate(zip(approaches, predictions)):
            accuracy = np.mean(pred == y_true)
            accuracies.append(accuracy)
            print(f"{name:12}: {accuracy:.3f}")
        
        # Visualize comparison
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(approaches, accuracies, color=['red', 'blue', 'green'], alpha=0.7)
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Agreement matrix
        plt.subplot(1, 2, 2)
        agreement_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                agreement = np.mean(predictions[i] == predictions[j])
                agreement_matrix[i, j] = agreement
        
        im = plt.imshow(agreement_matrix, cmap='Blues')
        plt.colorbar(im)
        plt.xticks(range(3), approaches)
        plt.yticks(range(3), approaches)
        plt.title('Inter-Paradigm Agreement')
        
        # Add agreement values
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='white' if agreement_matrix[i, j] > 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n💡 Insights:")
        print(f"• Symbolic AI: Interpretable but may be too rigid")
        print(f"• Statistical AI: Principled approach with good performance")
        print(f"• Connectionist AI: Flexible learning but less interpretable")


# ==========================================
# COMPREHENSIVE HISTORICAL DEMONSTRATION
# ==========================================

def comprehensive_ai_history_demo():
    """
    Complete demonstration of AI history through implementations
    """
    print("🕰️  Neural Odyssey - Week 10: Historical Context of AI")
    print("=" * 65)
    print("From Logic to Learning: Understanding AI's Evolutionary Journey")
    print("=" * 65)
    
    # Initialize timeline
    timeline = AIHistoryTimeline()
    timeline.display_timeline()
    
    print("\n" + "="*65)
    print("🎭 HISTORICAL ALGORITHM IMPLEMENTATIONS")
    print("="*65)
    
    # 1. McCulloch-Pitts Neuron (1943)
    print(f"\n1️⃣  1943: The Birth of Artificial Neurons")
    print("-" * 45)
    mp_neuron = McCullochPittsNeuron([1, 1], threshold=2)
    mp_neuron.demonstrate_logic_gates()
    
    # 2. Hebbian Learning (1949)
    print(f"\n2️⃣  1949: The First Learning Rule")
    print("-" * 40)
    hebbian = HebbianLearning(4)
    hebbian.demonstrate_pattern_learning()
    
    # 3. Perceptron Algorithm (1957)
    print(f"\n3️⃣  1957: Machine Learning is Born")
    print("-" * 42)
    perceptron = PerceptronAlgorithm(2)
    perceptron.demonstrate_linear_separation()
    
    # 4. XOR Limitation (1969)
    print(f"\n4️⃣  1969: The Limitation that Started AI Winter")
    print("-" * 48)
    perceptron.demonstrate_xor_limitation()
    
    # 5. K-Means Clustering (1975)
    print(f"\n5️⃣  1975: Unsupervised Learning Emerges")
    print("-" * 44)
    kmeans = KMeansHistorical(3)
    kmeans.demonstrate_clustering()
    kmeans.demonstrate_elbow_method()
    
    # 6. Decision Trees (1980s)
    print(f"\n6️⃣  1980s: Symbolic AI Reaches Maturity")
    print("-" * 45)
    decision_tree = DecisionTreeHistorical()
    decision_tree.demonstrate_decision_tree()
    
    # 7. Backpropagation Solution (1986)
    print(f"\n7️⃣  1986: The Neural Renaissance Begins")
    print("-" * 45)
    backprop_xor = BackpropXORSolution()
    backprop_xor.demonstrate_xor_solution()
    
    # 8. Paradigm Comparison
    print(f"\n8️⃣  Paradigm Evolution: Comparing Approaches")
    print("-" * 48)
    paradigm_comp = AIParadigmComparison()
    paradigm_comp.demonstrate_paradigm_comparison()
    
    print("\n" + "="*65)
    print("🎓 KEY HISTORICAL INSIGHTS")
    print("="*65)
    
    insights = [
        "🧠 1943-1950s: AI begins with logic and computation theory",
        "📈 1950s-1960s: First learning algorithms prove machines can learn",
        "❄️  1970s: AI Winter - limitations of linear models discovered",
        "🌳 1980s: Symbolic AI flourishes with expert systems",
        "🔥 1986: Backpropagation resurrects neural networks",
        "📊 1990s-2000s: Statistical methods dominate machine learning",
        "🚀 2010s: Deep learning revolution changes everything"
    ]
    
    for insight in insights:
        print(f"  {insight}")
    
    print(f"\n💡 The Evolution Pattern:")
    print(f"   Logic → Learning → Limitations → Solutions → New Problems → New Solutions")
    print(f"   Each crisis in AI led to innovation and new breakthroughs!")
    
    print(f"\n🔮 Modern Implications:")
    print(f"   • Understanding history helps predict future developments")
    print(f"   • Each paradigm contributes unique strengths")
    print(f"   • Today's deep learning builds on all previous work")
    print(f"   • Next breakthroughs may come from combining paradigms")
    
    return {
        'timeline': timeline,
        'mcculloch_pitts': mp_neuron,
        'hebbian': hebbian,
        'perceptron': perceptron,
        'kmeans': kmeans,
        'decision_tree': decision_tree,
        'backprop_xor': backprop_xor,
        'paradigm_comparison': paradigm_comp
    }


# ==========================================
# INTERACTIVE HISTORICAL EXPLORATION
# ==========================================

class InteractiveHistoryExplorer:
    """
    Interactive exploration of AI history and concepts
    """
    
    def __init__(self):
        self.timeline = AIHistoryTimeline()
        self.current_era = None
    
    def explore_era(self, era_name: str):
        """Explore specific historical era"""
        era_milestones = {milestone: data for milestone, data in self.timeline.milestones.items() 
                         if data['era'] == era_name}
        
        if not era_milestones:
            print(f"❌ Era '{era_name}' not found")
            return
        
        print(f"\n🔍 Exploring Era: {era_name}")
        print("=" * (20 + len(era_name)))
        
        for year, milestone in era_milestones.items():
            print(f"\n📅 {year}: {milestone['milestone']}")
            print(f"   Description: {milestone['description']}")
            print(f"   Significance: {milestone['significance']}")
            print(f"   Key People: {', '.join(milestone['key_people'])}")
            print(f"   Math Contribution: {milestone['mathematical_contribution']}")
    
    def compare_algorithms(self, algorithm1: str, algorithm2: str, dataset_type: str = 'classification'):
        """Compare two historical algorithms on the same problem"""
        print(f"\n⚔️  Algorithm Comparison: {algorithm1} vs {algorithm2}")
        print("=" * 60)
        
        # Generate appropriate dataset
        if dataset_type == 'classification':
            X, y = self._generate_classification_data()
        else:
            X, y = self._generate_regression_data()
        
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        
        # Run algorithms
        results = {}
        
        if algorithm1 == 'perceptron':
            results[algorithm1] = self._run_perceptron(X, y)
        elif algorithm1 == 'decision_tree':
            results[algorithm1] = self._run_decision_tree(X, y)
        elif algorithm1 == 'kmeans':
            results[algorithm1] = self._run_kmeans(X, y)
        
        if algorithm2 == 'perceptron':
            results[algorithm2] = self._run_perceptron(X, y)
        elif algorithm2 == 'decision_tree':
            results[algorithm2] = self._run_decision_tree(X, y)
        elif algorithm2 == 'kmeans':
            results[algorithm2] = self._run_kmeans(X, y)
        
        # Compare results
        self._visualize_algorithm_comparison(X, y, results)
        
        return results
    
    def _generate_classification_data(self):
        """Generate classification dataset"""
        np.random.seed(42)
        n_samples = 200
        
        # Two classes with some overlap
        X1 = np.random.multivariate_normal([2, 2], [[0.5, 0.2], [0.2, 0.5]], n_samples//2)
        X2 = np.random.multivariate_normal([4, 4], [[0.5, -0.2], [-0.2, 0.5]], n_samples//2)
        
        X = np.vstack([X1, X2])
        y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
        
        return X, y
    
    def _run_perceptron(self, X, y):
        """Run perceptron algorithm"""
        perceptron = PerceptronAlgorithm(X.shape[1])
        converged = perceptron.train(X, y, max_epochs=100)
        predictions = np.array([perceptron.predict(x) for x in X])
        accuracy = np.mean(predictions == y)
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'converged': converged,
            'algorithm': 'Perceptron'
        }
    
    def _run_decision_tree(self, X, y):
        """Run decision tree algorithm"""
        tree = DecisionTreeHistorical(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        accuracy = np.mean(predictions == y)
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'converged': True,  # Decision trees always "converge"
            'algorithm': 'Decision Tree'
        }
    
    def _run_kmeans(self, X, y):
        """Run k-means algorithm (unsupervised)"""
        kmeans = KMeansHistorical(k=len(np.unique(y)))
        predictions = kmeans.fit(X)
        
        # For clustering, we'll measure how well clusters align with true labels
        # This is not a fair comparison but illustrative
        accuracy = 0  # Placeholder for unsupervised metric
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'converged': True,
            'algorithm': 'K-Means'
        }
    
    def _visualize_algorithm_comparison(self, X, y, results):
        """Visualize algorithm comparison results"""
        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms + 1, figsize=(5 * (n_algorithms + 1), 5))
        
        colors = ['red', 'blue', 'green', 'purple']
        
        # Plot original data
        for class_val in np.unique(y):
            mask = y == class_val
            axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                           label=f'Class {class_val}', alpha=0.7)
        axes[0].set_title('Original Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot algorithm results
        for i, (alg_name, result) in enumerate(results.items()):
            ax = axes[i + 1]
            predictions = result['predictions']
            
            for class_val in np.unique(predictions):
                mask = predictions == class_val
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], alpha=0.7)
            
            ax.set_title(f'{alg_name}\nAccuracy: {result["accuracy"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison summary
        print(f"\n📊 Comparison Summary:")
        for alg_name, result in results.items():
            status = "✅ Converged" if result['converged'] else "❌ No convergence"
            print(f"   {alg_name:15}: Accuracy = {result['accuracy']:.3f} | {status}")
    
    def historical_timeline_quiz(self):
        """Interactive quiz about AI history"""
        print(f"\n🎯 AI History Quiz")
        print("=" * 20)
        
        questions = [
            {
                "question": "Who developed the first mathematical model of a neuron?",
                "options": ["A) Alan Turing", "B) McCulloch & Pitts", "C) Frank Rosenblatt", "D) Geoffrey Hinton"],
                "answer": "B",
                "explanation": "McCulloch & Pitts created the first artificial neuron model in 1943"
            },
            {
                "question": "What problem showed the limitation of single-layer perceptrons?",
                "options": ["A) AND gate", "B) OR gate", "C) XOR gate", "D) NOT gate"],
                "answer": "C",
                "explanation": "The XOR problem cannot be solved by linear classifiers like single perceptrons"
            },
            {
                "question": "Which algorithm ended the first AI winter?",
                "options": ["A) K-means", "B) Decision trees", "C) Backpropagation", "D) Support Vector Machines"],
                "answer": "C",
                "explanation": "Backpropagation (1986) showed how to train multilayer networks effectively"
            },
            {
                "question": "What was the main approach of AI in the 1980s?",
                "options": ["A) Neural networks", "B) Statistical methods", "C) Symbolic AI", "D) Deep learning"],
                "answer": "C",
                "explanation": "The 1980s were dominated by symbolic AI and expert systems"
            }
        ]
        
        score = 0
        for i, q in enumerate(questions):
            print(f"\nQuestion {i+1}: {q['question']}")
            for option in q['options']:
                print(f"   {option}")
            
            user_answer = input("Your answer (A/B/C/D): ").upper().strip()
            
            if user_answer == q['answer']:
                print("✅ Correct!")
                score += 1
            else:
                print(f"❌ Incorrect. The answer is {q['answer']}")
            
            print(f"💡 {q['explanation']}")
        
        print(f"\n🎯 Final Score: {score}/{len(questions)} ({score/len(questions)*100:.0f}%)")
        
        if score == len(questions):
            print("🏆 Perfect! You're an AI history expert!")
        elif score >= len(questions) * 0.7:
            print("👍 Great job! You know your AI history well.")
        else:
            print("📚 Keep studying! AI history is fascinating and worth learning.")
        
        return score


# ==========================================
# HISTORICAL IMPACT ANALYSIS
# ==========================================

def analyze_historical_impact():
    """Analyze the impact of historical breakthroughs"""
    print(f"\n📈 Historical Impact Analysis")
    print("=" * 35)
    
    # Define impact metrics for each breakthrough
    breakthroughs = {
        "McCulloch-Pitts Neuron (1943)": {
            "theoretical_impact": 10,
            "practical_impact": 3,
            "longevity": 9,
            "influence_on_modern_ai": 8
        },
        "Perceptron (1957)": {
            "theoretical_impact": 9,
            "practical_impact": 5,
            "longevity": 7,
            "influence_on_modern_ai": 8
        },
        "Backpropagation (1986)": {
            "theoretical_impact": 10,
            "practical_impact": 10,
            "longevity": 10,
            "influence_on_modern_ai": 10
        },
        "K-means (1975)": {
            "theoretical_impact": 7,
            "practical_impact": 9,
            "longevity": 9,
            "influence_on_modern_ai": 6
        },
        "Decision Trees (1980s)": {
            "theoretical_impact": 6,
            "practical_impact": 8,
            "longevity": 8,
            "influence_on_modern_ai": 7
        }
    }
    
    # Create impact visualization
    metrics = list(next(iter(breakthroughs.values())).keys())
    n_metrics = len(metrics)
    n_breakthroughs = len(breakthroughs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Impact by metric
    for i, metric in enumerate(metrics):
        values = [data[metric] for data in breakthroughs.values()]
        names = list(breakthroughs.keys())
        
        ax = axes[i]
        bars = ax.bar(range(len(names)), values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45, ha='right')
        ax.set_ylim(0, 10)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate overall impact scores
    overall_scores = {}
    for name, impacts in breakthroughs.items():
        overall_scores[name] = sum(impacts.values()) / len(impacts)
    
    # Sort by overall impact
    sorted_breakthroughs = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Overall Impact Ranking:")
    for i, (name, score) in enumerate(sorted_breakthroughs):
        print(f"   {i+1}. {name}: {score:.1f}/10")
    
    return breakthroughs, overall_scores


# ==========================================
# MAIN EXECUTION AND TESTING
# ==========================================

if __name__ == "__main__":
    """
    Run this file to explore AI history through implementations!
    
    This comprehensive exploration covers:
    1. Complete AI timeline from 1943 to present
    2. Implementation of key historical algorithms
    3. Understanding paradigm shifts in AI
    4. Interactive exploration and comparisons
    5. Impact analysis of major breakthroughs
    
    To get started, run: python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey - Week 10: Historical Context!")
    print("Embark on a journey through AI history with hands-on implementations.")
    print("\nThis historical exploration includes:")
    print("1. 🕰️  Complete AI timeline with key milestones")
    print("2. 🧠 McCulloch-Pitts neurons and logic gates")
    print("3. 📈 Hebbian learning and associative memory")
    print("4. 🎯 Perceptron algorithm and its limitations")
    print("5. 🌳 Decision trees and symbolic AI")
    print("6. 📊 K-means clustering and unsupervised learning")
    print("7. 🔥 Backpropagation solving the XOR problem")
    print("8. 🔄 Paradigm comparison across AI history")
    print("9. 🎮 Interactive historical exploration")
    print("10. 📈 Impact analysis of breakthroughs")
    
    # Run the comprehensive demonstration
    print("\n" + "="*65)
    print("🎭 Starting Historical AI Journey...")
    print("="*65)
    
    # Main historical demonstration
    historical_results = comprehensive_ai_history_demo()
    
    # Interactive exploration
    print("\n" + "="*65)
    print("🎮 Interactive Historical Exploration")
    print("="*65)
    
    explorer = InteractiveHistoryExplorer()
    
    # Explore specific eras
    print("\n🔍 Exploring Different AI Eras:")
    eras = ["Birth of AI", "Learning Theory", "Neural Renaissance", "Deep Learning Revolution"]
    for era in eras:
        explorer.explore_era(era)
    
    # Algorithm comparison
    print("\n⚔️  Historical Algorithm Battle:")
    comparison_results = explorer.compare_algorithms('perceptron', 'decision_tree', 'classification')
    
    # Historical quiz
    quiz_score = explorer.historical_timeline_quiz()
    
    # Impact analysis
    print("\n📈 Analyzing Historical Impact...")
    impact_analysis = analyze_historical_impact()
    
    print("\n" + "="*65)
    print("🎓 Week 10 Complete: Historical Context Mastered!")
    print("="*65)
    print("🧠 You now understand:")
    print("   ✅ The complete evolution of AI from 1943 to present")
    print("   ✅ Key algorithms that shaped the field")
    print("   ✅ Why certain approaches succeeded or failed")
    print("   ✅ How mathematical breakthroughs drove progress")
    print("   ✅ The cyclical nature of AI development")
    print("   ✅ Connections between historical and modern methods")
    
    print("\n🚀 Ready for Week 11: Probability and Statistics!")
    print("   Next up: Bayesian thinking, distributions, and statistical learning")
    
    # Historical wisdom
    print("\n🔮 Historical Lessons for the Future:")
    wisdom = [
        "Every AI winter led to a stronger spring",
        "Mathematical foundations always prove crucial",
        "Interdisciplinary collaboration drives breakthroughs",
        "Simple ideas, when combined, create complex intelligence",
        "Understanding the past illuminates the future"
    ]
    
    for lesson in wisdom:
        print(f"   💡 {lesson}")
    
    print("\n📚 Suggested Further Exploration:")
    suggestions = [
        "Read original papers by Turing, McCulloch & Pitts, Rosenblatt",
        "Study the AI winter periods and their causes",
        "Explore connections between neuroscience and AI",
        "Compare historical AI approaches to current deep learning",
        "Investigate how historical algorithms influenced modern methods"
    ]
    
    for suggestion in suggestions:
        print(f"   📖 {suggestion}")
    
    # Return comprehensive results
    final_results = {
        'historical_implementations': historical_results,
        'interactive_exploration': {
            'comparison_results': comparison_results,
            'quiz_score': quiz_score
        },
        'impact_analysis': impact_analysis
    }
    
    print(f"\n🎯 Historical Journey Statistics:")
    print(f"   Algorithms implemented: {len(historical_results)}")
    print(f"   Historical milestones covered: {len(explorer.timeline.milestones)}")
    print(f"   Quiz score: {quiz_score}/{4}")
    print(f"   Understanding level: {'Expert' if quiz_score >= 3 else 'Intermediate' if quiz_score >= 2 else 'Beginner'}")
    
    print("\n🌟 The past is the key to understanding the present and shaping the future!")
    print("    You are now equipped with the historical wisdom of AI pioneers.")