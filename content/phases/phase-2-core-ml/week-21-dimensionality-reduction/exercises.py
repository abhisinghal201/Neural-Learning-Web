#!/usr/bin/env python3

"""
Neural Odyssey - Week 20: Dimensionality Reduction Exercises
Phase 2: Core Machine Learning - Month 5

This week focuses on advanced dimensionality reduction techniques that go beyond PCA.
You'll implement and understand t-SNE, UMAP, autoencoders, and manifold learning methods.

Learning Objectives:
1. Master t-SNE for non-linear dimensionality reduction
2. Implement UMAP for preserving both local and global structure  
3. Build autoencoders for neural dimensionality reduction
4. Apply manifold learning techniques (Isomap, LLE, MDS)
5. Compare different dimensionality reduction approaches
6. Handle high-dimensional real-world datasets

Author: Neural Explorer
Week: 20 (Phase 2 - Core ML)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, load_wine, make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
np.random.seed(42)

def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian kernel between two points."""
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

def perplexity_to_sigma(distances, target_perplexity=30.0, tolerance=1e-5, max_iter=50):
    """
    Convert perplexity to sigma for t-SNE using binary search.
    
    Parameters:
    -----------
    distances : array-like
        Distances from a point to all other points
    target_perplexity : float
        Target perplexity value
    tolerance : float
        Convergence tolerance
    max_iter : int
        Maximum iterations for binary search
    
    Returns:
    --------
    sigma : float
        Optimal sigma value
    """
    # Binary search for optimal sigma
    sigma_min, sigma_max = 1e-20, 1000.0
    
    for _ in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2.0
        
        # Compute probabilities
        prob = np.exp(-distances**2 / (2 * sigma**2))
        prob[0] = 0  # Set self-probability to 0
        prob_sum = np.sum(prob)
        
        if prob_sum == 0:
            sigma_min = sigma
            continue
            
        prob = prob / prob_sum
        
        # Compute entropy and perplexity
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        perplexity = 2**entropy
        
        # Check convergence
        if abs(perplexity - target_perplexity) <= tolerance:
            break
            
        if perplexity > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma
    
    return sigma

class TSNEImplementation:
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation from scratch.
    
    Based on the original paper:
    "Visualizing Data using t-SNE" by van der Maaten and Hinton (2008)
    """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, 
                 n_iter=1000, early_exaggeration=12.0, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        
    def _compute_joint_probabilities(self, X):
        """Compute joint probability matrix P in high-dimensional space."""
        n = X.shape[0]
        P = np.zeros((n, n))
        
        print("Computing pairwise affinities...")
        distances = pairwise_distances(X, metric='euclidean')
        
        # Compute conditional probabilities for each point
        for i in range(n):
            if i % 100 == 0:
                print(f"Processing point {i}/{n}")
                
            # Find optimal sigma for this point
            sigma = perplexity_to_sigma(distances[i], self.perplexity)
            
            # Compute conditional probabilities
            prob_i = np.exp(-distances[i]**2 / (2 * sigma**2))
            prob_i[i] = 0  # Set self-probability to 0
            prob_i = prob_i / np.sum(prob_i)
            
            P[i] = prob_i
        
        # Make probabilities symmetric
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)  # Avoid numerical issues
        
        return P
    
    def _compute_low_dim_affinities(self, Y):
        """Compute affinities in low-dimensional space using t-distribution."""
        n = Y.shape[0]
        distances_sq = pairwise_distances(Y, metric='euclidean')**2
        
        # Use t-distribution with 1 degree of freedom
        Q = 1 / (1 + distances_sq)
        np.fill_diagonal(Q, 0)  # Set diagonal to 0
        Q = Q / np.sum(Q)  # Normalize
        Q = np.maximum(Q, 1e-12)  # Avoid numerical issues
        
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """Compute gradient of KL divergence."""
        n = Y.shape[0]
        PQ_diff = P - Q
        
        # Compute gradient
        gradient = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y  # Shape: (n, n_components)
            distances_sq = np.sum(diff**2, axis=1)
            
            # t-SNE gradient formula
            inv_distances = 1 / (1 + distances_sq)
            inv_distances[i] = 0  # Set self-distance to 0
            
            weights = PQ_diff[i] * inv_distances
            gradient[i] = 4 * np.sum(weights[:, np.newaxis] * diff, axis=0)
        
        return gradient
    
    def fit_transform(self, X):
        """
        Fit t-SNE and return transformed data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        Y : array-like, shape (n_samples, n_components)
            Transformed data in low-dimensional space
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        print("🎯 Starting t-SNE Implementation")
        print("=" * 50)
        
        # Step 1: Compute joint probabilities in high-dimensional space
        print("Step 1: Computing joint probabilities...")
        P = self._compute_joint_probabilities(X)
        
        # Step 2: Initialize low-dimensional embedding
        print("Step 2: Initializing low-dimensional embedding...")
        Y = np.random.normal(0, 1e-4, (n_samples, self.n_components))
        
        # Step 3: Gradient descent optimization
        print("Step 3: Gradient descent optimization...")
        
        # Initialize momentum terms
        momentum = 0.5
        Y_velocity = np.zeros_like(Y)
        
        costs = []
        
        for iteration in range(self.n_iter):
            # Apply early exaggeration
            if iteration < 250:
                P_effective = P * self.early_exaggeration
            else:
                P_effective = P
                momentum = 0.8  # Increase momentum after early exaggeration
            
            # Compute affinities in low-dimensional space
            Q = self._compute_low_dim_affinities(Y)
            
            # Compute gradient
            gradient = self._compute_gradient(P_effective, Q, Y)
            
            # Update embedding with momentum
            Y_velocity = momentum * Y_velocity - self.learning_rate * gradient
            Y = Y + Y_velocity
            
            # Center the embedding
            Y = Y - np.mean(Y, axis=0)
            
            # Compute cost (KL divergence)
            if iteration % 100 == 0:
                cost = np.sum(P_effective * np.log(P_effective / Q))
                costs.append(cost)
                print(f"Iteration {iteration}: Cost = {cost:.4f}")
        
        print("✅ t-SNE optimization completed!")
        
        # Store cost history
        self.costs_ = costs
        
        return Y
    
    def plot_cost_evolution(self):
        """Plot the evolution of cost during optimization."""
        if hasattr(self, 'costs_'):
            plt.figure(figsize=(10, 6))
            plt.plot(range(0, len(self.costs_) * 100, 100), self.costs_, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('KL Divergence')
            plt.title('t-SNE Cost Evolution')
            plt.grid(True, alpha=0.3)
            plt.show()

class UMAPImplementation:
    """
    UMAP (Uniform Manifold Approximation and Projection) simplified implementation.
    
    Based on the paper:
    "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
    by McInnes, Healy, and Melville (2018)
    """
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 learning_rate=1.0, n_epochs=200, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
    
    def _compute_knn_graph(self, X):
        """Compute k-nearest neighbor graph."""
        n_samples = X.shape[0]
        distances = pairwise_distances(X)
        
        # For each point, find k nearest neighbors
        knn_indices = np.zeros((n_samples, self.n_neighbors), dtype=int)
        knn_distances = np.zeros((n_samples, self.n_neighbors))
        
        for i in range(n_samples):
            # Get sorted indices of distances (excluding self)
            sorted_indices = np.argsort(distances[i])
            knn_indices[i] = sorted_indices[1:self.n_neighbors+1]  # Exclude self
            knn_distances[i] = distances[i][knn_indices[i]]
        
        return knn_indices, knn_distances
    
    def _compute_fuzzy_simplicial_set(self, knn_indices, knn_distances):
        """Compute fuzzy simplicial set representation."""
        n_samples = knn_indices.shape[0]
        
        # Initialize sparse matrix representation
        rows = []
        cols = []
        vals = []
        
        for i in range(n_samples):
            # Compute sigma (bandwidth) for this point
            rho = knn_distances[i, 0]  # Distance to nearest neighbor
            
            # Binary search for sigma
            sigma_min, sigma_max = 0.0, np.inf
            target = np.log2(self.n_neighbors)
            
            for _ in range(64):  # Maximum iterations
                if sigma_max == np.inf:
                    sigma = sigma_min * 2
                else:
                    sigma = (sigma_min + sigma_max) / 2
                
                # Compute probabilities
                distances_adj = np.maximum(0, knn_distances[i] - rho)
                if sigma > 0:
                    probs = np.exp(-distances_adj / sigma)
                else:
                    probs = np.zeros_like(distances_adj)
                    probs[0] = 1.0
                
                entropy = -np.sum(probs * np.log2(probs + 1e-8))
                
                if abs(entropy - target) < 1e-5:
                    break
                
                if entropy > target:
                    sigma_max = sigma
                else:
                    sigma_min = sigma
            
            # Store edges
            for j, neighbor_idx in enumerate(knn_indices[i]):
                if sigma > 0:
                    val = np.exp(-max(0, knn_distances[i, j] - rho) / sigma)
                else:
                    val = 1.0 if j == 0 else 0.0
                
                rows.append(i)
                cols.append(neighbor_idx)
                vals.append(val)
        
        # Create adjacency matrix
        from scipy.sparse import coo_matrix
        graph = coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
        
        # Symmetrize: A_sym = A + A^T - A ⊙ A^T
        graph_t = graph.T
        intersection = graph.multiply(graph_t)
        graph_sym = graph + graph_t - intersection
        
        return graph_sym
    
    def fit_transform(self, X):
        """
        Fit UMAP and return transformed data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        Y : array-like, shape (n_samples, n_components)
            Transformed data in low-dimensional space
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        print("🎯 Starting UMAP Implementation")
        print("=" * 50)
        
        # Step 1: Compute k-nearest neighbor graph
        print("Step 1: Computing k-NN graph...")
        knn_indices, knn_distances = self._compute_knn_graph(X)
        
        # Step 2: Compute fuzzy simplicial set
        print("Step 2: Computing fuzzy simplicial set...")
        graph = self._compute_fuzzy_simplicial_set(knn_indices, knn_distances)
        
        # Step 3: Initialize low-dimensional embedding
        print("Step 3: Initializing embedding...")
        Y = np.random.uniform(-10, 10, (n_samples, self.n_components))
        
        # Step 4: Optimize embedding using simplified force-directed layout
        print("Step 4: Optimizing embedding...")
        
        # Convert sparse matrix to dense for simplicity
        graph_dense = graph.toarray()
        
        for epoch in range(self.n_epochs):
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}")
            
            # Compute attractive and repulsive forces
            for i in range(n_samples):
                attractive_force = np.zeros(self.n_components)
                repulsive_force = np.zeros(self.n_components)
                
                for j in range(n_samples):
                    if i == j:
                        continue
                    
                    diff = Y[i] - Y[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        # Attractive force (for connected points)
                        if graph_dense[i, j] > 0:
                            attractive_force += graph_dense[i, j] * diff / (1 + dist**2)
                        
                        # Repulsive force (simplified)
                        repulsive_force -= 0.01 * diff / (1 + dist**2)**2
                
                # Update position
                total_force = attractive_force + repulsive_force
                Y[i] += self.learning_rate * total_force
            
            # Reduce learning rate
            self.learning_rate *= 0.99
        
        print("✅ UMAP optimization completed!")
        return Y

class ManifoldLearningToolkit:
    """Comprehensive toolkit for manifold learning and dimensionality reduction."""
    
    def __init__(self):
        self.methods = {}
        self.results = {}
    
    def compare_methods(self, X, y=None, dataset_name="Dataset"):
        """
        Compare different dimensionality reduction methods.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples,), optional
            Target labels for coloring
        dataset_name : str
            Name of the dataset for plotting
        """
        print(f"\n🔍 Comparing Dimensionality Reduction Methods on {dataset_name}")
        print("=" * 60)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define methods to compare
        methods = {
            'PCA': PCA(n_components=2, random_state=42),
            't-SNE (sklearn)': TSNE(n_components=2, random_state=42, perplexity=30),
            't-SNE (custom)': TSNEImplementation(n_components=2, perplexity=30),
            'Isomap': Isomap(n_components=2, n_neighbors=10),
            'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42),
            'MDS': MDS(n_components=2, random_state=42),
        }
        
        # Apply each method
        results = {}
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (name, method) in enumerate(methods.items()):
            print(f"Applying {name}...")
            
            try:
                if name == 't-SNE (custom)':
                    # Use our custom implementation
                    X_transformed = method.fit_transform(X_scaled)
                else:
                    # Use sklearn implementation
                    X_transformed = method.fit_transform(X_scaled)
                
                results[name] = X_transformed
                
                # Plot results
                ax = axes[idx]
                scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                   c=y if y is not None else 'blue', 
                                   cmap='tab10', alpha=0.7, s=50)
                ax.set_title(f'{name}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.grid(True, alpha=0.3)
                
                if y is not None and idx == 0:
                    plt.colorbar(scatter, ax=ax)
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                axes[idx].text(0.5, 0.5, f'{name}\nFailed', 
                              transform=axes[idx].transAxes, 
                              ha='center', va='center')
        
        plt.suptitle(f'Dimensionality Reduction Comparison: {dataset_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        self.results[dataset_name] = results
        return results
    
    def evaluate_neighborhood_preservation(self, X_original, X_transformed, k=10):
        """
        Evaluate how well local neighborhoods are preserved.
        
        Parameters:
        -----------
        X_original : array-like
            Original high-dimensional data
        X_transformed : array-like
            Transformed low-dimensional data
        k : int
            Number of neighbors to consider
            
        Returns:
        --------
        preservation_score : float
            Fraction of neighbors preserved (0-1)
        """
        n_samples = X_original.shape[0]
        
        # Compute k-NN in original space
        distances_orig = pairwise_distances(X_original)
        neighbors_orig = np.argsort(distances_orig, axis=1)[:, 1:k+1]
        
        # Compute k-NN in transformed space
        distances_trans = pairwise_distances(X_transformed)
        neighbors_trans = np.argsort(distances_trans, axis=1)[:, 1:k+1]
        
        # Calculate preservation score
        preserved = 0
        total = n_samples * k
        
        for i in range(n_samples):
            preserved += len(set(neighbors_orig[i]) & set(neighbors_trans[i]))
        
        preservation_score = preserved / total
        return preservation_score

# TODO: Implement autoencoder for dimensionality reduction
class AutoencoderDimensionalityReduction:
    """
    Neural network autoencoder for non-linear dimensionality reduction.
    
    TODO: Implement this class with:
    1. Encoder network (input -> latent space)
    2. Decoder network (latent space -> output)  
    3. Training with reconstruction loss
    4. Dimensionality reduction via encoder
    """
    
    def __init__(self, encoding_dim=2, hidden_layers=[64, 32], 
                 activation='relu', learning_rate=0.001, epochs=100):
        """
        Initialize autoencoder for dimensionality reduction.
        
        Parameters:
        -----------
        encoding_dim : int
            Dimensionality of encoded representation
        hidden_layers : list
            Sizes of hidden layers in encoder
        activation : str
            Activation function
        learning_rate : float
            Learning rate for training
        epochs : int
            Number of training epochs
        """
        # TODO: Implement autoencoder initialization
        pass
    
    def fit(self, X):
        """
        Train the autoencoder on input data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        # TODO: Implement autoencoder training
        # 1. Build encoder and decoder networks
        # 2. Compile model with reconstruction loss
        # 3. Train on input data
        # 4. Store training history
        pass
    
    def transform(self, X):
        """
        Transform data to lower-dimensional representation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_encoded : array-like, shape (n_samples, encoding_dim)
            Encoded representation
        """
        # TODO: Use encoder part to transform data
        pass
    
    def inverse_transform(self, X_encoded):
        """
        Reconstruct data from encoded representation.
        
        Parameters:
        -----------
        X_encoded : array-like, shape (n_samples, encoding_dim)
            Encoded data
            
        Returns:
        --------
        X_reconstructed : array-like, shape (n_samples, n_features)
            Reconstructed data
        """
        # TODO: Use decoder part to reconstruct data
        pass
    
    def fit_transform(self, X):
        """Fit autoencoder and return encoded representation."""
        self.fit(X)
        return self.transform(X)

def demonstrate_tsne_implementation():
    """Demonstrate custom t-SNE implementation."""
    print("\n🧮 t-SNE Implementation Demonstration")
    print("=" * 50)
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Apply PCA first to reduce computational cost
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    
    print(f"After PCA: {X_pca.shape}")
    print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Apply custom t-SNE
    tsne_custom = TSNEImplementation(n_components=2, perplexity=30, n_iter=1000)
    X_tsne = tsne_custom.fit_transform(X_pca)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # PCA visualization
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=50)
    plt.title('PCA (First 2 Components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    # t-SNE visualization
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=50)
    plt.title('t-SNE (Custom Implementation)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost evolution
    tsne_custom.plot_cost_evolution()

def demonstrate_umap_implementation():
    """Demonstrate custom UMAP implementation."""
    print("\n🗺️ UMAP Implementation Demonstration")
    print("=" * 50)
    
    # Generate swiss roll dataset
    X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    
    print(f"Swiss roll dataset shape: {X.shape}")
    
    # Apply custom UMAP
    umap_custom = UMAPImplementation(n_components=2, n_neighbors=15, n_epochs=200)
    X_umap = umap_custom.fit_transform(X)
    
    # Compare with PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Visualize results
    fig = plt.figure(figsize=(18, 6))
    
    # Original 3D data
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='Spectral', s=50)
    ax1.set_title('Original Swiss Roll (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # PCA visualization
    ax2 = fig.add_subplot(1, 3, 2)
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='Spectral', s=50)
    ax2.set_title('PCA')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)
    
    # UMAP visualization
    ax3 = fig.add_subplot(1, 3, 3)
    scatter = ax3.scatter(X_umap[:, 0], X_umap[:, 1], c=color, cmap='Spectral', s=50)
    ax3.set_title('UMAP (Custom Implementation)')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_manifold_learning_comparison():
    """Demonstrate comprehensive comparison of manifold learning methods."""
    print("\n🔬 Manifold Learning Methods Comparison")
    print("=" * 50)
    
    toolkit = ManifoldLearningToolkit()
    
    # Test on different datasets
    datasets = {
        'Digits': (load_digits().data, load_digits().target),
        'Wine': (load_wine().data, load_wine().target),
        'S-Curve': make_s_curve(n_samples=1000, noise=0.1, random_state=42),
        'Swiss Roll': make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    }
    
    for name, (X, y) in datasets.items():
        if X.shape[0] > 500:  # Subsample large datasets for demo
            indices = np.random.choice(X.shape[0], 500, replace=False)
            X = X[indices]
            y = y[indices] if y is not None else None
        
        print(f"\n📊 Analyzing {name} dataset...")
        results = toolkit.compare_methods(X, y, name)
        
        # Evaluate neighborhood preservation for some methods
        if 'PCA' in results and 't-SNE (sklearn)' in results:
            pca_score = toolkit.evaluate_neighborhood_preservation(X, results['PCA'])
            tsne_score = toolkit.evaluate_neighborhood_preservation(X, results['t-SNE (sklearn)'])
            
            print(f"Neighborhood preservation (k=10):")
            print(f"  PCA: {pca_score:.3f}")
            print(f"  t-SNE: {tsne_score:.3f}")

def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to hyperparameters in dimensionality reduction."""
    print("\n⚙️ Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Generate synthetic data
    X, color = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
    
    # Test t-SNE with different perplexity values
    perplexity_values = [5, 15, 30, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, perplexity in enumerate(perplexity_values):
        print(f"Testing t-SNE with perplexity={perplexity}...")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        ax = axes[idx]
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='Spectral', s=50)
        ax.set_title(f't-SNE (perplexity={perplexity})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
   
   plt.suptitle('t-SNE Sensitivity to Perplexity Parameter', fontsize=16)
   plt.tight_layout()
   plt.show()
   
   # Test UMAP with different n_neighbors values
   try:
       from umap import UMAP
       
       n_neighbors_values = [5, 15, 30, 50]
       
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       axes = axes.ravel()
       
       for idx, n_neighbors in enumerate(n_neighbors_values):
           print(f"Testing UMAP with n_neighbors={n_neighbors}...")
           
           umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
           X_umap = umap_model.fit_transform(X)
           
           ax = axes[idx]
           scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=color, cmap='Spectral', s=50)
           ax.set_title(f'UMAP (n_neighbors={n_neighbors})')
           ax.set_xlabel('UMAP 1')
           ax.set_ylabel('UMAP 2')
           ax.grid(True, alpha=0.3)
       
       plt.suptitle('UMAP Sensitivity to n_neighbors Parameter', fontsize=16)
       plt.tight_layout()
       plt.show()
       
   except ImportError:
       print("UMAP not installed. Skipping UMAP parameter sensitivity analysis.")

def analyze_computational_complexity():
   """Analyze computational complexity of different methods."""
   print("\n⏱️ Computational Complexity Analysis")
   print("=" * 50)
   
   import time
   
   # Test on datasets of increasing size
   sample_sizes = [100, 200, 500, 1000]
   methods = {
       'PCA': lambda X: PCA(n_components=2, random_state=42).fit_transform(X),
       't-SNE': lambda X: TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]//4)).fit_transform(X),
       'Isomap': lambda X: Isomap(n_components=2, n_neighbors=min(10, X.shape[0]//2)).fit_transform(X),
       'MDS': lambda X: MDS(n_components=2, random_state=42).fit_transform(X)
   }
   
   results = {method: [] for method in methods}
   
   for n_samples in sample_sizes:
       print(f"\nTesting with {n_samples} samples...")
       
       # Generate random data
       X = np.random.randn(n_samples, 20)
       
       for method_name, method_func in methods.items():
           try:
               start_time = time.time()
               _ = method_func(X)
               elapsed_time = time.time() - start_time
               results[method_name].append(elapsed_time)
               print(f"  {method_name}: {elapsed_time:.3f}s")
           except Exception as e:
               print(f"  {method_name}: Failed ({e})")
               results[method_name].append(np.nan)
   
   # Plot results
   plt.figure(figsize=(10, 6))
   for method_name, times in results.items():
       valid_times = [(size, time) for size, time in zip(sample_sizes, times) if not np.isnan(time)]
       if valid_times:
           sizes, times = zip(*valid_times)
           plt.plot(sizes, times, 'o-', label=method_name, linewidth=2, markersize=8)
   
   plt.xlabel('Number of Samples')
   plt.ylabel('Computation Time (seconds)')
   plt.title('Computational Complexity Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.yscale('log')
   plt.show()

def demonstrate_quality_metrics():
   """Demonstrate quality metrics for dimensionality reduction."""
   print("\n📊 Quality Metrics for Dimensionality Reduction")
   print("=" * 50)
   
   from sklearn.metrics import silhouette_score
   from scipy.stats import spearmanr
   
   # Load digits dataset
   digits = load_digits()
   X = digits.data
   y = digits.target
   
   # Subsample for computational efficiency
   indices = np.random.choice(X.shape[0], 500, replace=False)
   X = X[indices]
   y = y[indices]
   
   # Apply different methods
   methods = {
       'PCA': PCA(n_components=2, random_state=42),
       't-SNE': TSNE(n_components=2, random_state=42, perplexity=30),
       'Isomap': Isomap(n_components=2, n_neighbors=10)
   }
   
   results = {}
   quality_metrics = {}
   
   for name, method in methods.items():
       print(f"Applying {name}...")
       X_transformed = method.fit_transform(X)
       results[name] = X_transformed
       
       # Calculate quality metrics
       metrics = {}
       
       # 1. Silhouette score (clustering quality)
       try:
           sil_score = silhouette_score(X_transformed, y)
           metrics['Silhouette Score'] = sil_score
       except:
           metrics['Silhouette Score'] = np.nan
       
       # 2. Trustworthiness (neighborhood preservation)
       toolkit = ManifoldLearningToolkit()
       trust_score = toolkit.evaluate_neighborhood_preservation(X, X_transformed, k=10)
       metrics['Trustworthiness'] = trust_score
       
       # 3. Stress (distance preservation for MDS-like methods)
       try:
           distances_orig = pairwise_distances(X)
           distances_trans = pairwise_distances(X_transformed)
           
           # Flatten upper triangular parts
           triu_indices = np.triu_indices_from(distances_orig, k=1)
           orig_flat = distances_orig[triu_indices]
           trans_flat = distances_trans[triu_indices]
           
           # Compute Spearman correlation (rank correlation)
           corr, _ = spearmanr(orig_flat, trans_flat)
           metrics['Distance Correlation'] = corr
           
       except:
           metrics['Distance Correlation'] = np.nan
       
       quality_metrics[name] = metrics
   
   # Display results
   print("\n📈 Quality Metrics Summary:")
   print("-" * 60)
   print(f"{'Method':<15} {'Silhouette':<12} {'Trustworth.':<12} {'Dist. Corr.':<12}")
   print("-" * 60)
   
   for method, metrics in quality_metrics.items():
       sil = f"{metrics['Silhouette Score']:.3f}" if not np.isnan(metrics['Silhouette Score']) else "N/A"
       trust = f"{metrics['Trustworthiness']:.3f}" if not np.isnan(metrics['Trustworthiness']) else "N/A"
       corr = f"{metrics['Distance Correlation']:.3f}" if not np.isnan(metrics['Distance Correlation']) else "N/A"
       print(f"{method:<15} {sil:<12} {trust:<12} {corr:<12}")
   
   # Visualize results with quality scores
   fig, axes = plt.subplots(1, 3, figsize=(18, 5))
   
   for idx, (name, X_transformed) in enumerate(results.items()):
       ax = axes[idx]
       scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                          c=y, cmap='tab10', alpha=0.7, s=50)
       
       # Add quality metrics to title
       metrics = quality_metrics[name]
       sil = metrics['Silhouette Score']
       trust = metrics['Trustworthiness']
       
       title = f"{name}\n"
       if not np.isnan(sil):
           title += f"Silhouette: {sil:.3f}, "
       title += f"Trust: {trust:.3f}"
       
       ax.set_title(title)
       ax.set_xlabel('Component 1')
       ax.set_ylabel('Component 2')
       ax.grid(True, alpha=0.3)
       
       if idx == 0:
           plt.colorbar(scatter, ax=ax)
   
   plt.tight_layout()
   plt.show()

def create_dimensionality_reduction_report():
   """Create comprehensive dimensionality reduction analysis report."""
   print("\n📋 Comprehensive Dimensionality Reduction Report")
   print("=" * 60)
   
   report = """
   DIMENSIONALITY REDUCTION ANALYSIS REPORT
   ========================================
   
   METHODS IMPLEMENTED:
   ===================
   ✅ Principal Component Analysis (PCA)
      - Linear method preserving global variance
      - Best for: Linear relationships, preprocessing
      - Time complexity: O(min(n²p, np²))
   
   ✅ t-SNE (t-Distributed Stochastic Neighbor Embedding)
      - Non-linear method preserving local neighborhoods
      - Best for: Data visualization, cluster discovery
      - Time complexity: O(n²)
   
   ✅ UMAP (Uniform Manifold Approximation and Projection)
      - Non-linear method preserving local and global structure
      - Best for: General-purpose dimensionality reduction
      - Time complexity: O(n log n)
   
   ✅ Isomap (Isometric Mapping)
      - Non-linear method using geodesic distances
      - Best for: Manifolds with intrinsic geometry
      - Time complexity: O(n²)
   
   ✅ LLE (Locally Linear Embedding)
      - Non-linear method preserving local linear relationships
      - Best for: Locally linear manifolds
      - Time complexity: O(n²)
   
   ✅ MDS (Multidimensional Scaling)
      - Method preserving pairwise distances
      - Best for: Distance-based analysis
      - Time complexity: O(n³)
   
   QUALITY METRICS:
   ================
   ✅ Trustworthiness (Neighborhood Preservation)
      - Measures preservation of local neighborhoods
      - Range: [0, 1], higher is better
   
   ✅ Silhouette Score (Clustering Quality)
      - Measures separation between clusters
      - Range: [-1, 1], higher is better
   
   ✅ Distance Correlation (Global Structure)
      - Measures preservation of pairwise distances
      - Range: [-1, 1], higher is better
   
   IMPLEMENTATION DETAILS:
   ======================
   • Custom t-SNE with proper perplexity optimization
   • Simplified UMAP with fuzzy simplicial sets
   • Comprehensive comparison framework
   • Parameter sensitivity analysis
   • Computational complexity evaluation
   
   RECOMMENDATIONS:
   ===============
   📌 Use PCA for:
      - Initial data exploration
      - Preprocessing before other methods
      - Linear dimensionality reduction
      - Feature extraction with interpretability
   
   📌 Use t-SNE for:
      - Data visualization (2D/3D)
      - Cluster discovery and analysis
      - Non-linear pattern exploration
      - Final visualization step
   
   📌 Use UMAP for:
      - General-purpose dimensionality reduction
      - Preserving both local and global structure
      - Faster alternative to t-SNE
      - Downstream machine learning tasks
   
   📌 Use Isomap for:
      - Data on known manifolds
      - Preserving geodesic distances
      - Non-linear dimensionality with global preservation
   
   COMMON PITFALLS:
   ===============
   ⚠️  t-SNE hyperparameter sensitivity (perplexity)
   ⚠️  UMAP parameter tuning (n_neighbors, min_dist)
   ⚠️  Computational complexity for large datasets
   ⚠️  Over-interpretation of cluster distances in t-SNE
   ⚠️  Assuming linear relationships with non-linear data
   
   NEXT STEPS:
   ===========
   🎯 Implement variational autoencoders for dimensionality reduction
   🎯 Study deep dimensionality reduction methods
   🎯 Apply to domain-specific problems (text, images, etc.)
   🎯 Combine with clustering and classification tasks
   """
   
   print(report)

def main():
   """Main function to run all dimensionality reduction demonstrations."""
   print("🌌 Neural Odyssey Week 20: Advanced Dimensionality Reduction")
   print("=" * 65)
   print("Welcome to the world of manifold learning and non-linear dimensionality reduction!")
   print("This week, we'll explore methods that can uncover hidden structure in complex data.")
   print()
   
   try:
       # Core implementations
       demonstrate_tsne_implementation()
       demonstrate_umap_implementation()
       
       # Comprehensive comparisons
       demonstrate_manifold_learning_comparison()
       
       # Analysis and evaluation
       demonstrate_parameter_sensitivity()
       analyze_computational_complexity()
       demonstrate_quality_metrics()
       
       # Final report
       create_dimensionality_reduction_report()
       
       print("\n🎉 Week 20 Completed Successfully!")
       print("=" * 50)
       print("Key Achievements:")
       print("✅ Implemented t-SNE from scratch")
       print("✅ Built simplified UMAP algorithm")
       print("✅ Compared multiple manifold learning methods")
       print("✅ Analyzed parameter sensitivity and computational complexity")
       print("✅ Developed quality metrics for evaluation")
       print("✅ Created comprehensive analysis framework")
       
       print("\n🔮 Next Week Preview: Semi-Supervised Learning")
       print("You'll learn how to leverage both labeled and unlabeled data!")
       
   except Exception as e:
       print(f"❌ Error in main execution: {e}")
       print("\nTroubleshooting tips:")
       print("1. Check that all required packages are installed")
       print("2. Verify dataset loading functions work correctly")
       print("3. Review function implementations for errors")
       print("4. Start with individual demonstrations")

if __name__ == "__main__":
   """
   Run this file to explore advanced dimensionality reduction!
   
   This week focuses on non-linear methods that can uncover
   hidden manifold structure in high-dimensional data.
   
   Key learning objectives:
   1. Understand when and why to use non-linear dimensionality reduction
   2. Implement t-SNE algorithm from mathematical principles
   3. Build simplified UMAP for manifold learning
   4. Compare different methods on various datasets
   5. Evaluate quality using appropriate metrics
   6. Handle real-world dimensionality reduction challenges
   
   Usage:
   python week20_dimensionality_reduction_exercises.py
   """
   
   print("🎯 To get started:")
   print("1. Run individual demonstrations to understand each method")
   print("2. Experiment with different datasets and parameters")
   print("3. Compare results across multiple quality metrics")
   print("4. TODO: Implement the AutoencoderDimensionalityReduction class")
   print("5. Apply methods to your own high-dimensional datasets")
   
   print("\n💡 Pro Tips:")
   print("• PCA first, then non-linear methods for large datasets")
   print("• t-SNE great for visualization, UMAP better for downstream tasks")
   print("• Always validate with multiple quality metrics")
   print("• Parameter tuning is crucial for optimal results")
   
   print("\n🏆 Master Challenge:")
   print("Build a complete pipeline that automatically selects the best")
   print("dimensionality reduction method for any given dataset!")
   
   # Uncomment to run all demonstrations
   main()