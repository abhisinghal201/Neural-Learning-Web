#!/usr/bin/env python3
"""
Neural Odyssey - Phase 2: Core Machine Learning
Week 20: Advanced Clustering Algorithms
Complete Exercise Implementation

This comprehensive module extends Week 19's foundations to explore advanced clustering
algorithms, density-based methods, spectral clustering, and modern approaches that
handle complex data structures and challenging real-world scenarios.

Learning Path:
1. DBSCAN and density-based clustering for arbitrary cluster shapes
2. Spectral clustering and graph-based approaches
3. Mean Shift clustering and mode-seeking algorithms
4. OPTICS for variable density clustering
5. Affinity Propagation for cluster exemplars
6. Online and streaming clustering algorithms
7. Deep clustering and neural network approaches

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional, Union, Dict, List, Any, Callable
import warnings
from abc import ABC, abstractmethod
import time
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_blobs, make_circles, make_moons, load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from collections import defaultdict, deque
import itertools

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

class AdvancedClusteringFoundations:
    """
    Advanced foundations and utilities for sophisticated clustering algorithms
    """
    
    @staticmethod
    def compute_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """Compute pairwise distance matrix"""
        if metric == 'euclidean':
            return cdist(X, X, metric='euclidean')
        elif metric == 'manhattan':
            return cdist(X, X, metric='manhattan')
        elif metric == 'cosine':
            return cdist(X, X, metric='cosine')
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def compute_k_neighbors(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute k-nearest neighbors for each point"""
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        # Remove self (first neighbor)
        return distances[:, 1:], indices[:, 1:]
    
    @staticmethod
    def compute_local_density(X: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute local density using Gaussian kernel"""
        n_samples = len(X)
        density = np.zeros(n_samples)
        
        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
            density[i] = np.sum(weights)
        
        return density
    
    @staticmethod
    def create_similarity_matrix(X: np.ndarray, 
                                sigma: float = 1.0, 
                                k_neighbors: Optional[int] = None) -> np.ndarray:
        """Create similarity matrix for spectral clustering"""
        n_samples = len(X)
        
        # Compute pairwise distances
        distances = cdist(X, X, metric='euclidean')
        
        # Gaussian similarity
        similarity = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        
        # Optional: k-nearest neighbors sparsification
        if k_neighbors is not None:
            knn_mask = np.zeros_like(similarity, dtype=bool)
            for i in range(n_samples):
                k_nearest = np.argsort(distances[i])[1:k_neighbors+1]  # Exclude self
                knn_mask[i, k_nearest] = True
                knn_mask[k_nearest, i] = True  # Make symmetric
            similarity = similarity * knn_mask
        
        # Set diagonal to zero
        np.fill_diagonal(similarity, 0)
        
        return similarity

class DBSCANFromScratch:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    
    Mathematical Foundation:
    - Core points: Points with at least min_samples neighbors within eps distance
    - Border points: Non-core points within eps distance of a core point
    - Noise points: Points that are neither core nor border
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        """
        Initialize DBSCAN clustering
        
        Parameters:
        -----------
        eps : float, maximum distance between two samples for them to be considered neighbors
        min_samples : int, minimum number of samples in neighborhood for core point
        metric : str, distance metric to use
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        # Fitted attributes
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
    
    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Get indices of neighbors within eps distance"""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()
    
    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray, 
                       point_idx: int, neighbors: List[int], 
                       cluster_id: int, core_samples: set) -> None:
        """Expand cluster using density-reachability"""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor is noise, change to border point
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # Get neighbor's neighbors
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # If neighbor is core point, add its neighbors to expansion list
                if len(neighbor_neighbors) >= self.min_samples:
                    core_samples.add(neighbor_idx)
                    neighbors.extend(neighbor_neighbors)
            
            i += 1
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array of shape (n_samples,), cluster labels (-1 for noise)
        """
        X = np.array(X)
        n_samples = len(X)
        
        # Initialize labels (0 = unvisited, -1 = noise, >0 = cluster)
        labels = np.zeros(n_samples, dtype=int)
        core_samples = set()
        
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != 0:
                continue
            
            # Get neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # Mark as noise
            else:
                # Start new cluster
                cluster_id += 1
                core_samples.add(point_idx)
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id, core_samples)
        
        # Store results
        self.labels_ = labels
        self.core_sample_indices_ = np.array(list(core_samples))
        self.n_clusters_ = cluster_id
        self.n_noise_ = np.sum(labels == -1)
        
        return labels

class SpectralClusteringFromScratch:
    """
    Spectral Clustering using eigendecomposition of graph Laplacian
    
    Mathematical Foundation:
    - Build similarity graph from data
    - Compute normalized graph Laplacian
    - Find smallest eigenvectors of Laplacian
    - Apply k-means to eigenvector embeddings
    """
    
    def __init__(self, 
                 n_clusters: int = 2,
                 gamma: float = 1.0,
                 k_neighbors: Optional[int] = None,
                 laplacian_type: str = 'normalized'):
        """
        Initialize Spectral Clustering
        
        Parameters:
        -----------
        n_clusters : int, number of clusters
        gamma : float, kernel coefficient for RBF similarity
        k_neighbors : int or None, sparsify using k-nearest neighbors
        laplacian_type : str, type of Laplacian ('unnormalized' or 'normalized')
        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.laplacian_type = laplacian_type
        
        # Fitted attributes
        self.similarity_matrix_ = None
        self.laplacian_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.labels_ = None
    
    def _compute_laplacian(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute graph Laplacian matrix"""
        # Degree matrix
        degree = np.sum(similarity_matrix, axis=1)
        
        if self.laplacian_type == 'unnormalized':
            # L = D - W
            laplacian = np.diag(degree) - similarity_matrix
        
        elif self.laplacian_type == 'normalized':
            # L_norm = D^(-1/2) * (D - W) * D^(-1/2)
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-10))
            laplacian = degree_sqrt_inv @ (np.diag(degree) - similarity_matrix) @ degree_sqrt_inv
        
        else:
            raise ValueError(f"Unknown Laplacian type: {self.laplacian_type}")
        
        return laplacian
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform spectral clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array of shape (n_samples,), cluster labels
        """
        X = np.array(X)
        
        # Step 1: Build similarity matrix
        self.similarity_matrix_ = AdvancedClusteringFoundations.create_similarity_matrix(
            X, sigma=1.0/self.gamma, k_neighbors=self.k_neighbors
        )
        
        # Step 2: Compute Laplacian
        self.laplacian_ = self._compute_laplacian(self.similarity_matrix_)
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian_)
        
        # Sort by eigenvalues (smallest first)
        idx = np.argsort(eigenvalues)
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        
        # Step 4: Use smallest eigenvectors for embedding
        embedding = self.eigenvectors_[:, :self.n_clusters]
        
        # Step 5: Normalize rows (for normalized Laplacian)
        if self.laplacian_type == 'normalized':
            embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
        
        # Step 6: Apply k-means to embedding
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = kmeans.fit_predict(embedding)
        
        return self.labels_

class MeanShiftFromScratch:
    """
    Mean Shift clustering algorithm
    
    Mathematical Foundation:
    - Iteratively shift points toward modes of kernel density estimate
    - Convergence points become cluster centers
    - Points converging to same mode belong to same cluster
    """
    
    def __init__(self, bandwidth: float = 1.0, max_iters: int = 300, tol: float = 1e-3):
        """
        Initialize Mean Shift clustering
        
        Parameters:
        -----------
        bandwidth : float, kernel bandwidth parameter
        max_iters : int, maximum number of iterations
        tol : float, convergence tolerance
        """
        self.bandwidth = bandwidth
        self.max_iters = max_iters
        self.tol = tol
        
        # Fitted attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_clusters_ = None
    
    def _gaussian_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Gaussian kernel function"""
        return np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2))
    
    def _mean_shift_step(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Single mean shift iteration"""
        distances = np.linalg.norm(X - center, axis=1)
        weights = self._gaussian_kernel(distances)
        
        # Avoid division by zero
        total_weight = np.sum(weights)
        if total_weight == 0:
            return center
        
        # Weighted mean
        new_center = np.sum(weights[:, np.newaxis] * X, axis=0) / total_weight
        return new_center
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Mean Shift clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array of shape (n_samples,), cluster labels
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize: each point is a potential mode
        modes = X.copy()
        
        # Iteratively shift each point toward local mode
        for iteration in range(self.max_iters):
            new_modes = np.zeros_like(modes)
            
            for i in range(n_samples):
                new_modes[i] = self._mean_shift_step(X, modes[i])
            
            # Check convergence
            shifts = np.linalg.norm(new_modes - modes, axis=1)
            if np.all(shifts < self.tol):
                break
            
            modes = new_modes
        
        # Merge nearby modes
        cluster_centers = []
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            mode = modes[i]
            
            # Check if mode is close to existing cluster center
            assigned = False
            for j, center in enumerate(cluster_centers):
                if np.linalg.norm(mode - center) < self.bandwidth:
                    labels[i] = j
                    assigned = True
                    break
            
            # Create new cluster center
            if not assigned:
                cluster_centers.append(mode)
                labels[i] = len(cluster_centers) - 1
        
        # Store results
        self.cluster_centers_ = np.array(cluster_centers)
        self.labels_ = labels
        self.n_clusters_ = len(cluster_centers)
        
        return labels

class OPTICSFromScratch:
    """
    OPTICS (Ordering Points To Identify Clustering Structure)
    
    Mathematical Foundation:
    - Extension of DBSCAN that handles varying densities
    - Produces reachability plot showing cluster hierarchy
    - Extract clusters by analyzing reachability distances
    """
    
    def __init__(self, min_samples: int = 5, max_eps: float = np.inf, metric: str = 'euclidean'):
        """
        Initialize OPTICS clustering
        
        Parameters:
        -----------
        min_samples : int, minimum number of samples in neighborhood
        max_eps : float, maximum distance between two samples
        metric : str, distance metric to use
        """
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        
        # Fitted attributes
        self.ordering_ = None
        self.reachability_ = None
        self.core_distances_ = None
        self.labels_ = None
    
    def _core_distance(self, X: np.ndarray, point_idx: int) -> float:
        """Compute core distance for a point"""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        distances = np.sort(distances)
        
        # Core distance is distance to min_samples-th neighbor
        if len(distances) >= self.min_samples:
            return distances[self.min_samples - 1]
        else:
            return np.inf
    
    def _reachability_distance(self, X: np.ndarray, point_idx: int, neighbor_idx: int) -> float:
        """Compute reachability distance between two points"""
        core_dist = self._core_distance(X, neighbor_idx)
        direct_dist = np.linalg.norm(X[point_idx] - X[neighbor_idx])
        return max(core_dist, direct_dist)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform OPTICS clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array of shape (n_samples,), cluster labels
        """
        X = np.array(X)
        n_samples = len(X)
        
        # Initialize
        processed = np.zeros(n_samples, dtype=bool)
        ordering = []
        reachability = []
        core_distances = np.array([self._core_distance(X, i) for i in range(n_samples)])
        
        # Priority queue (using list for simplicity)
        # In practice, would use heapq for efficiency
        seeds = []
        
        for point_idx in range(n_samples):
            if processed[point_idx]:
                continue
            
            # Add current point to ordering
            processed[point_idx] = True
            ordering.append(point_idx)
            reachability.append(np.inf)  # First point has infinite reachability
            
            # If core point, add neighbors to seeds
            if core_distances[point_idx] <= self.max_eps:
                # Get neighbors within max_eps
                distances = np.linalg.norm(X - X[point_idx], axis=1)
                neighbors = np.where((distances <= self.max_eps) & ~processed)[0]
                
                # Update seeds
                for neighbor_idx in neighbors:
                    if not processed[neighbor_idx]:
                        reach_dist = self._reachability_distance(X, neighbor_idx, point_idx)
                        seeds.append((reach_dist, neighbor_idx))
            
            # Process seeds in order of reachability distance
            while seeds:
                # Sort seeds by reachability distance
                seeds.sort(key=lambda x: x[0])
                reach_dist, next_point = seeds.pop(0)
                
                if processed[next_point]:
                    continue
                
                # Add to ordering
                processed[next_point] = True
                ordering.append(next_point)
                reachability.append(reach_dist)
                
                # If core point, update seeds
                if core_distances[next_point] <= self.max_eps:
                    distances = np.linalg.norm(X - X[next_point], axis=1)
                    neighbors = np.where((distances <= self.max_eps) & ~processed)[0]
                    
                    for neighbor_idx in neighbors:
                        if not processed[neighbor_idx]:
                            new_reach_dist = self._reachability_distance(X, neighbor_idx, next_point)
                            
                            # Update if better reachability distance
                            updated = False
                            for i, (old_dist, old_idx) in enumerate(seeds):
                                if old_idx == neighbor_idx and new_reach_dist < old_dist:
                                    seeds[i] = (new_reach_dist, neighbor_idx)
                                    updated = True
                                    break
                            
                            if not updated:
                                seeds.append((new_reach_dist, neighbor_idx))
        
        # Store results
        self.ordering_ = np.array(ordering)
        self.reachability_ = np.array(reachability)
        self.core_distances_ = core_distances
        
        # Extract clusters using simple threshold method
        # In practice, would use more sophisticated methods
        threshold = np.percentile(self.reachability_[1:], 75)  # Exclude first inf value
        
        labels = np.zeros(n_samples, dtype=int)
        current_cluster = 0
        
        for i, reach_dist in enumerate(self.reachability_):
            if i == 0 or reach_dist > threshold:
                current_cluster += 1
            labels[self.ordering_[i]] = current_cluster - 1
        
        self.labels_ = labels
        return labels

class AffinityPropagationFromScratch:
    """
    Affinity Propagation clustering algorithm
    
    Mathematical Foundation:
    - Find exemplars by passing messages between data points
    - Responsibility: how suitable point j is as exemplar for point i
    - Availability: how appropriate it is for point i to choose point j as exemplar
    """
    
    def __init__(self, 
                 damping: float = 0.5, 
                 max_iters: int = 200,
                 convergence_iters: int = 15,
                 preference: Optional[float] = None):
        """
        Initialize Affinity Propagation
        
        Parameters:
        -----------
        damping : float, damping factor between 0.5 and 1
        max_iters : int, maximum number of iterations
        convergence_iters : int, number of iterations with no change for convergence
        preference : float or None, preference values (diagonal of similarity matrix)
        """
        self.damping = damping
        self.max_iters = max_iters
        self.convergence_iters = convergence_iters
        self.preference = preference
        
        # Fitted attributes
        self.cluster_centers_indices_ = None
        self.labels_ = None
        self.n_clusters_ = None
        self.similarity_matrix_ = None
    
    def _compute_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute similarity matrix (negative squared distances)"""
        distances = cdist(X, X, metric='euclidean')
        similarity = -distances ** 2
        
        # Set preference values (diagonal)
        if self.preference is None:
            preference = np.median(similarity)
        else:
            preference = self.preference
        
        np.fill_diagonal(similarity, preference)
        return similarity
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Affinity Propagation clustering
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array of shape (n_samples,), cluster labels
        """
        X = np.array(X)
        n_samples = len(X)
        
        # Compute similarity matrix
        S = self._compute_similarity_matrix(X)
        self.similarity_matrix_ = S
        
        # Initialize responsibility and availability matrices
        R = np.zeros((n_samples, n_samples))  # Responsibility
        A = np.zeros((n_samples, n_samples))  # Availability
        
        # Track convergence
        exemplars_history = []
        
        for iteration in range(self.max_iters):
            # Update responsibilities
            # R(i,k) = S(i,k) - max{A(i,k') + S(i,k') : k' ≠ k}
            AS = A + S
            I = np.argmax(AS, axis=1)
            Y = AS[np.arange(n_samples), I]
            
            AS[np.arange(n_samples), I] = -np.inf
            Y2 = np.max(AS, axis=1)
            
            R_new = S - Y[:, np.newaxis]
            R_new[np.arange(n_samples), I] = S[np.arange(n_samples), I] - Y2
            
            # Damping
            R = self.damping * R + (1 - self.damping) * R_new
            
            # Update availabilities
            # A(i,k) = min{0, R(k,k) + sum{max(0, R(i',k)) : i' ∉ {i,k}}}
            Rp = np.maximum(R, 0)
            for k in range(n_samples):
                A[:, k] = np.minimum(0, R[k, k] + np.sum(Rp[:, k]) - Rp[:, k] - Rp[k, k])
            
            # Set diagonal
            A[np.arange(n_samples), np.arange(n_samples)] = np.sum(np.maximum(R, 0), axis=0) - np.diag(R)
            
            # Damping
            A = self.damping * A + (1 - self.damping) * A
            
            # Check for convergence
            exemplars = np.where(np.diag(R + A) > 0)[0]
            exemplars_history.append(set(exemplars))
            
            if len(exemplars_history) >= self.convergence_iters:
                if len(set.union(*exemplars_history[-self.convergence_iters:])) == len(exemplars):
                    break
        
        # Identify exemplars and assign labels
        exemplars = np.where(np.diag(R + A) > 0)[0]
        self.cluster_centers_indices_ = exemplars
        self.n_clusters_ = len(exemplars)
        
        if self.n_clusters_ == 0:
            # No exemplars found, assign all points to single cluster
            self.labels_ = np.zeros(n_samples, dtype=int)
        else:
            # Assign each point to closest exemplar
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                exemplar_similarities = S[i, exemplars]
                labels[i] = np.argmax(exemplar_similarities)
            
            self.labels_ = labels
        
        return self.labels_

class OnlineKMeans:
    """
    Online K-means for streaming data
    
    Mathematical Foundation:
    - Update cluster centers incrementally as new data arrives
    - Maintain running averages and counts for each cluster
    - Handle concept drift through adaptive learning rates
    """
    
    def __init__(self, 
                 n_clusters: int = 3,
                 learning_rate: float = 0.1,
                 decay_factor: float = 0.9):
        """
        Initialize Online K-means
        
        Parameters:
        -----------
        n_clusters : int, number of clusters
        learning_rate : float, initial learning rate
        decay_factor : float, learning rate decay factor
        """
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        
        # Streaming attributes
        self.cluster_centers_ = None
        self.cluster_counts_ = None
        self.n_samples_seen_ = 0
        self.current_learning_rate_ = learning_rate
    
    def partial_fit(self, X: np.ndarray) -> 'OnlineKMeans':
        """
        Update model with new batch of data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize on first batch
        if self.cluster_centers_ is None:
            # Use first k points as initial centers
            if n_samples >= self.n_clusters:
                self.cluster_centers_ = X[:self.n_clusters].copy()
            else:
                # Repeat points if not enough samples
                indices = np.tile(np.arange(n_samples), (self.n_clusters // n_samples) + 1)[:self.n_clusters]
                self.cluster_centers_ = X[indices].copy()
            
            self.cluster_counts_ = np.ones(self.n_clusters)
        
        # Process each sample
        for sample in X:
            # Find closest cluster
            distances = np.linalg.norm(self.cluster_centers_ - sample, axis=1)
            closest_cluster = np.argmin(distances)
            
            # Update cluster center
            self.cluster_counts_[closest_cluster] += 1
            alpha = self.current_learning_rate_ / self.cluster_counts_[closest_cluster]
            
            self.cluster_centers_[closest_cluster] += alpha * (sample - self.cluster_centers_[closest_cluster])
        
        # Update learning rate and sample count
        self.n_samples_seen_ += n_samples
        self.current_learning_rate_ *= self.decay_factor
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

class ClusteringBenchmark:
    """
    Comprehensive benchmarking suite for clustering algorithms
    """
    
    @staticmethod
    def generate_test_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate diverse test datasets for algorithm comparison"""
        datasets = {}
        
        # Well-separated blobs
        X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
        datasets['blobs'] = (X1, y1)
        
        # Circles (non-convex)
        X2, y2 = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        datasets['circles'] = (X2, y2)
        
        # Moons (non-convex)
        X3, y3 = make_moons(n_samples=300, noise=0.1, random_state=42)
        datasets['moons'] = (X3, y3)
        
        # Variable density clusters
        np.random.seed(42)
        dense_cluster = np.random.multivariate_normal([0, 0], [[0.2, 0], [0, 0.2]], 100)
        sparse_cluster = np.random.multivariate_normal([3, 3], [[1.5, 0], [0, 1.5]], 100)
        outliers = np.random.uniform(-2, 5, (20, 2))
        
        X4 = np.vstack([dense_cluster, sparse_cluster, outliers])
        y4 = np.hstack([np.zeros(100), np.ones(100), np.full(20, 2)])
        datasets['variable_density'] = (X4, y4)
        
        # Elongated clusters
        cluster1 = np.random.multivariate_normal([0, 0], [[3, 2], [2, 3]], 100)
        cluster2 = np.random.multivariate_normal([6, 6], [[3, -2], [-2, 3]], 100)
        X5 = np.vstack([cluster1, cluster2])
        y5 = np.hstack([np.zeros(100), np.ones(100)])
        datasets['elongated'] = (X5, y5)
        
        return datasets
    
    @staticmethod
    def run_algorithm_comparison(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                figsize: Tuple[int, int] = (20, 12)):
        """Compare multiple clustering algorithms on different datasets"""
        
        algorithms = {
            'DBSCAN': lambda X: DBSCANFromScratch(eps=0.3, min_samples=5).fit_predict(X),
            'Spectral': lambda X: SpectralClusteringFromScratch(n_clusters=2, gamma=1.0).fit_predict(X),
            'Mean Shift': lambda X: MeanShiftFromScratch(bandwidth=0.8).fit_predict(X),
            'OPTICS': lambda X: OPTICSFromScratch(min_samples=5, max_eps=1.0).fit_predict(X),
            'Affinity Prop': lambda X: AffinityPropagationFromScratch(damping=0.7).fit_predict(X)
        }
        
        n_datasets = len(datasets)
        n_algorithms = len(algorithms)
        
        fig, axes = plt.subplots(n_datasets, n_algorithms + 1, figsize=figsize)
        
        for row, (dataset_name, (X, y_true)) in enumerate(datasets.items()):
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Plot ground truth
            axes[row, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
            axes[row, 0].set_title(f'{dataset_name}\n(Ground Truth)')
            axes[row, 0].grid(True, alpha=0.3)
            
            # Apply each algorithm
            for col, (alg_name, alg_func) in enumerate(algorithms.items()):
                try:
                    labels = alg_func(X_scaled)
                    
                    # Calculate metrics
                    if len(np.unique(labels)) > 1:
                        ari = adjusted_rand_score(y_true, labels)
                        sil = silhouette_score(X_scaled, labels)
                    else:
                        ari = sil = 0.0
                    
                    # Plot results
                    scatter = axes[row, col + 1].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                                        c=labels, cmap='viridis', alpha=0.7)
                    axes[row, col + 1].set_title(f'{alg_name}\nARI: {ari:.3f}, Sil: {sil:.3f}')
                    axes[row, col + 1].grid(True, alpha=0.3)
                    
                    # Highlight noise points for DBSCAN/OPTICS
                    if alg_name in ['DBSCAN', 'OPTICS']:
                        noise_mask = labels == -1
                        if np.any(noise_mask):
                            axes[row, col + 1].scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
                                                      c='red', marker='x', s=50, alpha=0.8)
                
                except Exception as e:
                    axes[row, col + 1].text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                                           ha='center', va='center', transform=axes[row, col + 1].transAxes)
                    axes[row, col + 1].set_title(f'{alg_name}\n(Failed)')
        
        plt.tight_layout()
        plt.show()

class AdvancedClusteringExperiments:
    """Advanced clustering experiments and real-world applications"""
    
    @staticmethod
    def experiment_1_dbscan_density_analysis():
        """Experiment 1: DBSCAN for density-based clustering and noise detection"""
        print("="*70)
        print("EXPERIMENT 1: DBSCAN DENSITY-BASED CLUSTERING AND NOISE DETECTION")
        print("="*70)
        
        # Generate complex dataset with noise
        np.random.seed(42)
        
        # Main clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 80)
        cluster2 = np.random.multivariate_normal([7, 7], [[0.8, 0.3], [0.3, 0.8]], 60)
        cluster3 = np.random.multivariate_normal([2, 7], [[0.3, 0], [0, 1.2]], 70)
        
        # Add noise points
        noise = np.random.uniform(0, 10, (30, 2))
        
        X_dbscan = np.vstack([cluster1, cluster2, cluster3, noise])
        
        print(f"Dataset: {len(X_dbscan)} points (210 cluster points + 30 noise points)")
        
        print("\n1.1 PARAMETER SENSITIVITY ANALYSIS")
        print("-" * 50)
        
        # Test different eps values
        eps_values = [0.3, 0.5, 0.8, 1.2]
        min_samples = 5
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, eps in enumerate(eps_values):
            dbscan = DBSCANFromScratch(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_dbscan)
            
            # Plot results
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for k, color in zip(unique_labels, colors):
                if k == -1:
                    # Noise points
                    class_member_mask = (labels == k)
                    xy = X_dbscan[class_member_mask]
                    axes[idx].scatter(xy[:, 0], xy[:, 1], c='red', marker='x', s=50, alpha=0.8, label='Noise')
                else:
                    # Cluster points
                    class_member_mask = (labels == k)
                    xy = X_dbscan[class_member_mask]
                    axes[idx].scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7, s=50)
            
            # Highlight core points
            if len(dbscan.core_sample_indices_) > 0:
                axes[idx].scatter(X_dbscan[dbscan.core_sample_indices_, 0], 
                                X_dbscan[dbscan.core_sample_indices_, 1],
                                s=100, facecolors='none', edgecolors='black', linewidth=2, alpha=0.8)
            
            axes[idx].set_title(f'eps={eps}, min_samples={min_samples}\n'
                              f'Clusters: {dbscan.n_clusters_}, Noise: {dbscan.n_noise_}')
            axes[idx].grid(True, alpha=0.3)
            
            print(f"eps={eps}: {dbscan.n_clusters_} clusters, {dbscan.n_noise_} noise points")
        
        plt.tight_layout()
        plt.show()
        
        print("\n1.2 OPTIMAL PARAMETER SELECTION")
        print("-" * 50)
        
        # k-distance plot for eps selection
        k = min_samples
        distances, _ = AdvancedClusteringFoundations.compute_k_neighbors(X_dbscan, k)
        k_distances = distances[:, -1]  # Distance to k-th neighbor
        k_distances_sorted = np.sort(k_distances)[::-1]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=2)
        plt.xlabel('Data Points (sorted by distance)')
        plt.ylabel(f'{k}-th Nearest Neighbor Distance')
        plt.title(f'{k}-Distance Plot for eps Selection')
        plt.grid(True, alpha=0.3)
        
        # Find elbow point (simplified method)
        # In practice, would use more sophisticated elbow detection
        elbow_idx = np.argmax(np.diff(k_distances_sorted)) + 1
        suggested_eps = k_distances_sorted[elbow_idx]
        plt.axhline(y=suggested_eps, color='red', linestyle='--', 
                   label=f'Suggested eps: {suggested_eps:.3f}')
        plt.legend()
        
        # Apply DBSCAN with suggested parameters
        optimal_dbscan = DBSCANFromScratch(eps=suggested_eps, min_samples=min_samples)
        optimal_labels = optimal_dbscan.fit_predict(X_dbscan)
        
        plt.subplot(1, 2, 2)
        unique_labels = np.unique(optimal_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for k, color in zip(unique_labels, colors):
            if k == -1:
                class_member_mask = (optimal_labels == k)
                xy = X_dbscan[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c='red', marker='x', s=50, alpha=0.8)
            else:
                class_member_mask = (optimal_labels == k)
                xy = X_dbscan[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7, s=50)
        
        plt.title(f'Optimal DBSCAN\neps={suggested_eps:.3f}, '
                 f'Clusters: {optimal_dbscan.n_clusters_}, Noise: {optimal_dbscan.n_noise_}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Suggested eps: {suggested_eps:.3f}")
        print(f"Optimal clustering: {optimal_dbscan.n_clusters_} clusters, {optimal_dbscan.n_noise_} noise points")
        
        return {
            'data': X_dbscan,
            'optimal_params': {'eps': suggested_eps, 'min_samples': min_samples},
            'optimal_labels': optimal_labels,
            'optimal_dbscan': optimal_dbscan
        }
    
    @staticmethod
    def experiment_2_spectral_clustering_graphs():
        """Experiment 2: Spectral clustering for graph-based and manifold data"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: SPECTRAL CLUSTERING FOR GRAPH-BASED AND MANIFOLD DATA")
        print("="*70)
        
        # Generate manifold-like data
        np.random.seed(42)
        
        # Two interlocking spirals
        n_points = 150
        theta = np.linspace(0, 4*np.pi, n_points)
        
        # Spiral 1
        r1 = theta / (2*np.pi)
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)
        spiral1 = np.column_stack([x1, y1])
        
        # Spiral 2 (rotated)
        r2 = theta / (2*np.pi) 
        x2 = r2 * np.cos(theta + np.pi)
        y2 = r2 * np.sin(theta + np.pi)
        spiral2 = np.column_stack([x2, y2])
        
        # Add noise
        spiral1 += np.random.normal(0, 0.1, spiral1.shape)
        spiral2 += np.random.normal(0, 0.1, spiral2.shape)
        
        X_spiral = np.vstack([spiral1, spiral2])
        y_spiral = np.hstack([np.zeros(n_points), np.ones(n_points)])
        
        print(f"Spiral dataset: {len(X_spiral)} points in 2 interlocking spirals")
        
        print("\n2.1 SIMILARITY MATRIX ANALYSIS")
        print("-" * 50)
        
        # Compare different similarity matrices
        sigma_values = [0.5, 1.0, 2.0]
        k_neighbors_values = [None, 5, 10]
        
        fig, axes = plt.subplots(len(sigma_values), len(k_neighbors_values) + 1, figsize=(16, 12))
        
        # Plot original data
        for i in range(len(sigma_values)):
            axes[i, 0].scatter(X_spiral[y_spiral == 0, 0], X_spiral[y_spiral == 0, 1], 
                             c='red', alpha=0.7, s=30, label='Spiral 1')
            axes[i, 0].scatter(X_spiral[y_spiral == 1, 0], X_spiral[y_spiral == 1, 1], 
                             c='blue', alpha=0.7, s=30, label='Spiral 2')
            axes[i, 0].set_title('Original Data')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
        
        spectral_results = {}
        
        for i, sigma in enumerate(sigma_values):
            for j, k_neighbors in enumerate(k_neighbors_values):
                # Apply spectral clustering
                spectral = SpectralClusteringFromScratch(
                    n_clusters=2, gamma=1.0/sigma, k_neighbors=k_neighbors
                )
                labels = spectral.fit_predict(X_spiral)
                
                # Calculate accuracy (handle label permutation)
                ari = adjusted_rand_score(y_spiral, labels)
                
                # Store results
                spectral_results[(sigma, k_neighbors)] = {
                    'labels': labels,
                    'ari': ari,
                    'similarity_matrix': spectral.similarity_matrix_,
                    'eigenvalues': spectral.eigenvalues_
                }
                
                # Plot results
                axes[i, j + 1].scatter(X_spiral[:, 0], X_spiral[:, 1], c=labels, 
                                     cmap='viridis', alpha=0.7, s=30)
                axes[i, j + 1].set_title(f'σ={sigma}, k={k_neighbors}\nARI: {ari:.3f}')
                axes[i, j + 1].grid(True, alpha=0.3)
                
                print(f"σ={sigma}, k_neighbors={k_neighbors}: ARI = {ari:.3f}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n2.2 EIGENVALUE ANALYSIS")
        print("-" * 50)
        
        # Analyze eigenvalues for best performing configuration
        best_config = max(spectral_results.keys(), key=lambda x: spectral_results[x]['ari'])
        best_result = spectral_results[best_config]
        
        plt.figure(figsize=(12, 5))
        
        # Plot eigenvalues
        plt.subplot(1, 2, 1)
        eigenvalues = best_result['eigenvalues'][:20]  # First 20 eigenvalues
        plt.plot(range(len(eigenvalues)), eigenvalues, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.title('Smallest Eigenvalues of Graph Laplacian')
        plt.grid(True, alpha=0.3)
        
        # Highlight eigengap
        if len(eigenvalues) > 2:
            eigengap = eigenvalues[2] - eigenvalues[1]
            plt.axhline(y=eigenvalues[1], color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=eigenvalues[2], color='red', linestyle='--', alpha=0.7)
            plt.text(len(eigenvalues)/2, (eigenvalues[1] + eigenvalues[2])/2, 
                    f'Eigengap: {eigengap:.4f}', fontsize=12, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot similarity matrix
        plt.subplot(1, 2, 2)
        similarity_matrix = best_result['similarity_matrix']
        im = plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Similarity')
        plt.title(f'Similarity Matrix\n(σ={best_config[0]}, k={best_config[1]})')
        plt.xlabel('Data Point Index')
        plt.ylabel('Data Point Index')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Best configuration: σ={best_config[0]}, k_neighbors={best_config[1]}")
        print(f"Best ARI: {best_result['ari']:.3f}")
        
        return {
            'data': (X_spiral, y_spiral),
            'spectral_results': spectral_results,
            'best_config': best_config,
            'best_result': best_result
        }
    
    @staticmethod
    def experiment_3_clustering_comparison_benchmark():
        """Experiment 3: Comprehensive clustering algorithm comparison"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: COMPREHENSIVE CLUSTERING ALGORITHM COMPARISON")
        print("="*70)
        
        # Generate comprehensive test datasets
        datasets = ClusteringBenchmark.generate_test_datasets()
        
        print(f"Comparing algorithms on {len(datasets)} different dataset types:")
        for name, (X, y) in datasets.items():
            print(f"  {name}: {len(X)} points, {len(np.unique(y))} clusters")
        
        print("\n3.1 ALGORITHM PERFORMANCE MATRIX")
        print("-" * 50)
        
        # Run comprehensive comparison
        ClusteringBenchmark.run_algorithm_comparison(datasets)
        
        # Detailed performance analysis
        algorithms = {
            'DBSCAN': lambda X: DBSCANFromScratch(eps=0.3, min_samples=5).fit_predict(X),
            'Spectral': lambda X: SpectralClusteringFromScratch(n_clusters=2, gamma=1.0).fit_predict(X),
            'Mean Shift': lambda X: MeanShiftFromScratch(bandwidth=0.8).fit_predict(X),
            'OPTICS': lambda X: OPTICSFromScratch(min_samples=5, max_eps=1.0).fit_predict(X),
            'Affinity Prop': lambda X: AffinityPropagationFromScratch(damping=0.7).fit_predict(X)
        }
        
        # Performance metrics table
        performance_table = []
        
        for dataset_name, (X, y_true) in datasets.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            row = {'Dataset': dataset_name}
            
            for alg_name, alg_func in algorithms.items():
                try:
                    start_time = time.time()
                    labels = alg_func(X_scaled)
                    end_time = time.time()
                    
                    if len(np.unique(labels)) > 1:
                        ari = adjusted_rand_score(y_true, labels)
                        sil = silhouette_score(X_scaled, labels)
                    else:
                        ari = sil = 0.0
                    
                    runtime = end_time - start_time
                    
                    row[f'{alg_name}_ARI'] = ari
                    row[f'{alg_name}_Silhouette'] = sil
                    row[f'{alg_name}_Time'] = runtime
                    
                except Exception as e:
                    row[f'{alg_name}_ARI'] = 0.0
                    row[f'{alg_name}_Silhouette'] = 0.0
                    row[f'{alg_name}_Time'] = np.inf
            
            performance_table.append(row)
        
        # Create performance DataFrame
        df_performance = pd.DataFrame(performance_table)
        
        print("\nPerformance Summary (ARI scores):")
        print("-" * 40)
        for _, row in df_performance.iterrows():
            print(f"{row['Dataset']:15}", end=" | ")
            for alg in ['DBSCAN', 'Spectral', 'Mean Shift', 'OPTICS', 'Affinity Prop']:
                score = row[f'{alg}_ARI']
                print(f"{alg[:8]:8s}: {score:.3f}", end=" | ")
            print()
        
        print("\n3.2 ALGORITHM STRENGTHS AND WEAKNESSES")
        print("-" * 50)
        
        # Analyze which algorithm performs best on each dataset type
        algorithm_strengths = {}
        
        for alg in ['DBSCAN', 'Spectral', 'Mean Shift', 'OPTICS', 'Affinity Prop']:
            algorithm_strengths[alg] = {
                'best_datasets': [],
                'avg_ari': 0.0,
                'avg_runtime': 0.0
            }
        
        for _, row in df_performance.iterrows():
            dataset = row['Dataset']
            best_alg = None
            best_score = -1
            
            for alg in ['DBSCAN', 'Spectral', 'Mean Shift', 'OPTICS', 'Affinity Prop']:
                score = row[f'{alg}_ARI']
                if score > best_score:
                    best_score = score
                    best_alg = alg
            
            if best_alg:
                algorithm_strengths[best_alg]['best_datasets'].append(dataset)
        
        # Calculate average performance
        for alg in ['DBSCAN', 'Spectral', 'Mean Shift', 'OPTICS', 'Affinity Prop']:
            ari_scores = df_performance[f'{alg}_ARI'].values
            runtimes = df_performance[f'{alg}_Time'].values
            
            algorithm_strengths[alg]['avg_ari'] = np.mean(ari_scores)
            algorithm_strengths[alg]['avg_runtime'] = np.mean(runtimes[runtimes != np.inf])
        
        print("Algorithm Performance Analysis:")
        for alg, stats in algorithm_strengths.items():
            print(f"\n{alg}:")
            print(f"  Best on: {', '.join(stats['best_datasets']) if stats['best_datasets'] else 'None'}")
            print(f"  Average ARI: {stats['avg_ari']:.3f}")
            print(f"  Average Runtime: {stats['avg_runtime']:.4f}s")
        
        return {
            'datasets': datasets,
            'performance_table': df_performance,
            'algorithm_strengths': algorithm_strengths
        }
    
    @staticmethod
    def experiment_4_streaming_clustering():
        """Experiment 4: Online clustering for streaming data"""
        print("\n" + "="*70)
        print("EXPERIMENT 4: ONLINE CLUSTERING FOR STREAMING DATA")
        print("="*70)
        
        # Simulate streaming data with concept drift
        np.random.seed(42)
        
        def generate_stream_batch(batch_size: int, time_step: int) -> np.ndarray:
            """Generate batch of streaming data with concept drift"""
            # Gradual drift in cluster centers
            drift_factor = time_step * 0.1
            
            if time_step < 20:
                # Initial clusters
                center1 = [2 + drift_factor * 0.1, 2 + drift_factor * 0.1]
                center2 = [6 + drift_factor * 0.1, 6 + drift_factor * 0.1]
            else:
                # Concept drift: clusters move
                center1 = [2 + (time_step - 20) * 0.2, 2 - (time_step - 20) * 0.1]
                center2 = [6 - (time_step - 20) * 0.1, 6 + (time_step - 20) * 0.2]
            
            # Generate batch
            n_per_cluster = batch_size // 2
            cluster1 = np.random.multivariate_normal(center1, [[0.5, 0], [0, 0.5]], n_per_cluster)
            cluster2 = np.random.multivariate_normal(center2, [[0.5, 0], [0, 0.5]], n_per_cluster)
            
            return np.vstack([cluster1, cluster2])
        
        print("Simulating streaming data with concept drift...")
        print("Initial phase: Stable clusters")
        print("Drift phase: Clusters gradually move")
        
        print("\n4.1 ONLINE K-MEANS ADAPTATION")
        print("-" * 50)
        
        # Initialize online k-means
        online_kmeans = OnlineKMeans(n_clusters=2, learning_rate=0.1, decay_factor=0.95)
        
        # Track cluster centers over time
        center_history = []
        batch_data_history = []
        
        n_batches = 40
        batch_size = 50
        
        for t in range(n_batches):
            # Generate new batch
            batch = generate_stream_batch(batch_size, t)
            batch_data_history.append(batch)
            
            # Update model
            online_kmeans.partial_fit(batch)
            center_history.append(online_kmeans.cluster_centers_.copy())
            
            if t % 10 == 0:
                print(f"Batch {t}: Centers = {online_kmeans.cluster_centers_}")
        
        print(f"\nProcessed {n_batches} batches ({n_batches * batch_size} total points)")
        
        # Visualize streaming results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot cluster center evolution
        axes[0, 0].set_title('Cluster Center Evolution')
        center_history = np.array(center_history)
        
        # Cluster 1 trajectory
        axes[0, 0].plot(center_history[:, 0, 0], center_history[:, 0, 1], 
                       'o-', label='Cluster 1', linewidth=2, markersize=4)
        
        # Cluster 2 trajectory  
        axes[0, 0].plot(center_history[:, 1, 0], center_history[:, 1, 1], 
                       's-', label='Cluster 2', linewidth=2, markersize=4)
        
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Y coordinate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot learning rate decay
        axes[0, 1].set_title('Learning Rate Decay')
        learning_rates = [online_kmeans.learning_rate * (online_kmeans.decay_factor ** t) 
                         for t in range(n_batches)]
        axes[0, 1].plot(range(n_batches), learning_rates, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Batch Number')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot early vs late data
        early_data = np.vstack(batch_data_history[:10])
        late_data = np.vstack(batch_data_history[-10:])
        
        axes[1, 0].scatter(early_data[:, 0], early_data[:, 1], alpha=0.6, s=20, label='Early batches')
        axes[1, 0].scatter(center_history[9, 0, 0], center_history[9, 0, 1], 
                          c='red', marker='x', s=100, linewidth=3, label='Early centers')
        axes[1, 0].scatter(center_history[9, 1, 0], center_history[9, 1, 1], 
                          c='red', marker='x', s=100, linewidth=3)
        axes[1, 0].set_title('Early Data (Batches 0-9)')
        axes[1, 0].legen#!/usr/bin/env python3