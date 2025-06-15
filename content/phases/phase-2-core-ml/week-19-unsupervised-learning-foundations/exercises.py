#!/usr/bin/env python3
"""
Neural Odyssey - Phase 2: Core Machine Learning
Week 19: Unsupervised Learning Foundations
Complete Exercise Implementation

This comprehensive module explores unsupervised learning from first principles,
covering pattern discovery, dimensionality reduction, clustering, and the mathematical
foundations that enable machines to find structure in unlabeled data.

Learning Path:
1. Pattern discovery without labels - the unsupervised learning paradigm
2. K-means clustering algorithm from mathematical foundations
3. Hierarchical clustering and dendrogram analysis
4. Principal Component Analysis (PCA) for dimensionality reduction
5. Gaussian Mixture Models and Expectation-Maximization
6. Density estimation and outlier detection
7. Real-world applications and modern extensions

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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
import seaborn as sns
from collections import defaultdict
import itertools

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

class UnsupervisedLearningFoundations:
    """
    Foundational concepts and utilities for unsupervised learning
    """
    
    @staticmethod
    def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    @staticmethod
    def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate cosine distance between two points"""
        dot_product = np.dot(x1, x2)
        norm_product = np.linalg.norm(x1) * np.linalg.norm(x2)
        if norm_product == 0:
            return 1.0
        return 1 - (dot_product / norm_product)
    
    @staticmethod
    def within_cluster_sum_of_squares(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (WCSS)"""
        wcss = 0
        for i in range(len(centers)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centers[i]) ** 2)
        return wcss
    
    @staticmethod
    def between_cluster_sum_of_squares(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate between-cluster sum of squares (BCSS)"""
        overall_center = np.mean(X, axis=0)
        bcss = 0
        for i in range(len(centers)):
            cluster_size = np.sum(labels == i)
            if cluster_size > 0:
                bcss += cluster_size * np.sum((centers[i] - overall_center) ** 2)
        return bcss

class KMeansFromScratch:
    """
    K-means clustering algorithm implemented from mathematical first principles
    
    Mathematical Foundation:
    - Objective: Minimize within-cluster sum of squares
    - Algorithm: Iterative optimization using Lloyd's algorithm
    - Convergence: Guaranteed for finite datasets
    """
    
    def __init__(self, 
                 n_clusters: int = 3,
                 max_iters: int = 300,
                 tol: float = 1e-4,
                 init: str = 'k-means++',
                 random_state: Optional[int] = None):
        """
        Initialize K-means clustering
        
        Parameters:
        -----------
        n_clusters : int, number of clusters
        max_iters : int, maximum number of iterations
        tol : float, tolerance for convergence
        init : str, initialization method ('random', 'k-means++')
        random_state : int, random seed
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        # Fitted attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.history_ = []
        
        if random_state:
            np.random.seed(random_state)
    
    def _initialize_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centers using specified method"""
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            return X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        elif self.init == 'k-means++':
            # K-means++ initialization for better convergence
            centers = []
            
            # Choose first center randomly
            centers.append(X[np.random.randint(n_samples)])
            
            # Choose remaining centers
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) 
                                    for x in X])
                probabilities = distances / np.sum(distances)
                cumulative_probabilities = np.cumsum(probabilities)
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centers.append(X[j])
                        break
            
            return np.array(centers)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
    
    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        """
        Fit K-means clustering to data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Initialize centers
        centers = self._initialize_centers(X)
        
        for iteration in range(self.max_iters):
            # Store previous centers for convergence check
            prev_centers = centers.copy()
            
            # Assign points to nearest cluster
            distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update cluster centers
            new_centers = []
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    # Keep old center if no points assigned
                    new_centers.append(centers[k])
            
            centers = np.array(new_centers)
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = UnsupervisedLearningFoundations.within_cluster_sum_of_squares(X, labels, centers)
            
            # Store history
            self.history_.append({
                'iteration': iteration,
                'centers': centers.copy(),
                'labels': labels.copy(),
                'inertia': inertia
            })
            
            # Check for convergence
            center_shift = np.sqrt(np.sum((centers - prev_centers) ** 2))
            if center_shift < self.tol:
                break
        
        # Store final results
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels"""
        self.fit(X)
        return self.labels_

class HierarchicalClustering:
    """
    Hierarchical clustering with multiple linkage criteria
    
    Builds a tree of clusters using bottom-up (agglomerative) approach
    """
    
    def __init__(self, 
                 linkage: str = 'ward',
                 distance_metric: str = 'euclidean'):
        """
        Initialize hierarchical clustering
        
        Parameters:
        -----------
        linkage : str, linkage criterion ('single', 'complete', 'average', 'ward')
        distance_metric : str, distance metric
        """
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.linkage_matrix_ = None
        self.distance_matrix_ = None
    
    def _calculate_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix"""
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if self.distance_metric == 'euclidean':
                    dist = UnsupervisedLearningFoundations.euclidean_distance(X[i], X[j])
                elif self.distance_metric == 'manhattan':
                    dist = UnsupervisedLearningFoundations.manhattan_distance(X[i], X[j])
                elif self.distance_metric == 'cosine':
                    dist = UnsupervisedLearningFoundations.cosine_distance(X[i], X[j])
                else:
                    raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                
                distances[i, j] = distances[j, i] = dist
        
        return distances
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """
        Fit hierarchical clustering
        
        Uses scipy's implementation for efficiency while maintaining educational value
        """
        X = np.array(X)
        
        # Calculate distance matrix
        self.distance_matrix_ = self._calculate_distance_matrix(X)
        
        # Perform hierarchical clustering using scipy
        if self.linkage == 'ward' and self.distance_metric == 'euclidean':
            self.linkage_matrix_ = linkage(X, method='ward')
        else:
            condensed_distances = pdist(X, metric=self.distance_metric)
            self.linkage_matrix_ = linkage(condensed_distances, method=self.linkage)
        
        return self
    
    def get_clusters(self, n_clusters: int) -> np.ndarray:
        """Get cluster labels for specified number of clusters"""
        if self.linkage_matrix_ is None:
            raise ValueError("Model must be fitted first")
        
        return fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust') - 1
    
    def plot_dendrogram(self, figsize: Tuple[int, int] = (12, 8), 
                       truncate_mode: Optional[str] = None,
                       max_d: Optional[float] = None):
        """Plot dendrogram of hierarchical clustering"""
        if self.linkage_matrix_ is None:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=figsize)
        
        dendrogram_kwargs = {'leaf_rotation': 90, 'leaf_font_size': 10}
        
        if truncate_mode:
            dendrogram_kwargs['truncate_mode'] = truncate_mode
            dendrogram_kwargs['p'] = 30
        
        if max_d:
            dendrogram_kwargs['color_threshold'] = max_d
        
        dendrogram(self.linkage_matrix_, **dendrogram_kwargs)
        
        plt.title(f'Hierarchical Clustering Dendrogram\n({self.linkage} linkage, {self.distance_metric} distance)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        if max_d:
            plt.axhline(y=max_d, color='red', linestyle='--', 
                       label=f'Cut threshold = {max_d:.2f}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis for dimensionality reduction
    
    Mathematical Foundation:
    - Find directions of maximum variance in data
    - Use eigendecomposition of covariance matrix
    - Project data onto principal components
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA
        
        Parameters:
        -----------
        n_components : int or None, number of components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray) -> 'PrincipalComponentAnalysis':
        """
        Fit PCA to data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(self.n_components, n_features)
        
        # Store results
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal component space"""
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space"""
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transform")
        
        return X_transformed @ self.components_ + self.mean_
    
    def explained_variance_ratio_cumsum(self) -> np.ndarray:
        """Return cumulative explained variance ratio"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fitted first")
        
        return np.cumsum(self.explained_variance_ratio_)

class GaussianMixtureModel:
    """
    Gaussian Mixture Model using Expectation-Maximization algorithm
    
    Mathematical Foundation:
    - Model data as mixture of Gaussian distributions
    - Use EM algorithm for parameter estimation
    - E-step: Calculate responsibilities (posterior probabilities)
    - M-step: Update parameters using weighted maximum likelihood
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 max_iters: int = 100,
                 tol: float = 1e-6,
                 random_state: Optional[int] = None):
        """
        Initialize Gaussian Mixture Model
        
        Parameters:
        -----------
        n_components : int, number of Gaussian components
        max_iters : int, maximum number of EM iterations
        tol : float, convergence tolerance
        random_state : int, random seed
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        # Model parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = None
        self.log_likelihood_history_ = []
        
        if random_state:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X: np.ndarray):
        """Initialize GMM parameters"""
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using k-means++
        self.means_ = []
        remaining_points = X.copy()
        
        # First mean
        self.means_.append(remaining_points[np.random.randint(len(remaining_points))])
        
        # Subsequent means
        for _ in range(1, self.n_components):
            distances = np.array([
                min([np.linalg.norm(x - mean)**2 for mean in self.means_]) 
                for x in remaining_points
            ])
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    self.means_.append(remaining_points[j])
                    break
        
        self.means_ = np.array(self.means_)
        
        # Initialize covariances as identity matrices
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def _multivariate_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate multivariate Gaussian probability density"""
        n_features = X.shape[1]
        diff = X - mean
        
        # Add small value to diagonal for numerical stability
        cov_reg = cov + np.eye(n_features) * 1e-6
        
        try:
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            inv_cov = np.linalg.pinv(cov_reg)
            det_cov = np.maximum(np.linalg.det(cov_reg), 1e-10)
        
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * det_cov)
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        
        return norm_const * np.exp(exponent)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Expectation step: calculate responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Calculate likelihood for each component
        for k in range(self.n_components):
            responsibilities[:, k] = (
                self.weights_[k] * 
                self._multivariate_gaussian_pdf(X, self.means_[k], self.covariances_[k])
            )
        
        # Normalize to get responsibilities
        total_likelihood = np.sum(responsibilities, axis=1, keepdims=True)
        total_likelihood = np.maximum(total_likelihood, 1e-10)  # Avoid division by zero
        responsibilities /= total_likelihood
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """Maximization step: update parameters"""
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        Nk = np.sum(responsibilities, axis=0)
        
        # Update weights
        self.weights_ = Nk / n_samples
        
        # Update means
        for k in range(self.n_components):
            if Nk[k] > 0:
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]
        
        # Update covariances
        for k in range(self.n_components):
            if Nk[k] > 0:
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / Nk[k]
                
                # Add regularization for numerical stability
                self.covariances_[k] += np.eye(n_features) * 1e-6
    
    def _calculate_log_likelihood(self, X: np.ndarray) -> float:
        """Calculate log-likelihood of data"""
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += (
                    self.weights_[k] * 
                    self._multivariate_gaussian_pdf(X[i:i+1], self.means_[k], self.covariances_[k])[0]
                )
            log_likelihood += np.log(np.maximum(sample_likelihood, 1e-10))
        
        return log_likelihood
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit Gaussian Mixture Model using EM algorithm
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.array(X)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iters):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(X)
            self.log_likelihood_history_.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break
            
            prev_log_likelihood = log_likelihood
        
        self.n_iter_ = iteration + 1
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities"""
        return self._e_step(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels"""
        self.fit(X)
        return self.predict(X)

class ClusteringEvaluation:
    """
    Comprehensive clustering evaluation metrics and methods
    """
    
    @staticmethod
    def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score manually"""
        n_samples = len(X)
        if len(np.unique(labels)) == 1:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # Calculate a(i): mean distance to points in same cluster
            same_cluster_mask = labels == labels[i]
            same_cluster_points = X[same_cluster_mask]
            
            if len(same_cluster_points) > 1:
                a_i = np.mean([UnsupervisedLearningFoundations.euclidean_distance(X[i], point) 
                              for point in same_cluster_points if not np.array_equal(point, X[i])])
            else:
                a_i = 0
            
            # Calculate b(i): min mean distance to points in other clusters
            b_i = np.inf
            for cluster in np.unique(labels):
                if cluster != labels[i]:
                    other_cluster_points = X[labels == cluster]
                    if len(other_cluster_points) > 0:
                        mean_dist = np.mean([UnsupervisedLearningFoundations.euclidean_distance(X[i], point) 
                                           for point in other_cluster_points])
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette score for this point
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)
    
    @staticmethod
    def elbow_method(X: np.ndarray, max_k: int = 10, 
                    algorithm: str = 'kmeans') -> Tuple[List[int], List[float]]:
        """
        Perform elbow method to find optimal number of clusters
        
        Returns:
        --------
        k_values : list of k values tested
        wcss_values : list of within-cluster sum of squares
        """
        k_values = list(range(1, max_k + 1))
        wcss_values = []
        
        for k in k_values:
            if algorithm == 'kmeans':
                if k == 1:
                    # For k=1, WCSS is total variance
                    center = np.mean(X, axis=0)
                    wcss = np.sum((X - center) ** 2)
                else:
                    kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
                    kmeans.fit(X)
                    wcss = kmeans.inertia_
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            wcss_values.append(wcss)
        
        return k_values, wcss_values
    
    @staticmethod
    def plot_clustering_metrics(X: np.ndarray, max_k: int = 10, 
                               figsize: Tuple[int, int] = (15, 5)):
        """Plot various clustering evaluation metrics"""
        k_values = list(range(2, max_k + 1))
        silhouette_scores = []
        calinski_scores = []
        
        # Calculate elbow method
        elbow_k, elbow_wcss = ClusteringEvaluation.elbow_method(X, max_k)
        
        # Calculate other metrics
        for k in k_values:
            kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Silhouette score
            sil_score = ClusteringEvaluation.silhouette_score_manual(X, labels)
            silhouette_scores.append(sil_score)
            
            # Calinski-Harabasz score (using sklearn for efficiency)
            ch_score = calinski_harabasz_score(X, labels)
            calinski_scores.append(ch_score)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Elbow method
        ax1.plot(elbow_k, elbow_wcss, 'bo-')
        ax1.set_xlabel('Number of clusters (k)')
        ax1.set_ylabel('Within-cluster sum of squares')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score
        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Calinski-Harabasz score
        ax3.plot(k_values, calinski_scores, 'go-')
        ax3.set_xlabel('Number of clusters (k)')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Index')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'elbow': (elbow_k, elbow_wcss),
            'silhouette': (k_values, silhouette_scores),
            'calinski_harabasz': (k_values, calinski_scores)
        }

class UnsupervisedLearningExperiments:
    """Comprehensive unsupervised learning experiments and case studies"""
    
    @staticmethod
    def experiment_1_kmeans_fundamentals():
        """Experiment 1: K-means fundamentals and algorithm behavior"""
        print("="*70)
        print("EXPERIMENT 1: K-MEANS FUNDAMENTALS AND ALGORITHM BEHAVIOR")
        print("="*70)
        
        # Generate synthetic datasets with different characteristics
        np.random.seed(42)
        
        # Dataset 1: Well-separated spherical clusters
        X1, y1_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, 
                                 center_box=(-10.0, 10.0), random_state=42)
        
        # Dataset 2: Different cluster sizes
        cluster_centers = np.array([[-5, -5], [0, 0], [5, 5]])
        X2 = np.vstack([
            np.random.multivariate_normal(cluster_centers[0], [[1, 0], [0, 1]], 50),
            np.random.multivariate_normal(cluster_centers[1], [[1, 0], [0, 1]], 150),
            np.random.multivariate_normal(cluster_centers[2], [[1, 0], [0, 1]], 100)
        ])
        
        # Dataset 3: Non-spherical clusters
        X3, y3_true = make_moons(n_samples=300, noise=0.1, random_state=42)
        
        datasets = [
            (X1, "Well-separated spherical clusters"),
            (X2, "Different cluster sizes"),
            (X3, "Non-spherical clusters")
        ]
        
        print("\n1.1 K-MEANS ON DIFFERENT DATASET TYPES")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (X, title) in enumerate(datasets):
            # Original data
            axes[0, idx].scatter(X[:, 0], X[:, 1], alpha=0.7, s=50)
            axes[0, idx].set_title(f'Original Data: {title}')
            axes[0, idx].grid(True, alpha=0.3)
            
            # K-means clustering
            k = 3 if idx == 2 else 4
            if idx == 2:  # For moons, use k=2
                k = 2
            
            kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Plot clustered data
            colors = plt.cm.viridis(np.linspace(0, 1, k))
            for i in range(k):
                cluster_points = X[labels == i]
                axes[1, idx].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                   c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {i}')
            
            # Plot centroids
            axes[1, idx].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                               c='red', marker='x', s=200, linewidths=3, label='Centroids')
            
            axes[1, idx].set_title(f'K-means Result (k={k})')
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
            
            print(f"Dataset {idx+1}: {title}")
            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.2f}, Iterations: {kmeans.n_iter_}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n1.2 ALGORITHM CONVERGENCE ANALYSIS")
        print("-" * 50)
        
        # Analyze convergence for well-separated clusters
        kmeans_detailed = KMeansFromScratch(n_clusters=4, random_state=42)
        kmeans_detailed.fit(X1)
        
        # Plot convergence
        iterations = [h['iteration'] for h in kmeans_detailed.history_]
        inertias = [h['inertia'] for h in kmeans_detailed.history_]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, inertias, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (WCSS)')
        plt.title('K-means Convergence')
        plt.grid(True, alpha=0.3)
        
        # Show cluster evolution
        plt.subplot(1, 2, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, 4))
        
        # Plot final clusters
        final_labels = kmeans_detailed.labels_
        for i in range(4):
            cluster_points = X1[final_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], alpha=0.7, s=50)
        
        # Show centroid evolution
        for i, history in enumerate(kmeans_detailed.history_[::2]):  # Every other iteration
            centers = history['centers']
            plt.scatter(centers[:, 0], centers[:, 1], 
                       c='red', marker='x', s=100, alpha=0.3 + 0.7*i/len(kmeans_detailed.history_))
        
        # Final centroids
        plt.scatter(kmeans_detailed.cluster_centers_[:, 0], kmeans_detailed.cluster_centers_[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Final Centroids')
        
        plt.title('Centroid Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Convergence achieved in {kmeans_detailed.n_iter_} iterations")
        print(f"Final inertia: {kmeans_detailed.inertia_:.2f}")
        
        return {
            'datasets': datasets,
            'kmeans_results': [kmeans_detailed],
            'convergence_history': kmeans_detailed.history_
        }
    
    @staticmethod
    def experiment_2_hierarchical_clustering():
        """Experiment 2: Hierarchical clustering and dendrogram analysis"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: HIERARCHICAL CLUSTERING AND DENDROGRAM ANALYSIS")
        print("="*70)
        
        # Generate hierarchical dataset
        np.random.seed(42)
        
        # Create nested clusters
        # Main cluster 1
        cluster1a = np.random.multivariate_normal([-6, -6], [[1, 0], [0, 1]], 30)
        cluster1b = np.random.multivariate_normal([-4, -4], [[1, 0], [0, 1]], 30)
        
        # Main cluster 2
        cluster2a = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 30)
        cluster2b = np.random.multivariate_normal([4, 4], [[1, 0], [0, 1]], 30)
        
        # Outliers
        outliers = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 20)
        
        X_hier = np.vstack([cluster1a, cluster1b, cluster2a, cluster2b, outliers])
        
        print("\n2.1 DIFFERENT LINKAGE CRITERIA COMPARISON")
        print("-" * 50)
        
        linkage_methods = ['single', 'complete', 'average', 'ward']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        hierarchical_results = {}
        
        for idx, linkage_method in enumerate(linkage_methods):
            # Fit hierarchical clustering
            if linkage_method == 'ward':
                hc = HierarchicalClustering(linkage=linkage_method, distance_metric='euclidean')
            else:
                hc = HierarchicalClustering(linkage=linkage_method, distance_metric='euclidean')
            
            hc.fit(X_hier)
            hierarchical_results[linkage_method] = hc
            
            # Plot dendrogram
            plt.subplot(2, 4, idx + 1)
            dendrogram(hc.linkage_matrix_, truncate_mode='level', p=3)
            plt.title(f'{linkage_method.capitalize()} Linkage Dendrogram')
            plt.ylabel('Distance')
            
            # Plot clustering result
            plt.subplot(2, 4, idx + 5)
            labels = hc.get_clusters(4)
            
            colors = plt.cm.viridis(np.linspace(0, 1, 4))
            for i in range(4):
                cluster_points = X_hier[labels == i]
                if len(cluster_points) > 0:
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                               c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {i}')
            
            plt.title(f'{linkage_method.capitalize()} Linkage (4 clusters)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            print(f"{linkage_method.capitalize()} linkage: 4 clusters identified")
        
        plt.tight_layout()
        plt.show()
        
        print("\n2.2 OPTIMAL NUMBER OF CLUSTERS ANALYSIS")
        print("-" * 50)
        
        # Use Ward linkage for detailed analysis
        ward_hc = hierarchical_results['ward']
        
        # Test different numbers of clusters
        cluster_counts = range(2, 8)
        silhouette_scores = []
        
        for n_clusters in cluster_counts:
            labels = ward_hc.get_clusters(n_clusters)
            if len(np.unique(labels)) > 1:
                sil_score = ClusteringEvaluation.silhouette_score_manual(X_hier, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_counts, silhouette_scores, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Hierarchical Clustering: Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_k = cluster_counts[np.argmax(silhouette_scores)]
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        plt.legend()
        plt.show()
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return {
            'data': X_hier,
            'hierarchical_results': hierarchical_results,
            'optimal_clusters': optimal_k,
            'silhouette_scores': dict(zip(cluster_counts, silhouette_scores))
        }
    
    @staticmethod
    def experiment_3_pca_dimensionality_reduction():
        """Experiment 3: PCA for dimensionality reduction and visualization"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: PCA FOR DIMENSIONALITY REDUCTION AND VISUALIZATION")
        print("="*70)
        
        # Load high-dimensional dataset
        digits = load_digits()
        X_digits = digits.data  # 8x8 pixel images flattened to 64 features
        y_digits = digits.target
        
        print(f"Original dataset shape: {X_digits.shape}")
        print(f"Number of classes: {len(np.unique(y_digits))}")
        
        print("\n3.1 PCA ANALYSIS AND COMPONENT SELECTION")
        print("-" * 50)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_digits)
        
        # Fit PCA with all components
        pca_full = PrincipalComponentAnalysis()
        pca_full.fit(X_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                pca_full.explained_variance_ratio_, 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        cumsum_ratio = pca_full.explained_variance_ratio_cumsum()
        plt.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'o-')
        plt.axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
        plt.axhline(y=0.99, color='orange', linestyle='--', label='99% Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Find components for 95% variance
        n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
        n_components_99 = np.argmax(cumsum_ratio >= 0.99) + 1
        
        print(f"Components for 95% variance: {n_components_95}")
        print(f"Components for 99% variance: {n_components_99}")
        
        # Principal component visualization
        plt.subplot(1, 3, 3)
        
        # Show first few principal components as images
        n_components_viz = 6
        components_reshaped = pca_full.components_[:n_components_viz].reshape(n_components_viz, 8, 8)
        
        for i in range(n_components_viz):
            plt.subplot(2, 3, i + 1)
            plt.imshow(components_reshaped[i], cmap='RdBu_r')
            plt.title(f'PC {i+1}\n({pca_full.explained_variance_ratio_[i]:.1%} var)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n3.2 DIMENSIONALITY REDUCTION AND VISUALIZATION")
        print("-" * 50)
        
        # Apply PCA with different numbers of components
        components_to_test = [2, 10, 20, n_components_95]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        pca_results = {}
        
        for idx, n_comp in enumerate(components_to_test):
            pca = PrincipalComponentAnalysis(n_components=n_comp)
            X_reduced = pca.fit_transform(X_scaled)
            pca_results[n_comp] = pca
            
            if n_comp == 2:
                # 2D visualization
                scatter = axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                          c=y_digits, cmap='tab10', alpha=0.7, s=20)
                axes[idx].set_xlabel('First Principal Component')
                axes[idx].set_ylabel('Second Principal Component')
                axes[idx].set_title(f'PCA to 2D\n({pca.explained_variance_ratio_cumsum()[-1]:.1%} variance retained)')
                plt.colorbar(scatter, ax=axes[idx])
            else:
                # Reconstruction quality
                X_reconstructed = pca.inverse_transform(X_reduced)
                
                # Show original vs reconstructed
                sample_idx = 0
                original = X_scaled[sample_idx].reshape(8, 8)
                reconstructed = X_reconstructed[sample_idx].reshape(8, 8)
                
                # Plot side by side
                combined = np.hstack([original, reconstructed])
                axes[idx].imshow(combined, cmap='gray')
                axes[idx].set_title(f'{n_comp} Components\n({pca.explained_variance_ratio_cumsum()[-1]:.1%} variance)')
                axes[idx].axis('off')
                
                # Calculate reconstruction error
                mse = np.mean((X_scaled - X_reconstructed) ** 2)
                print(f"  {n_comp} components: {pca.explained_variance_ratio_cumsum()[-1]:.1%} variance, MSE: {mse:.4f}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n3.3 PCA FOR NOISE REDUCTION")
        print("-" * 50)
        
        # Add noise to dataset
        noise_level = 0.5
        X_noisy = X_scaled + np.random.normal(0, noise_level, X_scaled.shape)
        
        # Apply PCA for denoising
        pca_denoise = PrincipalComponentAnalysis(n_components=n_components_95)
        X_denoised = pca_denoise.fit_transform(X_noisy)
        X_reconstructed = pca_denoise.inverse_transform(X_denoised)
        
        # Visualize denoising effect
        plt.figure(figsize=(15, 5))
        
        sample_indices = [0, 1, 2]
        for i, idx in enumerate(sample_indices):
            # Original
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(X_scaled[idx].reshape(8, 8), cmap='gray')
            plt.title(f'Original {idx}')
            plt.axis('off')
            
            # Noisy
            plt.subplot(3, 3, i*3 + 2)
            plt.imshow(X_noisy[idx].reshape(8, 8), cmap='gray')
            plt.title(f'Noisy {idx}')
            plt.axis('off')
            
            # Denoised
            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(X_reconstructed[idx].reshape(8, 8), cmap='gray')
            plt.title(f'Denoised {idx}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate denoising performance
        original_mse = np.mean((X_scaled - X_noisy) ** 2)
        denoised_mse = np.mean((X_scaled - X_reconstructed) ** 2)
        
        print(f"Original vs Noisy MSE: {original_mse:.4f}")
        print(f"Original vs Denoised MSE: {denoised_mse:.4f}")
        print(f"Noise reduction: {((original_mse - denoised_mse) / original_mse * 100):.1f}%")
        
        return {
            'data': (X_digits, y_digits, X_scaled),
            'pca_full': pca_full,
            'pca_results': pca_results,
            'optimal_components': {
                '95_percent': n_components_95,
                '99_percent': n_components_99
            },
            'denoising_results': {
                'original_mse': original_mse,
                'denoised_mse': denoised_mse
            }
        }
    
    @staticmethod
    def experiment_4_gaussian_mixture_models():
        """Experiment 4: Gaussian Mixture Models and EM algorithm"""
        print("\n" + "="*70)
        print("EXPERIMENT 4: GAUSSIAN MIXTURE MODELS AND EM ALGORITHM")
        print("="*70)
        
        # Generate complex dataset with overlapping clusters
        np.random.seed(42)
        
        # Create overlapping Gaussian clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 150)
        cluster2 = np.random.multivariate_normal([6, 6], [[1, -0.5], [-0.5, 1]], 100)
        cluster3 = np.random.multivariate_normal([2, 6], [[2, 0], [0, 0.5]], 80)
        
        X_gmm = np.vstack([cluster1, cluster2, cluster3])
        true_labels = np.hstack([np.zeros(150), np.ones(100), np.full(80, 2)])
        
        print(f"Dataset shape: {X_gmm.shape}")
        print(f"True number of clusters: 3")
        
        print("\n4.1 GMM FITTING AND EM ALGORITHM CONVERGENCE")
        print("-" * 50)
        
        # Fit GMM with known number of components
        gmm = GaussianMixtureModel(n_components=3, random_state=42)
        predicted_labels = gmm.fit_predict(X_gmm)
        
        print(f"EM algorithm converged: {gmm.converged_}")
        print(f"Number of iterations: {gmm.n_iter_}")
        print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.2f}")
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original data with true labels
        axes[0, 0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('True Clusters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # GMM predicted labels
        axes[0, 1].scatter(X_gmm[:, 0], X_gmm[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
        
        # Plot Gaussian contours
        x_min, x_max = X_gmm[:, 0].min() - 1, X_gmm[:, 0].max() + 1
        y_min, y_max = X_gmm[:, 1].min() - 1, X_gmm[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        for i in range(gmm.n_components):
            # Plot 2-sigma ellipse for each component
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]
            
            # Eigendecomposition for ellipse
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(2 * eigenvals)  # 2-sigma
            
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(mean, width, height, angle=angle, 
                            facecolor='none', edgecolor='red', alpha=0.7, linewidth=2)
            axes[0, 1].add_patch(ellipse)
            
            # Mark centers
            axes[0, 1].scatter(mean[0], mean[1], c='red', marker='x', s=100, linewidth=3)
        
        axes[0, 1].set_title('GMM Predicted Clusters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log-likelihood convergence
        axes[1, 0].plot(gmm.log_likelihood_history_, 'o-')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Log-likelihood')
        axes[1, 0].set_title('EM Algorithm Convergence')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Component weights and parameters
        axes[1, 1].bar(range(gmm.n_components), gmm.weights_)
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Mixture Weight')
        axes[1, 1].set_title('Component Weights')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print component parameters
        print("\nLearned Component Parameters:")
        for i in range(gmm.n_components):
            print(f"Component {i}:")
            print(f"  Weight: {gmm.weights_[i]:.3f}")
            print(f"  Mean: {gmm.means_[i]}")
            print(f"  Covariance: {gmm.covariances_[i]}")
        
        print("\n4.2 MODEL SELECTION: FINDING OPTIMAL NUMBER OF COMPONENTS")
        print("-" * 50)
        
        # Test different numbers of components
        component_range = range(1, 8)
        aic_scores = []
        bic_scores = []
        log_likelihoods = []
        
        for n_comp in component_range:
            gmm_test = GaussianMixtureModel(n_components=n_comp, random_state=42)
            gmm_test.fit(X_gmm)
            
            log_likelihood = gmm_test.log_likelihood_history_[-1]
            n_params = n_comp * (1 + X_gmm.shape[1] + X_gmm.shape[1] * (X_gmm.shape[1] + 1) // 2) - 1
            
            # AIC and BIC
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(len(X_gmm))
            
            aic_scores.append(aic)
            bic_scores.append(bic)
            log_likelihoods.append(log_likelihood)
            
            print(f"Components: {n_comp}, Log-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        # Plot model selection criteria
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(component_range, log_likelihoods, 'o-')
        plt.xlabel('Number of Components')
        plt.ylabel('Log-likelihood')
        plt.title('Log-likelihood vs Components')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(component_range, aic_scores, 'o-', label='AIC')
        optimal_aic = component_range[np.argmin(aic_scores)]
        plt.axvline(x=optimal_aic, color='red', linestyle='--', label=f'Optimal AIC: {optimal_aic}')
        plt.xlabel('Number of Components')
        plt.ylabel('AIC')
        plt.title('AIC vs Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(component_range, bic_scores, 'o-', label='BIC')
        optimal_bic = component_range[np.argmin(bic_scores)]
        plt.axvline(x=optimal_bic, color='red', linestyle='--', label=f'Optimal BIC: {optimal_bic}')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC')
        plt.title('BIC vs Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nOptimal number of components:")
        print(f"  AIC: {optimal_aic}")
        print(f"  BIC: {optimal_bic}")
        
        print("\n4.3 SOFT CLUSTERING: PROBABILITY ASSIGNMENTS")
        print("-" * 50)
        
        # Show soft cluster assignments
        probabilities = gmm.predict_proba(X_gmm)
        
        # Plot probability distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i in range(gmm.n_components):
            scatter = axes[i].scatter(X_gmm[:, 0], X_gmm[:, 1], 
                                    c=probabilities[:, i], cmap='Reds', alpha=0.7)
            axes[i].set_title(f'Component {i} Probabilities')
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Show uncertainty quantification
        max_probs = np.max(probabilities, axis=1)
        uncertainty = 1 - max_probs
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=max_probs, cmap='viridis', alpha=0.7)
        plt.title('Maximum Assignment Probability')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=uncertainty, cmap='Reds', alpha=0.7)
        plt.title('Assignment Uncertainty (1 - max prob)')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Average assignment certainty: {np.mean(max_probs):.3f}")
        print(f"Points with high uncertainty (< 0.6 certainty): {np.sum(max_probs < 0.6)}")
        
        return {
            'data': (X_gmm, true_labels),
            'gmm_model': gmm,
            'predicted_labels': predicted_labels,
            'model_selection': {
                'aic_scores': aic_scores,
                'bic_scores': bic_scores,
                'optimal_aic': optimal_aic,
                'optimal_bic': optimal_bic
            },
            'probabilities': probabilities,
            'uncertainty': uncertainty
        }
    
    @staticmethod
    def experiment_5_clustering_evaluation():
        """Experiment 5: Comprehensive clustering evaluation metrics"""
        print("\n" + "="*70)
        print("EXPERIMENT 5: COMPREHENSIVE CLUSTERING EVALUATION METRICS")
        print("="*70)
        
        # Generate datasets with known ground truth
        np.random.seed(42)
        
        datasets = []
        
        # Well-separated clusters
        X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=42)
        datasets.append((X1, y1, "Well-separated"))
        
        # Overlapping clusters
        X2, y2 = make_blobs(n_samples=300, centers=3, cluster_std=3.0, random_state=42)
        datasets.append((X2, y2, "Overlapping"))
        
        # Different densities
        X3 = np.vstack([
            np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
            np.random.multivariate_normal([6, 6], [[0.2, 0], [0, 0.2]], 50),
            np.random.multivariate_normal([0, 6], [[3, 0], [0, 3]], 150)
        ])
        y3 = np.hstack([np.zeros(100), np.ones(50), np.full(150, 2)])
        datasets.append((X3, y3, "Different densities"))
        
        print("\n5.1 CLUSTERING ALGORITHMS COMPARISON")
        print("-" * 50)
        
        algorithms = [
            ("K-means", lambda X, k: KMeansFromScratch(n_clusters=k, random_state=42).fit_predict(X)),
            ("GMM", lambda X, k: GaussianMixtureModel(n_components=k, random_state=42).fit_predict(X)),
        ]
        
        results_summary = []
        
        fig, axes = plt.subplots(len(datasets), len(algorithms) + 1, figsize=(15, 12))
        
        for dataset_idx, (X, y_true, dataset_name) in enumerate(datasets):
            n_clusters = len(np.unique(y_true))
            
            # Plot true clusters
            axes[dataset_idx, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
            axes[dataset_idx, 0].set_title(f'{dataset_name}\n(True clusters)')
            axes[dataset_idx, 0].grid(True, alpha=0.3)
            
            dataset_results = {'dataset': dataset_name, 'true_clusters': n_clusters}
            
            for alg_idx, (alg_name, alg_func) in enumerate(algorithms):
                # Apply clustering algorithm
                y_pred = alg_func(X, n_clusters)
                
                # Calculate metrics
                if len(np.unique(y_pred)) > 1:
                    ari = adjusted_rand_score(y_true, y_pred)
                    silhouette = ClusteringEvaluation.silhouette_score_manual(X, y_pred)
                    calinski = calinski_harabasz_score(X, y_pred)
                else:
                    ari = silhouette = calinski = 0.0
                
                dataset_results[alg_name] = {
                    'ari': ari,
                    'silhouette': silhouette,
                    'calinski': calinski
                }
                
                # Plot predicted clusters
                axes[dataset_idx, alg_idx + 1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                axes[dataset_idx, alg_idx + 1].set_title(f'{alg_name}\nARI: {ari:.3f}, Sil: {silhouette:.3f}')
                axes[dataset_idx, alg_idx + 1].grid(True, alpha=0.3)
                
                print(f"{dataset_name} - {alg_name}:")
                print(f"  ARI: {ari:.3f}, Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}")
            
            results_summary.append(dataset_results)
        
        plt.tight_layout()
        plt.show()
        
        print("\n5.2 ELBOW METHOD AND OPTIMAL K SELECTION")
        print("-" * 50)
        
        # Demonstrate elbow method on well-separated data
        X_elbow = datasets[0][0]
        
        metrics_results = ClusteringEvaluation.plot_clustering_metrics(X_elbow, max_k=10)
        
        # Find elbow point using second derivative
        k_values, wcss_values = metrics_results['elbow']
        
        # Calculate second derivatives for elbow detection
        if len(wcss_values) >= 3:
            second_derivatives = []
            for i in range(1, len(wcss_values) - 1):
                second_deriv = wcss_values[i-1] - 2*wcss_values[i] + wcss_values[i+1]
                second_derivatives.append(second_deriv)
            
            elbow_k = k_values[np.argmax(second_derivatives) + 1]
            print(f"Elbow method suggests k = {elbow_k}")
        
        print("\n5.3 INTERNAL vs EXTERNAL VALIDATION")
        print("-" * 50)
        
        # Compare internal and external metrics
        X_compare, y_compare = datasets[0][0], datasets[0][1]
        true_k = len(np.unique(y_compare))
        
        test_k_values = range(2, 8)
        internal_metrics = {'silhouette': [], 'calinski': [], 'wcss': []}
        external_metrics = {'ari': [], 'nmi': []}
        
        for k in test_k_values:
            # K-means clustering
            kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_compare)
            
            # Internal metrics (don't use ground truth)
            if len(np.unique(labels)) > 1:
                sil = ClusteringEvaluation.silhouette_score_manual(X_compare, labels)
                cal = calinski_harabasz_score(X_compare, labels)
            else:
                sil = cal = 0
            
            internal_metrics['silhouette'].append(sil)
            internal_metrics['calinski'].append(cal)
            internal_metrics['wcss'].append(kmeans.inertia_)
            
            # External metrics (use ground truth)
            ari = adjusted_rand_score(y_compare, labels)
            from sklearn.metrics import normalized_mutual_info_score
            nmi = normalized_mutual_info_score(y_compare, labels)
            
            external_metrics['ari'].append(ari)
            external_metrics['nmi'].append(nmi)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Internal metrics
        axes[0, 0].plot(test_k_values, internal_metrics['silhouette'], 'o-', label='Silhouette')
        axes[0, 0].axvline(x=true_k, color='red', linestyle='--', label=f'True k={true_k}')
        axes[0, 0].set_xlabel('Number of clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Internal Validation: Silhouette')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(test_k_values, internal_metrics['calinski'], 'o-', label='Calinski-Harabasz')
        axes[0, 1].axvline(x=true_k, color='red', linestyle='--', label=f'True k={true_k}')
        axes[0, 1].set_xlabel('Number of clusters')
        axes[0, 1].set_ylabel('Calinski-Harabasz Score')
        axes[0, 1].set_title('Internal Validation: Calinski-Harabasz')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # External metrics
        axes[1, 0].plot(test_k_values, external_metrics['ari'], 'o-', label='ARI')
        axes[1, 0].axvline(x=true_k, color='red', linestyle='--', label=f'True k={true_k}')
        axes[1, 0].set_xlabel('Number of clusters')
        axes[1, 0].set_ylabel('Adjusted Rand Index')
        axes[1, 0].set_title('External Validation: ARI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(test_k_values, external_metrics['nmi'], 'o-', label='NMI')
        axes[1, 1].axvline(x=true_k, color='red', linestyle='--', label=f'True k={true_k}')
        axes[1, 1].set_xlabel('Number of clusters')
        axes[1, 1].set_ylabel('Normalized Mutual Information')
        axes[1, 1].set_title('External Validation: NMI')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k according to each metric
        optimal_k_internal = {
            'silhouette': test_k_values[np.argmax(internal_metrics['silhouette'])],
            'calinski': test_k_values[np.argmax(internal_metrics['calinski'])]
        }
        
        optimal_k_external = {
            'ari': test_k_values[np.argmax(external_metrics['ari'])],
            'nmi': test_k_values[np.argmax(external_metrics['nmi'])]
        }
        
        print(f"True number of clusters: {true_k}")
        print(f"Optimal k by internal metrics:")
        for metric, k in optimal_k_internal.items():
            print(f"  {metric}: {k}")
        print(f"Optimal k by external metrics:")
        for metric, k in optimal_k_external.items():
            print(f"  {metric}: {k}")
        
        return {
            'datasets': datasets,
            'results_summary': results_summary,
            'metrics_comparison': {
                'internal': internal_metrics,
                'external': external_metrics,
                'optimal_k_internal': optimal_k_internal,
                'optimal_k_external': optimal_k_external
            }
        }
    
    @staticmethod
    def experiment_6_real_world_applications():
        """Experiment 6: Real-world unsupervised learning applications"""
        print("\n" + "="*70)
        print("EXPERIMENT 6: REAL-WORLD UNSUPERVISED LEARNING APPLICATIONS")
        print("="*70)
        
        # Load Iris dataset for customer segmentation simulation
        iris = load_iris()
        X_iris = iris.data
        y_iris_true = iris.target
        feature_names = iris.feature_names
        
        print(f"Dataset: Iris (simulating customer features)")
        print(f"Features: {feature_names}")
        print(f"Samples: {X_iris.shape[0]}, Features: {X_iris.shape[1]}")
        
        print("\n6.1 CUSTOMER SEGMENTATION PIPELINE")
        print("-" * 50)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_iris)
        
        # Apply PCA for dimensionality reduction and visualization
        pca = PrincipalComponentAnalysis(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        # Apply multiple clustering algorithms
        algorithms = {
            'K-means': KMeansFromScratch(n_clusters=3, random_state=42),
            'GMM': GaussianMixtureModel(n_components=3, random_state=42),
        }
        
        clustering_results = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original data in PCA space
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris_true, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('True Species (PCA space)')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Apply clustering algorithms
        for idx, (name, algorithm) in enumerate(algorithms.items()):
            labels = algorithm.fit_predict(X_scaled)
            clustering_results[name] = labels
            
            # Plot in PCA space
            axes[0, idx + 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
            axes[0, idx + 1].set_title(f'{name} Clustering')
            axes[0, idx + 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[0, idx + 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[0, idx + 1].grid(True, alpha=0.3)
            
            # Calculate metrics
            ari = adjusted_rand_score(y_iris_true, labels)
            silhouette = silhouette_score(X_scaled, labels)
            
            print(f"{name}: ARI = {ari:.3f}, Silhouette = {silhouette:.3f}")
        
        # Feature importance analysis
        axes[1, 0].bar(range(len(feature_names)), pca.explained_variance_ratio_)
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Explained Variance Ratio')
        axes[1, 0].set_title('PCA Component Importance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster profiles for K-means
        kmeans_labels = clustering_results['K-means']
        cluster_profiles = []
        
        for cluster in range(3):
            cluster_mask = kmeans_labels == cluster
            cluster_mean = np.mean(X_iris[cluster_mask], axis=0)
            cluster_profiles.append(cluster_mean)
        
        cluster_profiles = np.array(cluster_profiles)
        
        # Plot cluster profiles
        axes[1, 1].imshow(cluster_profiles.T, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Feature')
        axes[1, 1].set_title('Cluster Profiles (Means)')
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels([name.replace(' (cm)', '') for name in feature_names])
        
        # Add text annotations
        for i in range(3):
            for j in range(len(feature_names)):
                axes[1, 1].text(i, j, f'{cluster_profiles[i, j]:.1f}', 
                               ha='center', va='center', color='white' if abs(cluster_profiles[i, j] - np.mean(cluster_profiles[:, j])) > np.std(cluster_profiles[:, j]) else 'black')
        
        # Cluster sizes
        cluster_sizes = [np.sum(kmeans_labels == i) for i in range(3)]
        axes[1, 2].pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(3)], autopct='%1.1f%%')
        axes[1, 2].set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.show()
        
        print("\n6.2 ANOMALY DETECTION WITH CLUSTERING")
        print("-" * 50)
        
        # Use clustering for anomaly detection
        # Points far from cluster centers are potential anomalies
        
        kmeans = algorithms['K-means']
        
        # Calculate distances to nearest cluster center
        distances_to_centers = []
        for i, point in enumerate(X_scaled):
            cluster = kmeans_labels[i]
            center = kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(point - center)
            distances_to_centers.append(distance)
        
        distances_to_centers = np.array(distances_to_centers)
        
        # Define anomalies as points beyond 95th percentile of distances
        anomaly_threshold = np.percentile(distances_to_centers, 95)
        anomalies = distances_to_centers > anomaly_threshold
        
        plt.figure(figsize=(15, 5))
        
        # Distance distribution
        plt.subplot(1, 3, 1)
        plt.hist(distances_to_centers, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=anomaly_threshold, color='red', linestyle='--', 
                   label=f'95th percentile: {anomaly_threshold:.3f}')
        plt.xlabel('Distance to Cluster Center')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Anomalies in PCA space
        plt.subplot(1, 3, 2)
        plt.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], 
                   c=kmeans_labels[~anomalies], cmap='viridis', alpha=0.7, s=50, label='Normal')
        plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], 
                   c='red', marker='x', s=100, label='Anomalies')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature space anomalies
        plt.subplot(1, 3, 3)
        feature_pair = [0, 2]  # Sepal length vs Petal length
        plt.scatter(X_iris[~anomalies, feature_pair[0]], X_iris[~anomalies, feature_pair[1]], 
                   c=kmeans_labels[~anomalies], cmap='viridis', alpha=0.7, s=50, label='Normal')
        plt.scatter(X_iris[anomalies, feature_pair[0]], X_iris[anomalies, feature_pair[1]], 
                   c='red', marker='x', s=100, label='Anomalies')
        plt.xlabel(feature_names[feature_pair[0]])
        plt.ylabel(feature_names[feature_pair[1]])
        plt.title('Anomalies in Feature Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_iris)} samples ({np.sum(anomalies)/len(X_iris)*100:.1f}%)")
        
        if np.sum(anomalies) > 0:
            print("Anomalous samples:")
            anomaly_indices = np.where(anomalies)[0]
            for idx in anomaly_indices:
                print(f"  Sample {idx}: {X_iris[idx]} (True class: {iris.target_names[y_iris_true[idx]]})")
        
        print("\n6.3 MARKET BASKET ANALYSIS SIMULATION")
        print("-" * 50)
        
        # Simulate transaction data using Iris features as product purchases
        # Convert continuous features to binary purchases using thresholds
        
        # Use median as threshold for binary conversion
        thresholds = np.median(X_iris, axis=0)
        X_binary = (X_iris > thresholds).astype(int)
        
        product_names = ['SepalLength_High', 'SepalWidth_High', 'PetalLength_High', 'PetalWidth_High']
        
        print("Simulated Market Basket Data:")
        print("Products:", product_names)
        print("Sample transactions:")
        for i in range(5):
            purchased = [product_names[j] for j in range(len(product_names)) if X_binary[i, j] == 1]
            print(f"  Transaction {i}: {purchased}")
        
        # Apply clustering to find customer segments
        hierarchical = HierarchicalClustering(linkage='average', distance_metric='manhattan')
        hierarchical.fit(X_binary)
        
        # Get 3 segments
        segments = hierarchical.get_clusters(3)
        
        # Analyze segment profiles
        segment_profiles = []
        for segment in range(3):
            segment_mask = segments == segment
            segment_profile = np.mean(X_binary[segment_mask], axis=0)
            segment_profiles.append(segment_profile)
            
            print(f"\nSegment {segment} ({np.sum(segment_mask)} customers):")
            for j, product in enumerate(product_names):
                print(f"  {product}: {segment_profile[j]:.2%} purchase rate")
        
        # Visualize segments
        plt.figure(figsize=(12, 8))
        
        # Segment profiles heatmap
        plt.subplot(2, 2, 1)
        segment_profiles = np.array(segment_profiles)
        im = plt.imshow(segment_profiles.T, cmap='RdYlBu_r', aspect='auto')
        plt.xlabel('Segment')
        plt.ylabel('Product')
        plt.title('Segment Purchase Profiles')
        plt.yticks(range(len(product_names)), product_names)
        plt.colorbar(im, label='Purchase Rate')
        
        # Add text annotations
        for i in range(3):
            for j in range(len(product_names)):
                plt.text(i, j, f'{segment_profiles[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if segment_profiles[i, j] < 0.5 else 'black')
        
        # Segment sizes
        plt.subplot(2, 2, 2)
        segment_sizes = [np.sum(segments == i) for i in range(3)]
        plt.pie(segment_sizes, labels=[f'Segment {i}' for i in range(3)], autopct='%1.1f%%')
        plt.title('Segment Size Distribution')
        
        # Purchase frequency by segment
        plt.subplot(2, 2, 3)
        total_purchases = np.sum(X_binary, axis=1)
        for segment in range(3):
            segment_purchases = total_purchases[segments == segment]
            plt.hist(segment_purchases, alpha=0.7, label=f'Segment {segment}', bins=range(6))
        plt.xlabel('Number of Products Purchased')
        plt.ylabel('Frequency')
        plt.title('Purchase Frequency by Segment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Dendrogram
        plt.subplot(2, 2, 4)
        dendrogram(hierarchical.linkage_matrix_, truncate_mode='level', p=3)
        plt.title('Customer Segmentation Dendrogram')
        plt.ylabel('Distance')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'data': (X_iris, y_iris_true, X_scaled, X_pca),
            'clustering_results': clustering_results,
            'cluster_profiles': cluster_profiles,
            'anomaly_detection': {
                'anomalies': anomalies,
                'threshold': anomaly_threshold,
                'distances': distances_to_centers
            },
            'market_basket': {
                'binary_data': X_binary,
                'segments': segments,
                'segment_profiles': segment_profiles
            }
        }

        @staticmethod
    def conceptual_foundations():
        """Tutorial on fundamental concepts of unsupervised learning"""
        print("="*80)
        print("UNSUPERVISED LEARNING CONCEPTUAL FOUNDATIONS")
        print("="*80)
        
        print("""
        What is Unsupervised Learning?
        =============================
        
        Unlike supervised learning, unsupervised learning finds patterns in data
        WITHOUT labeled examples. We only have input data X, not target labels y.
        
        Key Paradigms:
        
        1. CLUSTERING: Group similar data points together
           - K-means: Partition data into k spherical clusters
           - Hierarchical: Build tree of nested clusters
           - GMM: Model data as mixture of probability distributions
        
        2. DIMENSIONALITY REDUCTION: Find lower-dimensional representations
           - PCA: Find directions of maximum variance
           - t-SNE: Preserve local neighborhood structure
           - Autoencoders: Neural network-based compression
        
        3. DENSITY ESTIMATION: Model the probability distribution of data
           - Kernel Density Estimation: Non-parametric density modeling
           - Gaussian Mixture Models: Parametric density modeling
        
        4. ASSOCIATION RULES: Find relationships between features
           - Market basket analysis: "People who buy X also buy Y"
           - Frequent pattern mining: Discover common co-occurrences
        
        Mathematical Foundations:
        ========================
        
        Most unsupervised learning can be viewed as optimization problems:
        
        K-means: minimize Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
        PCA: maximize variance of projections
        GMM: maximize likelihood of data under mixture model
        
        The challenge: No ground truth to guide optimization!
        """)
    
    @staticmethod
    def clustering_intuition():
        """Interactive demonstration of clustering concepts"""
        print("\n" + "="*80)
        print("CLUSTERING INTUITION AND ALGORITHM COMPARISON")
        print("="*80)
        
        # Generate datasets that highlight different clustering challenges
        np.random.seed(42)
        
        # Dataset 1: Different cluster shapes
        n_samples = 200
        
        # Spherical clusters (good for k-means)
        spherical = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.5, random_state=42)[0]
        
        # Elongated clusters (challenging for k-means)
        elongated = np.vstack([
            np.random.multivariate_normal([0, 0], [[4, 3], [3, 4]], n_samples//2),
            np.random.multivariate_normal([8, 8], [[4, -3], [-3, 4]], n_samples//2)
        ])
        
        # Different densities
        dense_sparse = np.vstack([
            np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], n_samples//3),
            np.random.multivariate_normal([5, 5], [[3, 0], [0, 3]], 2*n_samples//3)
        ])
        
        datasets = [
            (spherical, "Spherical clusters"),
            (elongated, "Elongated clusters"), 
            (dense_sparse, "Different densities")
        ]
        
        print("Demonstrating how cluster shape affects algorithm performance:")
        
        fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 12))
        
        for row, (X, title) in enumerate(datasets):
            # Original data
            axes[row, 0].scatter(X[:, 0], X[:, 1], alpha=0.7, s=30, c='gray')
            axes[row, 0].set_title(f'{title}\n(Original data)')
            axes[row, 0].grid(True, alpha=0.3)
            
            # K-means
            kmeans = KMeansFromScratch(n_clusters=2, random_state=42)
            kmeans_labels = kmeans.fit_predict(X)
            
            axes[row, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7, s=30)
            axes[row, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                               c='red', marker='x', s=200, linewidth=3)
            axes[row, 1].set_title('K-means')
            axes[row, 1].grid(True, alpha=0.3)
            
            # GMM
            gmm = GaussianMixtureModel(n_components=2, random_state=42)
            gmm_labels = gmm.fit_predict(X)
            
            axes[row, 2].scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7, s=30)
            
            # Plot GMM ellipses
            for i in range(gmm.n_components):
                mean = gmm.means_[i]
                cov = gmm.covariances_[i]
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(2 * eigenvals)
                
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(mean, width, height, angle=angle, 
                                facecolor='none', edgecolor='red', alpha=0.7, linewidth=2)
                axes[row, 2].add_patch(ellipse)
            
            axes[row, 2].set_title('Gaussian Mixture Model')
            axes[row, 2].grid(True, alpha=0.3)
            
            # Calculate and print silhouette scores
            kmeans_sil = ClusteringEvaluation.silhouette_score_manual(X, kmeans_labels)
            gmm_sil = ClusteringEvaluation.silhouette_score_manual(X, gmm_labels)
            
            print(f"{title}:")
            print(f"  K-means silhouette: {kmeans_sil:.3f}")
            print(f"  GMM silhouette: {gmm_sil:.3f}")
        
        plt.tight_layout()
        plt.show()
        
        print("""
        Key Insights:
        ============
        
        1. SPHERICAL CLUSTERS: K-means works well when clusters are roughly spherical
           and similar in size. The assumption of spherical clusters is reasonable.
        
        2. ELONGATED CLUSTERS: K-means struggles with elongated or non-spherical clusters
           because it assumes spherical boundaries. GMM can handle elliptical shapes.
        
        3. DIFFERENT DENSITIES: Both algorithms can struggle when clusters have very
           different densities or sizes. This violates the assumption of similar
           cluster characteristics.
        
        Algorithm Selection Guidelines:
        ==============================
        
        Use K-means when:
        - Clusters are roughly spherical
        - Similar cluster sizes
        - Need fast, simple algorithm
        - Hard cluster assignments desired
        
        Use GMM when:
        - Clusters may be elliptical
        - Need probabilistic assignments
        - Different cluster sizes/shapes
        - Want to model uncertainty
        
        Use Hierarchical when:
        - Don't know number of clusters
        - Want to explore cluster structure
        - Need reproducible results
        - Have non-spherical clusters
        """)
    
    @staticmethod
    def dimensionality_reduction_concepts():
        """Tutorial on dimensionality reduction concepts"""
        print("\n" + "="*80)
        print("DIMENSIONALITY REDUCTION CONCEPTS")
        print("="*80)
        
        print("""
        The Curse of Dimensionality
        ==========================
        
        As the number of dimensions increases:
        1. Data becomes increasingly sparse
        2. Distance metrics become less meaningful
        3. Volume of space increases exponentially
        4. Visualization becomes impossible
        
        Solution: Find lower-dimensional representations that preserve important structure
        
        Principal Component Analysis (PCA)
        =================================
        
        Mathematical Foundation:
        - Find directions of maximum variance
        - These directions are eigenvectors of covariance matrix
        - Project data onto these principal components
        
        Steps:
        1. Center the data: X̃ = X - μ
        2. Compute covariance matrix: C = (1/n)X̃ᵀX̃
        3. Find eigenvalues and eigenvectors: Cv = λv
        4. Sort by eigenvalues (largest first)
        5. Project: Y = X̃W where W contains top eigenvectors
        
        Key Properties:
        - Linear transformation
        - Preserves maximum variance
        - Decorrelates features
        - Minimizes reconstruction error
        """)
        
        # Demonstrate PCA concepts with synthetic data
        np.random.seed(42)
        
        # Generate correlated 2D data
        mean = [0, 0]
        cov = [[3, 2], [2, 2]]
        data_2d = np.random.multivariate_normal(mean, cov, 200)
        
        # Apply PCA
        pca = PrincipalComponentAnalysis(n_components=2)
        data_pca = pca.fit_transform(data_2d)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data
        axes[0].scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
        axes[0].set_xlabel('X1')
        axes[0].set_ylabel('X2')
        axes[0].set_title('Original Correlated Data')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # Show principal components
        center = np.mean(data_2d, axis=0)
        for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
            direction = component * np.sqrt(variance) * 3  # Scale for visualization
            axes[0].arrow(center[0], center[1], direction[0], direction[1],
                         head_width=0.2, head_length=0.3, fc=f'C{i}', ec=f'C{i}',
                         linewidth=3, label=f'PC{i+1}')
        axes[0].legend()
        
        # Data in PCA space
        axes[1].scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.7)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1].set_title('Data in PCA Space')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        # 1D projection (first PC only)
        axes[2].scatter(data_pca[:, 0], np.zeros_like(data_pca[:, 0]), alpha=0.7)
        axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[2].set_ylabel('0')
        axes[2].set_title('1D Projection (PC1 only)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Original data shape: {data_2d.shape}")
        print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
        print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
        print(f"Total variance explained by both PCs: {np.sum(pca.explained_variance_ratio_):.1%}")
        
        # Show reconstruction error
        data_1d = data_pca[:, :1]  # Keep only first PC
        pca_1d = PrincipalComponentAnalysis(n_components=1)
        pca_1d.fit(data_2d)
        data_reconstructed = pca_1d.inverse_transform(pca_1d.transform(data_2d))
        
        reconstruction_error = np.mean((data_2d - data_reconstructed) ** 2)
        print(f"Reconstruction error (1D): {reconstruction_error:.4f}")
    
    @staticmethod
    def practical_guidelines():
        """Practical guidelines for unsupervised learning"""
        print("\n" + "="*80)
        print("PRACTICAL GUIDELINES FOR UNSUPERVISED LEARNING")
        print("="*80)
        
        guidelines = [
            {
                'category': 'Data Preprocessing',
                'recommendations': [
                    "Scale features when using distance-based algorithms (K-means, hierarchical)",
                    "Handle missing values appropriately - imputation or removal",
                    "Remove or transform outliers that might skew cluster centers",
                    "Consider feature selection to reduce noise and dimensionality",
                    "Normalize data for PCA if features have different units",
                    "Check for multicollinearity in PCA applications"
                ]
            },
            {
                'category': 'Algorithm Selection',
                'recommendations': [
                    "K-means: Fast, spherical clusters, known k",
                    "GMM: Probabilistic assignments, elliptical clusters",
                    "Hierarchical: Unknown k, dendrogram analysis",
                    "PCA: Linear relationships, data visualization",
                    "t-SNE: Non-linear visualization, local structure",
                    "DBSCAN: Noise handling, arbitrary cluster shapes"
                ]
            },
            {
                'category': 'Determining Number of Clusters',
                'recommendations': [
                    "Elbow method: Look for 'elbow' in within-cluster sum of squares",
                    "Silhouette analysis: Maximize average silhouette score",
                    "Gap statistic: Compare with random data",
                    "Information criteria: AIC/BIC for GMM",
                    "Domain knowledge: Business constraints and interpretability",
                    "Stability analysis: Consistent results across runs"
                ]
            },
            {
                'category': 'Validation and Evaluation',
                'recommendations': [
                    "Use multiple metrics: silhouette, Calinski-Harabasz, Davies-Bouldin",
                    "Visual inspection: Always plot results when possible",
                    "Stability testing: Run multiple times with different initializations",
                    "Domain expert review: Do clusters make business sense?",
                    "Cross-validation: For dimensionality reduction methods",
                    "Sensitivity analysis: How robust are results to parameters?"
                ]
            },
            {
                'category': 'Common Pitfalls',
                'recommendations': [
                    "Don't assume spherical clusters with K-means",
                    "Don't ignore the curse of dimensionality",
                    "Don't use clustering on purely random data",
                    "Don't over-interpret small clusters",
                    "Don't forget to validate cluster stability",
                    "Don't ignore domain knowledge in favor of metrics"
                ]
            }
        ]
        
        for guideline in guidelines:
            print(f"\n{guideline['category']}:")
            print("-" * len(guideline['category']))
            for recommendation in guideline['recommendations']:
                print(f"  • {recommendation}")
        
        print(f"\nWorkflow Checklist:")
        print("-" * 20)
        workflow_steps = [
            "1. Explore and preprocess data",
            "2. Choose appropriate algorithm(s)",
            "3. Determine optimal parameters",
            "4. Fit model and extract results",
            "5. Validate using multiple metrics",
            "6. Interpret and visualize results",
            "7. Test stability and robustness",
            "8. Document findings and assumptions"
        ]
        
        for step in workflow_steps:
            print(f"  {step}")

def main():
    """Main function to run all unsupervised learning experiments"""
    print("🧠 NEURAL ODYSSEY - WEEK 19: UNSUPERVISED LEARNING FOUNDATIONS")
    print("="*80)
    print("Complete exploration of pattern discovery without labels")
    print("="*80)
    
    # Foundational tutorials
    UnsupervisedLearningTutorial.conceptual_foundations()
    UnsupervisedLearningTutorial.clustering_intuition()
    UnsupervisedLearningTutorial.dimensionality_reduction_concepts()
    
    # Run all experiments
    experiments_results = {}
    
    # Experiment 1: K-means fundamentals
    experiments_results['kmeans_fundamentals'] = UnsupervisedLearningExperiments.experiment_1_kmeans_fundamentals()
    
    # Experiment 2: Hierarchical clustering
    experiments_results['hierarchical_clustering'] = UnsupervisedLearningExperiments.experiment_2_hierarchical_clustering()
    
    # Experiment 3: PCA dimensionality reduction
    experiments_results['pca_analysis'] = UnsupervisedLearningExperiments.experiment_3_pca_dimensionality_reduction()
    
    # Experiment 4: Gaussian Mixture Models
    experiments_results['gaussian_mixtures'] = UnsupervisedLearningExperiments.experiment_4_gaussian_mixture_models()
    
    # Experiment 5: Clustering evaluation
    experiments_results['clustering_evaluation'] = UnsupervisedLearningExperiments.experiment_5_clustering_evaluation()
    
    # Experiment 6: Real-world applications
    experiments_results['real_world_applications'] = UnsupervisedLearningExperiments.experiment_6_real_world_applications()
    
    # Practical guidelines
    UnsupervisedLearningTutorial.practical_guidelines()
    
    # Final summary
    print("\n" + "="*80)
    print("WEEK 19 SUMMARY: UNSUPERVISED LEARNING FOUNDATIONS MASTERY")
    print("="*80)
    
    summary_text = """
    🎯 ACHIEVEMENTS UNLOCKED:
    
    ✅ Clustering Mastery
       • Implemented K-means algorithm from mathematical first principles
       • Built hierarchical clustering with multiple linkage criteria
       • Developed Gaussian Mixture Models with EM algorithm
       • Mastered clustering evaluation metrics and validation
    
    ✅ Dimensionality Reduction Expertise
       • Derived PCA from eigendecomposition of covariance matrix
       • Applied PCA for visualization and noise reduction
       • Understood curse of dimensionality and mitigation strategies
       • Connected linear algebra foundations to practical applications
    
    ✅ Pattern Discovery Techniques
       • Learned to find structure in unlabeled data
       • Developed intuition for different clustering scenarios
       • Built comprehensive evaluation frameworks
       • Applied multiple algorithms to diverse datasets
    
    ✅ Real-World Applications
       • Customer segmentation and market analysis
       • Anomaly detection using clustering
       • Market basket analysis and association rules
       • Data compression and feature extraction
    
    ✅ Advanced Mathematical Understanding
       • EM algorithm for probabilistic modeling
       • Optimization landscapes in unsupervised learning
       • Information-theoretic model selection criteria
       • Statistical validation of unsupervised results
    
    🔗 CONNECTIONS TO BROADER ML:
       • Foundation for deep learning autoencoders
       • Preprocessing step for supervised learning
       • Feature engineering and selection methods
       • Generative modeling and density estimation
       • Semi-supervised and self-supervised learning
    
    📈 NEXT STEPS:
       • Advanced clustering methods (spectral, density-based)
       • Non-linear dimensionality reduction (t-SNE, UMAP)
       • Deep unsupervised learning (VAEs, GANs)
       • Manifold learning and topological data analysis
    """
    
    print(summary_text)
    
    return {
        'experiments': experiments_results,
        'status': 'completed'
    }

if __name__ == "__main__":
    # Set up plotting for better visualization
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # Run complete unsupervised learning module
    results = main()
    
    print("\n🎉 Week 19: Unsupervised Learning Foundations - COMPLETED!")
    print("Ready to advance to Week 20: Advanced Machine Learning Topics")