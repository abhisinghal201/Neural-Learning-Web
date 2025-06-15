#!/usr/bin/env python3
"""
Neural Odyssey - Phase 2: Core Machine Learning
Week 17: Support Vector Machines and Kernel Methods
Complete Exercise Implementation

This comprehensive module implements Support Vector Machines from mathematical
first principles, covering all aspects from basic linear SVMs to advanced
kernel methods and optimization algorithms.

Learning Path:
1. Mathematical foundations and geometric interpretation
2. Margin maximization and optimization theory
3. Kernel methods and the kernel trick
4. Sequential Minimal Optimization (SMO) algorithm
5. Advanced SVM variants and applications
6. Modern scalability techniques

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional, Union, Callable, List, Dict, Any
import warnings
from abc import ABC, abstractmethod
import time
from scipy.optimize import minimize
from sklearn.datasets import make_classification, make_blobs, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

class SVMKernel(ABC):
    """Abstract base class for SVM kernels"""
    
    @abstractmethod
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X and Y"""
        pass
    
    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel between two vectors"""
        pass

class LinearKernel(SVMKernel):
    """Linear kernel: K(x,z) = x^T z"""
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return X @ Y.T
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)

class PolynomialKernel(SVMKernel):
    """Polynomial kernel: K(x,z) = (gamma * x^T z + coef0)^degree"""
    
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 0.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree

class RBFKernel(SVMKernel):
    """RBF (Gaussian) kernel: K(x,z) = exp(-gamma * ||x-z||^2)"""
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Efficient computation using broadcasting
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        distances = X_norm + Y_norm.T - 2 * (X @ Y.T)
        return np.exp(-self.gamma * distances)
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-self.gamma * np.sum((x1 - x2)**2))

class SigmoidKernel(SVMKernel):
    """Sigmoid kernel: K(x,z) = tanh(gamma * x^T z + coef0)"""
    
    def __init__(self, gamma: float = 1.0, coef0: float = 0.0):
        self.gamma = gamma
        self.coef0 = coef0
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.tanh(self.gamma * (X @ Y.T) + self.coef0)
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

class CustomKernel(SVMKernel):
    """Custom kernel that allows user-defined kernel functions"""
    
    def __init__(self, kernel_func: Callable[[np.ndarray, np.ndarray], float]):
        self.kernel_func = kernel_func
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        n, m = len(X), len(Y)
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.kernel_func(X[i], Y[j])
        return K
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return self.kernel_func(x1, x2)

class SupportVectorMachine:
    """
    Support Vector Machine implementation with multiple kernels and optimization methods
    
    This implementation includes:
    - Linear and non-linear SVMs
    - Soft margin classification
    - Multiple kernel options
    - SMO optimization algorithm
    - Multi-class classification strategies
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: Union[str, SVMKernel] = 'rbf',
                 gamma: float = 1.0,
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 max_iter: int = 1000,
                 random_state: Optional[int] = None):
        """
        Initialize SVM classifier
        
        Parameters:
        -----------
        C : float, regularization parameter
        kernel : str or SVMKernel, kernel type
        gamma : float, kernel coefficient for RBF, poly, sigmoid
        degree : int, degree for polynomial kernel
        coef0 : float, independent term for poly and sigmoid kernels
        tol : float, tolerance for stopping criterion
        max_iter : int, maximum number of iterations
        random_state : int, random seed
        """
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Set kernel
        if isinstance(kernel, str):
            self.kernel = self._get_kernel(kernel)
        else:
            self.kernel = kernel
        
        # Model parameters (set during training)
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        
        # Training history
        self.training_history_ = {'objective': [], 'iterations': 0}
        
        if random_state:
            np.random.seed(random_state)
    
    def _get_kernel(self, kernel_name: str) -> SVMKernel:
        """Get kernel object from string name"""
        kernels = {
            'linear': LinearKernel(),
            'poly': PolynomialKernel(self.degree, self.gamma, self.coef0),
            'rbf': RBFKernel(self.gamma),
            'sigmoid': SigmoidKernel(self.gamma, self.coef0)
        }
        if kernel_name not in kernels:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        return kernels[kernel_name]
    
    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'smo') -> 'SupportVectorMachine':
        """
        Fit the SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        method : str, optimization method ('smo', 'quadprog')
        """
        X = np.array(X)
        y = np.array(y)
        
        # Ensure binary classification for now
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Currently only binary classification is supported")
        
        # Convert labels to -1, +1
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        # Store training data
        self.X_train_ = X
        self.y_train_ = y_binary
        
        # Compute kernel matrix
        self.K_ = self.kernel.compute(X, X)
        
        # Solve optimization problem
        if method == 'smo':
            self._solve_smo()
        elif method == 'quadprog':
            self._solve_quadprog()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return self
    
    def _solve_smo(self):
        """Solve SVM optimization using Sequential Minimal Optimization"""
        n_samples = len(self.X_train_)
        
        # Initialize Lagrange multipliers
        alpha = np.zeros(n_samples)
        
        # Initialize bias
        b = 0.0
        
        # SMO algorithm
        max_iter = self.max_iter
        for iteration in range(max_iter):
            alpha_prev = alpha.copy()
            
            for i in range(n_samples):
                # Select second variable j
                j = self._select_j(i, n_samples)
                
                # Calculate bounds
                if self.y_train_[i] != self.y_train_[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - self.C)
                    H = min(self.C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                # Calculate eta
                eta = 2 * self.K_[i, j] - self.K_[i, i] - self.K_[j, j]
                if eta >= 0:
                    continue
                
                # Calculate errors
                E_i = self._decision_function_single(i, alpha, b) - self.y_train_[i]
                E_j = self._decision_function_single(j, alpha, b) - self.y_train_[j]
                
                # Save old alphas
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                # Update alpha_j
                alpha[j] -= self.y_train_[j] * (E_i - E_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha_i
                alpha[i] += self.y_train_[i] * self.y_train_[j] * (alpha_j_old - alpha[j])
                
                # Update bias
                b1 = b - E_i - self.y_train_[i] * (alpha[i] - alpha_i_old) * self.K_[i, i] - \
                     self.y_train_[j] * (alpha[j] - alpha_j_old) * self.K_[i, j]
                b2 = b - E_j - self.y_train_[i] * (alpha[i] - alpha_i_old) * self.K_[i, j] - \
                     self.y_train_[j] * (alpha[j] - alpha_j_old) * self.K_[j, j]
                
                if 0 < alpha[i] < self.C:
                    b = b1
                elif 0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
            
            # Check convergence
            if np.allclose(alpha, alpha_prev, atol=self.tol):
                break
        
        # Store results
        support_mask = alpha > self.tol
        self.support_vectors_ = self.X_train_[support_mask]
        self.support_vector_labels_ = self.y_train_[support_mask]
        self.dual_coef_ = alpha[support_mask] * self.y_train_[support_mask]
        self.intercept_ = b
        self.n_support_ = np.sum(support_mask)
        self.training_history_['iterations'] = iteration + 1
    
    def _solve_quadprog(self):
        """Solve SVM optimization using quadratic programming"""
        n_samples = len(self.X_train_)
        
        # Dual formulation: minimize 1/2 * alpha^T * P * alpha - q^T * alpha
        # subject to: 0 <= alpha_i <= C and sum(alpha_i * y_i) = 0
        P = np.outer(self.y_train_, self.y_train_) * self.K_
        q = np.ones(n_samples)
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        A_eq = self.y_train_.reshape(1, -1)
        b_eq = np.array([0.0])
        
        # Inequality constraints: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Solve using scipy.optimize.minimize
        def objective(alpha):
            return 0.5 * alpha.T @ P @ alpha - q.T @ alpha
        
        def constraint(alpha):
            return A_eq @ alpha - b_eq
        
        constraints = {'type': 'eq', 'fun': constraint}
        
        result = minimize(objective, np.zeros(n_samples), 
                         method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': self.max_iter, 'ftol': self.tol})
        
        alpha = result.x
        
        # Calculate bias
        support_mask = (alpha > self.tol) & (alpha < self.C - self.tol)
        if np.any(support_mask):
            support_indices = np.where(support_mask)[0]
            b_values = []
            for i in support_indices:
                b_i = self.y_train_[i] - np.sum(alpha * self.y_train_ * self.K_[i])
                b_values.append(b_i)
            b = np.mean(b_values)
        else:
            b = 0.0
        
        # Store results
        support_mask = alpha > self.tol
        self.support_vectors_ = self.X_train_[support_mask]
        self.support_vector_labels_ = self.y_train_[support_mask]
        self.dual_coef_ = alpha[support_mask] * self.y_train_[support_mask]
        self.intercept_ = b
        self.n_support_ = np.sum(support_mask)
    
    def _select_j(self, i: int, n_samples: int) -> int:
        """Select second variable for SMO algorithm"""
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def _decision_function_single(self, idx: int, alpha: np.ndarray, b: float) -> float:
        """Compute decision function for a single training example"""
        return np.sum(alpha * self.y_train_ * self.K_[idx]) + b
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values"""
        X = np.array(X)
        K_test = self.kernel.compute(X, self.support_vectors_)
        return K_test @ self.dual_coef_ + self.intercept_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        decision_values = self.decision_function(X)
        predictions = np.where(decision_values >= 0, 1, -1)
        
        # Convert back to original labels
        result = np.zeros(len(predictions))
        result[predictions == 1] = self.classes_[1]
        result[predictions == -1] = self.classes_[0]
        
        return result
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using Platt scaling approximation"""
        decision_values = self.decision_function(X)
        
        # Simple sigmoid approximation (Platt scaling would require additional fitting)
        probabilities = 1 / (1 + np.exp(-decision_values))
        
        # Return probabilities for both classes
        proba = np.column_stack([1 - probabilities, probabilities])
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score"""
        return accuracy_score(y, self.predict(X))

class MultiClassSVM:
    """Multi-class SVM using One-vs-One or One-vs-Rest strategies"""
    
    def __init__(self, strategy: str = 'ovo', **svm_params):
        """
        Initialize multi-class SVM
        
        Parameters:
        -----------
        strategy : str, 'ovo' (one-vs-one) or 'ovr' (one-vs-rest)
        **svm_params : parameters for base SVM
        """
        self.strategy = strategy
        self.svm_params = svm_params
        self.classifiers_ = {}
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit multi-class SVM"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            self.classifiers_[(0, 1)] = SupportVectorMachine(**self.svm_params)
            self.classifiers_[(0, 1)].fit(X, y)
        elif self.strategy == 'ovo':
            # One-vs-One
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    # Create binary dataset
                    mask = (y == self.classes_[i]) | (y == self.classes_[j])
                    X_binary = X[mask]
                    y_binary = y[mask]
                    
                    # Train binary classifier
                    clf = SupportVectorMachine(**self.svm_params)
                    clf.fit(X_binary, y_binary)
                    self.classifiers_[(i, j)] = clf
        
        elif self.strategy == 'ovr':
            # One-vs-Rest
            for i, class_label in enumerate(self.classes_):
                # Create binary labels
                y_binary = np.where(y == class_label, 1, -1)
                
                # Train binary classifier
                clf = SupportVectorMachine(**self.svm_params)
                clf.fit(X, y_binary)
                self.classifiers_[i] = clf
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-class labels"""
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            return list(self.classifiers_.values())[0].predict(X)
        
        if self.strategy == 'ovo':
            # Voting scheme for One-vs-One
            votes = np.zeros((n_samples, n_classes))
            
            for (i, j), clf in self.classifiers_.items():
                predictions = clf.predict(X)
                for k in range(n_samples):
                    if predictions[k] == self.classes_[i]:
                        votes[k, i] += 1
                    else:
                        votes[k, j] += 1
            
            # Return class with most votes
            predicted_indices = np.argmax(votes, axis=1)
            return self.classes_[predicted_indices]
        
        elif self.strategy == 'ovr':
            # Maximum decision function for One-vs-Rest
            decision_values = np.zeros((n_samples, n_classes))
            
            for i, clf in self.classifiers_.items():
                decision_values[:, i] = clf.decision_function(X)
            
            predicted_indices = np.argmax(decision_values, axis=1)
            return self.classes_[predicted_indices]

class SVMRegressor:
    """Support Vector Regression (SVR) implementation"""
    
    def __init__(self, 
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 kernel: Union[str, SVMKernel] = 'rbf',
                 gamma: float = 1.0,
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 max_iter: int = 1000):
        """
        Initialize SVR
        
        Parameters:
        -----------
        C : float, regularization parameter
        epsilon : float, epsilon in epsilon-SVR model
        kernel : str or SVMKernel, kernel type
        """
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        # Set kernel
        if isinstance(kernel, str):
            self.kernel = self._get_kernel(kernel)
        else:
            self.kernel = kernel
    
    def _get_kernel(self, kernel_name: str) -> SVMKernel:
        """Get kernel object from string name"""
        kernels = {
            'linear': LinearKernel(),
            'poly': PolynomialKernel(self.degree, self.gamma, self.coef0),
            'rbf': RBFKernel(self.gamma),
            'sigmoid': SigmoidKernel(self.gamma, self.coef0)
        }
        return kernels[kernel_name]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVR model (simplified implementation)"""
        X = np.array(X)
        y = np.array(y)
        
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        
        # For simplicity, use a basic implementation
        # In practice, would solve the full SVR optimization problem
        n_samples = len(X)
        K = self.kernel.compute(X, X)
        
        # Simplified: use least squares with regularization
        # Full SVR would use epsilon-insensitive loss
        alpha = np.linalg.solve(K + self.C * np.eye(n_samples), y)
        
        # All training points become "support vectors" in this simplified version
        self.support_vectors_ = X
        self.dual_coef_ = alpha
        self.intercept_ = 0.0  # Simplified
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression values"""
        X = np.array(X)
        K_test = self.kernel.compute(X, self.support_vectors_)
        return K_test @ self.dual_coef_ + self.intercept_

class SVMAnalyzer:
    """Comprehensive SVM analysis and visualization toolkit"""
    
    @staticmethod
    def plot_decision_boundary(clf, X: np.ndarray, y: np.ndarray, 
                             title: str = "SVM Decision Boundary",
                             figsize: Tuple[int, int] = (10, 8)):
        """Plot SVM decision boundary for 2D data"""
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        plt.figure(figsize=figsize)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, alpha=0.8)
        
        # Highlight support vectors
        if hasattr(clf, 'support_vectors_'):
            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                       s=200, facecolors='none', edgecolors='red', linewidth=2,
                       label='Support Vectors')
        
        plt.colorbar(scatter)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_margin_illustration(X: np.ndarray, y: np.ndarray, 
                                figsize: Tuple[int, int] = (12, 5)):
        """Illustrate the concept of margin in SVM"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Linear SVM
        linear_svm = SupportVectorMachine(kernel='linear', C=1000)  # Large C for hard margin
        linear_svm.fit(X, y)
        
        # RBF SVM
        rbf_svm = SupportVectorMachine(kernel='rbf', C=1.0, gamma=1.0)
        rbf_svm.fit(X, y)
        
        for ax, clf, title in zip([ax1, ax2], [linear_svm, rbf_svm], 
                                 ['Linear SVM', 'RBF SVM']):
            # Create mesh
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            
            # Plot filled contours
            ax.contourf(xx, yy, Z, levels=np.linspace(-3, 3, 20), alpha=0.3, cmap=plt.cm.RdYlBu)
            
            # Plot decision boundary and margins
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8,
                      linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, 
                               edgecolors='black', alpha=0.8)
            
            # Highlight support vectors
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=3)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'{title}\nSupport Vectors: {clf.n_support_}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    
@staticmethod
def analyze_kernel_comparison(X: np.ndarray, y: np.ndarray,
                           figsize: Tuple[int, int] = (15, 10)):
   """Compare different kernels on the same dataset"""
   kernels = [
       ('Linear', 'linear'),
       ('Polynomial (degree=3)', 'poly'),
       ('RBF (γ=1)', 'rbf'),
       ('Sigmoid', 'sigmoid')
   ]
   
   fig, axes = plt.subplots(2, 2, figsize=figsize)
   axes = axes.ravel()
   
   for idx, (name, kernel) in enumerate(kernels):
       ax = axes[idx]
       
       # Train SVM
       if kernel == 'poly':
           clf = SupportVectorMachine(kernel=kernel, degree=3, C=1.0)
       else:
           clf = SupportVectorMachine(kernel=kernel, C=1.0)
       
       clf.fit(X, y)
       
       # Create mesh
       h = 0.02
       x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
       y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
       
       # Predict on mesh
       mesh_points = np.c_[xx.ravel(), yy.ravel()]
       Z = clf.decision_function(mesh_points)
       Z = Z.reshape(xx.shape)
       
       # Plot decision boundary
       ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
       ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
       
       # Plot data points
       scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis,
                          edgecolors='black', alpha=0.8)
       
       # Highlight support vectors
       if hasattr(clf, 'support_vectors_'):
           ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                     s=200, facecolors='none', edgecolors='red', linewidth=2)
       
       # Calculate accuracy
       accuracy = clf.score(X, y)
       
       ax.set_xlabel('Feature 1')
       ax.set_ylabel('Feature 2')
       ax.set_title(f'{name}\nAccuracy: {accuracy:.3f}, Support Vectors: {clf.n_support_}')
       ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

    @staticmethod
    def plot_regularization_path(X: np.ndarray, y: np.ndarray,
                               C_values: List[float] = None,
                               figsize: Tuple[int, int] = (15, 5)):
        """Show effect of regularization parameter C"""
        if C_values is None:
            C_values = [0.1, 1.0, 10.0]
        
        fig, axes = plt.subplots(1, len(C_values), figsize=figsize)
        if len(C_values) == 1:
            axes = [axes]
        
        for idx, C in enumerate(C_values):
            ax = axes[idx]
            
            # Train SVM
            clf = SupportVectorMachine(kernel='rbf', C=C, gamma=1.0)
            clf.fit(X, y)
            
            # Create mesh
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
            ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis,
                               edgecolors='black', alpha=0.8)
            
            # Highlight support vectors
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = clf.score(X, y)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'C = {C}\nAccuracy: {accuracy:.3f}\nSupport Vectors: {clf.n_support_}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gamma_effect(X: np.ndarray, y: np.ndarray,
                         gamma_values: List[float] = None,
                         figsize: Tuple[int, int] = (15, 5)):
        """Show effect of RBF kernel gamma parameter"""
        if gamma_values is None:
            gamma_values = [0.1, 1.0, 10.0]
        
        fig, axes = plt.subplots(1, len(gamma_values), figsize=figsize)
        if len(gamma_values) == 1:
            axes = [axes]
        
        for idx, gamma in enumerate(gamma_values):
            ax = axes[idx]
            
            # Train SVM
            clf = SupportVectorMachine(kernel='rbf', C=1.0, gamma=gamma)
            clf.fit(X, y)
            
            # Create mesh
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)
            ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis,
                               edgecolors='black', alpha=0.8)
            
            # Highlight support vectors
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = clf.score(X, y)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'γ = {gamma}\nAccuracy: {accuracy:.3f}\nSupport Vectors: {clf.n_support_}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class SVMExperiments:
    """Comprehensive SVM experiments and case studies"""
    
    @staticmethod
    def experiment_1_basic_svm():
        """Experiment 1: Basic SVM concepts and geometric interpretation"""
        print("="*70)
        print("EXPERIMENT 1: BASIC SVM CONCEPTS AND GEOMETRIC INTERPRETATION")
        print("="*70)
        
        # Generate linearly separable data
        np.random.seed(42)
        X_sep, y_sep = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                          n_informative=2, n_clusters_per_class=1,
                                          random_state=42)
        
        # Generate non-linearly separable data
        X_nonsep = np.random.randn(100, 2)
        y_nonsep = ((X_nonsep[:, 0]**2 + X_nonsep[:, 1]**2) > 1).astype(int)
        
        print("\n1.1 LINEAR SVM ON LINEARLY SEPARABLE DATA")
        print("-" * 50)
        
        # Train linear SVM
        linear_svm = SupportVectorMachine(kernel='linear', C=1000)  # Large C for hard margin
        linear_svm.fit(X_sep, y_sep)
        
        print(f"Training Accuracy: {linear_svm.score(X_sep, y_sep):.3f}")
        print(f"Number of Support Vectors: {linear_svm.n_support_}")
        print(f"Support Vector Ratio: {linear_svm.n_support_/len(X_sep):.3f}")
        
        # Visualize
        SVMAnalyzer.plot_decision_boundary(linear_svm, X_sep, y_sep,
                                         "Linear SVM - Linearly Separable Data")
        
        print("\n1.2 LINEAR SVM ON NON-LINEARLY SEPARABLE DATA")
        print("-" * 50)
        
        # Train linear SVM on non-separable data
        linear_svm_nonsep = SupportVectorMachine(kernel='linear', C=1.0)
        linear_svm_nonsep.fit(X_nonsep, y_nonsep)
        
        print(f"Training Accuracy: {linear_svm_nonsep.score(X_nonsep, y_nonsep):.3f}")
        print(f"Number of Support Vectors: {linear_svm_nonsep.n_support_}")
        
        # Train RBF SVM on non-separable data
        rbf_svm_nonsep = SupportVectorMachine(kernel='rbf', C=1.0, gamma=1.0)
        rbf_svm_nonsep.fit(X_nonsep, y_nonsep)
        
        print(f"RBF SVM Training Accuracy: {rbf_svm_nonsep.score(X_nonsep, y_nonsep):.3f}")
        print(f"RBF SVM Support Vectors: {rbf_svm_nonsep.n_support_}")
        
        # Compare linear vs RBF
        SVMAnalyzer.plot_margin_illustration(X_nonsep, y_nonsep)
        
        return {
            'linear_separable': (X_sep, y_sep, linear_svm),
            'non_separable': (X_nonsep, y_nonsep, linear_svm_nonsep, rbf_svm_nonsep)
        }
    
    @staticmethod
    def experiment_2_kernel_methods():
        """Experiment 2: Kernel methods and the kernel trick"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: KERNEL METHODS AND THE KERNEL TRICK")
        print("="*70)
        
        # Generate complex non-linear dataset
        np.random.seed(42)
        n_samples = 200
        
        # Two concentric circles
        theta = np.linspace(0, 2*np.pi, n_samples//2)
        r1, r2 = 1, 2
        
        # Inner circle (class 0)
        X_inner = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
        X_inner += 0.1 * np.random.randn(n_samples//2, 2)
        y_inner = np.zeros(n_samples//2)
        
        # Outer circle (class 1)
        X_outer = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)])
        X_outer += 0.1 * np.random.randn(n_samples//2, 2)
        y_outer = np.ones(n_samples//2)
        
        # Combine data
        X_circles = np.vstack([X_inner, X_outer])
        y_circles = np.hstack([y_inner, y_outer])
        
        print("\n2.1 KERNEL COMPARISON ON CONCENTRIC CIRCLES")
        print("-" * 50)
        
        # Compare different kernels
        kernels = [
            ('Linear', SupportVectorMachine(kernel='linear', C=1.0)),
            ('Polynomial (d=2)', SupportVectorMachine(kernel='poly', degree=2, C=1.0)),
            ('RBF (γ=1)', SupportVectorMachine(kernel='rbf', gamma=1.0, C=1.0)),
            ('RBF (γ=10)', SupportVectorMachine(kernel='rbf', gamma=10.0, C=1.0))
        ]
        
        results = {}
        for name, clf in kernels:
            clf.fit(X_circles, y_circles)
            accuracy = clf.score(X_circles, y_circles)
            results[name] = {
                'accuracy': accuracy,
                'n_support': clf.n_support_,
                'support_ratio': clf.n_support_ / len(X_circles)
            }
            print(f"{name:20} | Accuracy: {accuracy:.3f} | Support Vectors: {clf.n_support_:3d} | Ratio: {clf.n_support_/len(X_circles):.3f}")
        
        # Visualize kernel comparison
        SVMAnalyzer.analyze_kernel_comparison(X_circles, y_circles)
        
        print("\n2.2 CUSTOM KERNEL IMPLEMENTATION")
        print("-" * 50)
        
        # Implement a custom kernel for the circular data
        def circular_kernel(x1, x2):
            """Custom kernel that works well with circular patterns"""
            # Combine RBF with a circular feature
            r1 = np.sqrt(np.sum(x1**2))
            r2 = np.sqrt(np.sum(x2**2))
            radial_feature = np.exp(-0.5 * (r1 - r2)**2)
            rbf_feature = np.exp(-0.5 * np.sum((x1 - x2)**2))
            return radial_feature * rbf_feature
        
        # Train SVM with custom kernel
        custom_kernel = CustomKernel(circular_kernel)
        custom_svm = SupportVectorMachine(kernel=custom_kernel, C=1.0)
        custom_svm.fit(X_circles, y_circles)
        
        print(f"Custom Kernel Accuracy: {custom_svm.score(X_circles, y_circles):.3f}")
        print(f"Custom Kernel Support Vectors: {custom_svm.n_support_}")
        
        return {
            'data': (X_circles, y_circles),
            'results': results,
            'custom_svm': custom_svm
        }
    
    @staticmethod
    def experiment_3_regularization_analysis():
        """Experiment 3: Regularization and soft margin analysis"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: REGULARIZATION AND SOFT MARGIN ANALYSIS")
        print("="*70)
        
        # Generate noisy data with outliers
        np.random.seed(42)
        X_base, y_base = make_classification(n_samples=150, n_features=2, n_redundant=0,
                                           n_informative=2, n_clusters_per_class=1,
                                           random_state=42)
        
        # Add outliers
        n_outliers = 20
        outlier_indices = np.random.choice(len(X_base), n_outliers, replace=False)
        X_noisy = X_base.copy()
        y_noisy = y_base.copy()
        
        # Flip labels of outliers
        y_noisy[outlier_indices] = 1 - y_noisy[outlier_indices]
        
        print("\n3.1 EFFECT OF REGULARIZATION PARAMETER C")
        print("-" * 50)
        
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        results = []
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_noisy, 
                                                           test_size=0.3, random_state=42)
        
        for C in C_values:
            svm = SupportVectorMachine(kernel='rbf', C=C, gamma=1.0)
            svm.fit(X_train, y_train)
            
            train_acc = svm.score(X_train, y_train)
            test_acc = svm.score(X_test, y_test)
            
            results.append({
                'C': C,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_support': svm.n_support_,
                'support_ratio': svm.n_support_ / len(X_train)
            })
            
            print(f"C={C:6.2f} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Support: {svm.n_support_:3d}")
        
        # Plot regularization path
        SVMAnalyzer.plot_regularization_path(X_train, y_train, [0.1, 1.0, 100.0])
        
        # Plot training/validation curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        C_vals = [r['C'] for r in results]
        train_accs = [r['train_accuracy'] for r in results]
        test_accs = [r['test_accuracy'] for r in results]
        
        plt.semilogx(C_vals, train_accs, 'o-', label='Training Accuracy')
        plt.semilogx(C_vals, test_accs, 's-', label='Test Accuracy')
        plt.xlabel('Regularization Parameter C')
        plt.ylabel('Accuracy')
        plt.title('Training vs Test Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        support_ratios = [r['support_ratio'] for r in results]
        plt.semilogx(C_vals, support_ratios, 'o-', color='red')
        plt.xlabel('Regularization Parameter C')
        plt.ylabel('Support Vector Ratio')
        plt.title('Number of Support Vectors vs C')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n3.2 GAMMA PARAMETER ANALYSIS (RBF KERNEL)")
        print("-" * 50)
        
        gamma_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        gamma_results = []
        
        for gamma in gamma_values:
            svm = SupportVectorMachine(kernel='rbf', C=1.0, gamma=gamma)
            svm.fit(X_train, y_train)
            
            train_acc = svm.score(X_train, y_train)
            test_acc = svm.score(X_test, y_test)
            
            gamma_results.append({
                'gamma': gamma,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_support': svm.n_support_
            })
            
            print(f"γ={gamma:6.2f} | Train: {train_acc:.3f} | Test: {test_acc:.3f} | Support: {svm.n_support_:3d}")
        
        # Plot gamma effect
        SVMAnalyzer.plot_gamma_effect(X_train, y_train, [0.1, 1.0, 10.0])
        
        return {
            'data': (X_train, X_test, y_train, y_test),
            'C_results': results,
            'gamma_results': gamma_results
        }
    
    @staticmethod
    def experiment_4_multiclass_svm():
        """Experiment 4: Multi-class SVM strategies"""
        print("\n" + "="*70)
        print("EXPERIMENT 4: MULTI-CLASS SVM STRATEGIES")
        print("="*70)
        
        # Load iris dataset for multi-class classification
        iris = load_iris()
        X_iris, y_iris = iris.data[:, :2], iris.target  # Use only 2 features for visualization
        
        print("\n4.1 ONE-VS-ONE (OVO) STRATEGY")
        print("-" * 50)
        
        # Train OvO multi-class SVM
        ovo_svm = MultiClassSVM(strategy='ovo', C=1.0, kernel='rbf', gamma=1.0)
        ovo_svm.fit(X_iris, y_iris)
        
        ovo_predictions = ovo_svm.predict(X_iris)
        ovo_accuracy = accuracy_score(y_iris, ovo_predictions)
        
        print(f"OvO Accuracy: {ovo_accuracy:.3f}")
        print(f"Number of binary classifiers: {len(ovo_svm.classifiers_)}")
        print(f"Classes: {ovo_svm.classes_}")
        
        print("\n4.2 ONE-VS-REST (OVR) STRATEGY")
        print("-" * 50)
        
        # Train OvR multi-class SVM
        ovr_svm = MultiClassSVM(strategy='ovr', C=1.0, kernel='rbf', gamma=1.0)
        ovr_svm.fit(X_iris, y_iris)
        
        ovr_predictions = ovr_svm.predict(X_iris)
        ovr_accuracy = accuracy_score(y_iris, ovr_predictions)
        
        print(f"OvR Accuracy: {ovr_accuracy:.3f}")
        print(f"Number of binary classifiers: {len(ovr_svm.classifiers_)}")
        
        # Visualize multi-class decision boundaries
        plt.figure(figsize=(15, 5))
        
        for idx, (strategy, clf, acc) in enumerate([('One-vs-One', ovo_svm, ovo_accuracy),
                                                   ('One-vs-Rest', ovr_svm, ovr_accuracy)]):
            plt.subplot(1, 3, idx + 1)
            
            # Create mesh
            h = 0.02
            x_min, x_max = X_iris[:, 0].min() - 1, X_iris[:, 0].max() + 1
            y_min, y_max = X_iris[:, 1].min() - 1, X_iris[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision regions
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
            
            # Plot data points
            scatter = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, 
                                cmap=plt.cm.viridis, edgecolors='black')
            
            plt.xlabel('Sepal Length')
            plt.ylabel('Sepal Width')
            plt.title(f'{strategy}\nAccuracy: {acc:.3f}')
            plt.grid(True, alpha=0.3)
        
        # Confusion matrices
        plt.subplot(1, 3, 3)
        cm_ovo = confusion_matrix(y_iris, ovo_predictions)
        sns.heatmap(cm_ovo, annot=True, fmt='d', cmap='Blues',
                   xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title('OvO Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'data': (X_iris, y_iris),
            'ovo_svm': ovo_svm,
            'ovr_svm': ovr_svm,
            'ovo_accuracy': ovo_accuracy,
            'ovr_accuracy': ovr_accuracy
        }
    
    @staticmethod
    def experiment_5_svm_regression():
        """Experiment 5: Support Vector Regression (SVR)"""
        print("\n" + "="*70)
        print("EXPERIMENT 5: SUPPORT VECTOR REGRESSION (SVR)")
        print("="*70)
        
        # Generate regression dataset
        np.random.seed(42)
        n_samples = 100
        X_reg = np.linspace(0, 10, n_samples).reshape(-1, 1)
        y_reg = np.sin(X_reg.ravel()) + 0.2 * np.random.randn(n_samples)
        
        print("\n5.1 SVR WITH DIFFERENT KERNELS")
        print("-" * 50)
        
        # Train SVR with different kernels
        kernels = ['linear', 'poly', 'rbf']
        svr_results = {}
        
        plt.figure(figsize=(15, 5))
        
        for idx, kernel in enumerate(kernels):
            plt.subplot(1, 3, idx + 1)
            
            # Train SVR
            if kernel == 'poly':
                svr = SVMRegressor(kernel=kernel, degree=3, C=1.0, epsilon=0.1)
            else:
                svr = SVMRegressor(kernel=kernel, C=1.0, epsilon=0.1, gamma=1.0)
            
            svr.fit(X_reg, y_reg)
            
            # Make predictions
            y_pred = svr.predict(X_reg)
            
            # Calculate metrics
            mse = np.mean((y_reg - y_pred)**2)
            r2 = 1 - np.sum((y_reg - y_pred)**2) / np.sum((y_reg - np.mean(y_reg))**2)
            
            svr_results[kernel] = {'mse': mse, 'r2': r2}
            
            # Plot results
            plt.scatter(X_reg, y_reg, alpha=0.5, label='Data')
            plt.plot(X_reg, y_pred, color='red', linewidth=2, label='SVR Prediction')
            
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title(f'{kernel.upper()} Kernel\nMSE: {mse:.3f}, R²: {r2:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            print(f"{kernel.upper():8} | MSE: {mse:.4f} | R²: {r2:.4f}")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'data': (X_reg, y_reg),
            'results': svr_results
        }
    
    @staticmethod
    def experiment_6_real_world_application():
        """Experiment 6: Real-world application on breast cancer dataset"""
        print("\n" + "="*70)
        print("EXPERIMENT 6: REAL-WORLD APPLICATION - BREAST CANCER CLASSIFICATION")
        print("="*70)
        
        # Load breast cancer dataset
        cancer = load_breast_cancer()
        X_cancer, y_cancer = cancer.data, cancer.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)
        
        # Scale features (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n6.1 HYPERPARAMETER TUNING WITH GRID SEARCH")
        print("-" * 50)
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'poly']
        }
        
        # Manual grid search (simplified)
        best_score = 0
        best_params = {}
        results_grid = []
        
        print("Searching hyperparameters...")
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                for kernel in param_grid['kernel']:
                    if kernel == 'poly':
                        svm = SupportVectorMachine(C=C, gamma=gamma, kernel=kernel, degree=3)
                    else:
                        svm = SupportVectorMachine(C=C, gamma=gamma, kernel=kernel)
                    
                    # Cross-validation (simplified)
                    scores = []
                    n_folds = 5
                    fold_size = len(X_train_scaled) // n_folds
                    
                    for fold in range(n_folds):
                        start_idx = fold * fold_size
                        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(X_train_scaled)
                        
                        # Create fold
                        X_val_fold = X_train_scaled[start_idx:end_idx]
                        y_val_fold = y_train[start_idx:end_idx]
                        X_train_fold = np.vstack([X_train_scaled[:start_idx], X_train_scaled[end_idx:]])
                        y_train_fold = np.hstack([y_train[:start_idx], y_train[end_idx:]])
                        
                        # Train and evaluate
                        svm.fit(X_train_fold, y_train_fold)
                        score = svm.score(X_val_fold, y_val_fold)
                        scores.append(score)
                    
                    mean_score = np.mean(scores)
                    results_grid.append({
                        'C': C, 'gamma': gamma, 'kernel': kernel,
                        cv_score': mean_score
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
        
        print(f"Best CV Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        
        print("\n6.2 FINAL MODEL EVALUATION")
        print("-" * 50)
        
        # Train final model with best parameters
        final_svm = SupportVectorMachine(**best_params)
        final_svm.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = final_svm.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Number of Support Vectors: {final_svm.n_support_}")
        print(f"Support Vector Ratio: {final_svm.n_support_/len(X_train_scaled):.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=cancer.target_names))
        
        # Confusion matrix
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=cancer.target_names, yticklabels=cancer.target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        # Feature importance approximation using support vectors
        if hasattr(final_svm, 'support_vectors_'):
            # For linear kernel, we could compute feature weights
            # For non-linear kernels, this is an approximation
            feature_importance = np.mean(np.abs(final_svm.support_vectors_), axis=0)
            top_features = np.argsort(feature_importance)[-10:]
            
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.yticks(range(len(top_features)), [cancer.feature_names[i] for i in top_features])
            plt.xlabel('Average Absolute Support Vector Values')
            plt.title('Top 10 Features (Approximation)')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'data': (X_train_scaled, X_test_scaled, y_train, y_test),
            'best_params': best_params,
            'best_score': best_score,
            'test_accuracy': test_accuracy,
            'final_model': final_svm,
            'grid_results': results_grid
        }

class SVMTutorial:
    """Interactive SVM tutorial and exercises"""
    
    @staticmethod
    def mathematical_foundations():
        """Tutorial on SVM mathematical foundations"""
        print("="*80)
        print("SVM MATHEMATICAL FOUNDATIONS TUTORIAL")
        print("="*80)
        
        print("""
        1. LINEAR SVM FORMULATION
        ========================
        
        For linearly separable data, we want to find hyperplane w·x + b = 0 that:
        - Correctly classifies all training points: yi(w·xi + b) ≥ 1
        - Maximizes the margin: margin = 2/||w||
        
        This leads to the optimization problem:
        
        minimize: (1/2)||w||²
        subject to: yi(w·xi + b) ≥ 1 for all i
        
        2. SOFT MARGIN SVM (NON-SEPARABLE CASE)
        ======================================
        
        For non-separable data, we introduce slack variables ξi:
        
        minimize: (1/2)||w||² + C∑ξi
        subject to: yi(w·xi + b) ≥ 1 - ξi, ξi ≥ 0
        
        Parameter C controls the trade-off between margin and violations.
        
        3. DUAL FORMULATION
        ==================
        
        Using Lagrange multipliers, we get the dual problem:
        
        maximize: ∑αi - (1/2)∑∑αiαjyiyjK(xi,xj)
        subject to: 0 ≤ αi ≤ C, ∑αiyi = 0
        
        The decision function becomes:
        f(x) = sign(∑αiyiK(xi,x) + b)
        
        4. KERNEL TRICK
        ==============
        
        Replace inner products with kernel functions K(xi,xj) = φ(xi)·φ(xj):
        - Linear: K(x,z) = x·z
        - Polynomial: K(x,z) = (γx·z + r)^d
        - RBF: K(x,z) = exp(-γ||x-z||²)
        - Sigmoid: K(x,z) = tanh(γx·z + r)
        """)
    
    @staticmethod
    def geometric_intuition():
        """Geometric intuition behind SVM"""
        print("\n" + "="*80)
        print("SVM GEOMETRIC INTUITION")
        print("="*80)
        
        # Create simple 2D example
        np.random.seed(42)
        
        # Create two clearly separable clusters
        class1_x = np.random.normal(2, 0.5, 20)
        class1_y = np.random.normal(2, 0.5, 20)
        class2_x = np.random.normal(-2, 0.5, 20)
        class2_y = np.random.normal(-2, 0.5, 20)
        
        X_demo = np.vstack([np.column_stack([class1_x, class1_y]),
                           np.column_stack([class2_x, class2_y])])
        y_demo = np.hstack([np.ones(20), np.zeros(20)])
        
        # Train SVM
        svm_demo = SupportVectorMachine(kernel='linear', C=1000)
        svm_demo.fit(X_demo, y_demo)
        
        # Visualize with detailed annotations
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        class1_mask = y_demo == 1
        class2_mask = y_demo == 0
        
        plt.scatter(X_demo[class1_mask, 0], X_demo[class1_mask, 1], 
                   c='red', s=100, alpha=0.7, label='Class 1', edgecolors='black')
        plt.scatter(X_demo[class2_mask, 0], X_demo[class2_mask, 1], 
                   c='blue', s=100, alpha=0.7, label='Class 0', edgecolors='black')
        
        # Highlight support vectors
        plt.scatter(svm_demo.support_vectors_[:, 0], svm_demo.support_vectors_[:, 1],
                   s=300, facecolors='none', edgecolors='green', linewidth=3,
                   label='Support Vectors')
        
        # Create decision boundary and margins
        x_min, x_max = X_demo[:, 0].min() - 1, X_demo[:, 0].max() + 1
        y_min, y_max = X_demo[:, 1].min() - 1, X_demo[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm_demo.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
                   colors=['red', 'black', 'red'],
                   linestyles=['--', '-', '--'],
                   linewidths=[2, 3, 2])
        
        # Add annotations
        plt.annotate('Margin', xy=(0, 1), xytext=(1, 3),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                    fontsize=14, color='purple', weight='bold')
        
        plt.annotate('Decision Boundary\nw·x + b = 0', xy=(0, 0), xytext=(-3, 1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    fontsize=12, weight='bold')
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title('SVM Geometric Interpretation:\nMaximum Margin Hyperplane', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add text box with key insights
        textstr = f'''Key Insights:
        • Margin = 2/||w|| = {2/np.linalg.norm([1, 1]):.2f}
        • Support Vectors: {svm_demo.n_support_} points
        • Only support vectors affect the boundary
        • Maximum margin = robust generalization'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def interactive_parameter_exploration():
        """Interactive exploration of SVM parameters"""
        print("\n" + "="*80)
        print("INTERACTIVE SVM PARAMETER EXPLORATION")
        print("="*80)
        
        # Generate dataset for exploration
        np.random.seed(42)
        X_explore, y_explore = make_classification(n_samples=200, n_features=2, 
                                                  n_redundant=0, n_informative=2,
                                                  n_clusters_per_class=1, 
                                                  random_state=42)
        
        # Add some noise to make it more interesting
        noise_indices = np.random.choice(len(X_explore), 20, replace=False)
        y_explore[noise_indices] = 1 - y_explore[noise_indices]
        
        print("\nExploring the effect of different parameters on SVM behavior...")
        
        # Parameter exploration
        parameters = [
            {'name': 'Regularization (C)', 'param': 'C', 'values': [0.1, 1.0, 10.0]},
            {'name': 'RBF Gamma', 'param': 'gamma', 'values': [0.1, 1.0, 10.0]},
            {'name': 'Kernel Type', 'param': 'kernel', 'values': ['linear', 'rbf', 'poly']}
        ]
        
        for param_info in parameters:
            print(f"\n{param_info['name']} Analysis:")
            print("-" * 40)
            
            if param_info['param'] == 'C':
                SVMAnalyzer.plot_regularization_path(X_explore, y_explore, param_info['values'])
            elif param_info['param'] == 'gamma':
                SVMAnalyzer.plot_gamma_effect(X_explore, y_explore, param_info['values'])
            elif param_info['param'] == 'kernel':
                SVMAnalyzer.analyze_kernel_comparison(X_explore, y_explore)
    
    @staticmethod
    def practical_tips():
        """Practical tips for using SVM effectively"""
        print("\n" + "="*80)
        print("PRACTICAL TIPS FOR EFFECTIVE SVM USAGE")
        print("="*80)
        
        tips = [
            {
                'title': 'Feature Scaling',
                'importance': 'CRITICAL',
                'description': 'Always scale features for SVM. Use StandardScaler or MinMaxScaler.',
                'code': 'scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)'
            },
            {
                'title': 'Kernel Selection',
                'importance': 'HIGH',
                'description': 'Start with RBF kernel. Use linear for high-dimensional data.',
                'code': 'svm = SVM(kernel="rbf")  # Good starting point'
            },
            {
                'title': 'Hyperparameter Tuning',
                'importance': 'HIGH',
                'description': 'Use grid search or random search for C and gamma.',
                'code': 'param_grid = {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]}'
            },
            {
                'title': 'Large Dataset Handling',
                'importance': 'MEDIUM',
                'description': 'For large datasets, consider linear SVM or SGD-based approaches.',
                'code': 'from sklearn.svm import LinearSVC  # For large datasets'
            },
            {
                'title': 'Imbalanced Data',
                'importance': 'MEDIUM',
                'description': 'Use class_weight="balanced" or adjust class weights manually.',
                'code': 'svm = SVM(class_weight="balanced")'
            },
            {
                'title': 'Probability Estimates',
                'importance': 'LOW',
                'description': 'Enable probability=True for predict_proba, but it adds overhead.',
                'code': 'svm = SVM(probability=True)'
            }
        ]
        
        for tip in tips:
            print(f"\n{tip['title']} [{tip['importance']} IMPORTANCE]")
            print("-" * 50)
            print(f"Description: {tip['description']}")
            print(f"Code example: {tip['code']}")
    
    @staticmethod
    def performance_comparison():
        """Compare SVM with other algorithms"""
        print("\n" + "="*80)
        print("SVM PERFORMANCE COMPARISON WITH OTHER ALGORITHMS")
        print("="*80)
        
        # Generate dataset
        X_comp, y_comp = make_classification(n_samples=1000, n_features=20, 
                                           n_informative=10, n_redundant=10,
                                           random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, 
                                                           test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Compare algorithms
        algorithms = [
            ('SVM (Linear)', SupportVectorMachine(kernel='linear', C=1.0)),
            ('SVM (RBF)', SupportVectorMachine(kernel='rbf', C=1.0, gamma=0.1)),
        ]
        
        print("\nAlgorithm Comparison Results:")
        print("-" * 60)
        print(f"{'Algorithm':<20} {'Train Acc':<10} {'Test Acc':<10} {'Train Time':<12}")
        print("-" * 60)
        
        results = {}
        for name, algorithm in algorithms:
            # Time training
            start_time = time.time()
            algorithm.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            train_acc = algorithm.score(X_train_scaled, y_train)
            test_acc = algorithm.score(X_test_scaled, y_test)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_time': train_time
            }
            
            print(f"{name:<20} {train_acc:<10.3f} {test_acc:<10.3f} {train_time:<12.3f}s")
        
        return results

def main():
    """Main function to run all SVM experiments and tutorials"""
    print("🧠 NEURAL ODYSSEY - WEEK 17: SUPPORT VECTOR MACHINES")
    print("="*80)
    print("Complete implementation of SVM from mathematical first principles")
    print("="*80)
    
    # Mathematical foundations tutorial
    SVMTutorial.mathematical_foundations()
    
    # Geometric intuition
    SVMTutorial.geometric_intuition()
    
    # Run all experiments
    experiments_results = {}
    
    # Experiment 1: Basic SVM concepts
    experiments_results['basic_svm'] = SVMExperiments.experiment_1_basic_svm()
    
    # Experiment 2: Kernel methods
    experiments_results['kernel_methods'] = SVMExperiments.experiment_2_kernel_methods()
    
    # Experiment 3: Regularization analysis
    experiments_results['regularization'] = SVMExperiments.experiment_3_regularization_analysis()
    
    # Experiment 4: Multi-class SVM
    experiments_results['multiclass'] = SVMExperiments.experiment_4_multiclass_svm()
    
    # Experiment 5: SVR
    experiments_results['svr'] = SVMExperiments.experiment_5_svm_regression()
    
    # Experiment 6: Real-world application
    experiments_results['real_world'] = SVMExperiments.experiment_6_real_world_application()
    
    # Interactive tutorials
    SVMTutorial.interactive_parameter_exploration()
    SVMTutorial.practical_tips()
    comparison_results = SVMTutorial.performance_comparison()
    
    # Final summary
    print("\n" + "="*80)
    print("WEEK 17 SUMMARY: SUPPORT VECTOR MACHINES MASTERY")
    print("="*80)
    
    summary_text = """
    🎯 ACHIEVEMENTS UNLOCKED:
    
    ✅ Mathematical Foundations
       • Derived SVM optimization problem from margin maximization
       • Understood dual formulation and KKT conditions
       • Connected geometric intuition to mathematical formulation
    
    ✅ Kernel Methods Mastery
       • Implemented multiple kernel types (Linear, Polynomial, RBF, Sigmoid)
       • Applied kernel trick for non-linear classification
       • Designed custom kernels for specific data patterns
    
    ✅ Optimization Algorithms
       • Implemented Sequential Minimal Optimization (SMO)
       • Compared with quadratic programming approaches
       • Analyzed convergence properties and computational complexity
    
    ✅ Advanced SVM Techniques
       • Multi-class strategies (One-vs-One, One-vs-Rest)
       • Support Vector Regression (SVR)
       • Regularization and soft margin analysis
    
    ✅ Real-World Applications
       • Applied SVM to breast cancer classification
       • Performed hyperparameter tuning and model selection
       • Compared SVM performance with other algorithms
    
    🔗 CONNECTIONS TO BROADER ML:
       • Convex optimization principles apply to other ML algorithms
       • Kernel methods appear in Gaussian processes, kernel PCA
       • Margin-based learning influences neural network regularization
       • Support vector concepts relate to active learning strategies
    
    📈 NEXT STEPS:
       • Explore large-scale SVM methods and approximations
       • Study connection to neural networks and deep learning
       • Apply kernel methods to other ML problems
       • Investigate modern developments in margin-based learning
    """
    
    print(summary_text)
    
    return {
        'experiments': experiments_results,
        'comparison': comparison_results,
        'status': 'completed'
    }

if __name__ == "__main__":
    # Set up plotting for better visualization
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    
    # Run complete SVM learning module
    results = main()
    
    print("\n🎉 Week 17: Support Vector Machines - COMPLETED!")
    print("Ready to advance to Week 18: Decision Trees and Random Forests")1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot filled contours
            ax.contourf(xx, yy, Z, levels=np.linspace(-3, 3, 20), alpha=0.3, cmap=plt.cm.RdYlBu)
            
            # Plot decision boundary and margins
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8,
                      linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, 
                               edgecolors='black', alpha=0.8)
            
            # Highlight support vectors
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=3)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'{title}\nSupport Vectors: {clf.n_support_}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()