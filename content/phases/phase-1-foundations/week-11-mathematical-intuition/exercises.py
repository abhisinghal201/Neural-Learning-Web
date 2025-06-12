"""
Neural Odyssey - Week 11: Mathematical Intuition Development
Phase 1: Mathematical Foundations (Final Week)

Building Deep Mathematical Understanding for ML

This week focuses on developing profound mathematical intuition that bridges
abstract concepts with practical ML applications. You'll build visual and
geometric understanding of the mathematical tools that power machine learning.

Learning Objectives:
- Develop geometric intuition for linear algebra operations
- Visualize calculus concepts in the context of optimization
- Build probabilistic thinking for uncertainty handling
- Connect mathematical abstractions to ML implementations
- Create mental models for complex mathematical relationships

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional, Callable, Union
import math
import random
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# GEOMETRIC INTUITION FOR LINEAR ALGEBRA
# ==========================================

class LinearAlgebraIntuition:
    """
    Visual and geometric understanding of linear algebra concepts
    """
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    def visualize_vector_operations(self):
        """
        Visualize fundamental vector operations geometrically
        """
        print("🎯 Geometric Intuition: Vector Operations")
        print("=" * 45)
        
        # Define vectors
        v1 = np.array([3, 2])
        v2 = np.array([1, 3])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Vector Addition
        ax = axes[0, 0]
        self._plot_vector(ax, [0, 0], v1, 'red', 'v1', linewidth=3)
        self._plot_vector(ax, v1, v2, 'blue', 'v2', linewidth=3)
        self._plot_vector(ax, [0, 0], v1 + v2, 'green', 'v1 + v2', linewidth=3, linestyle='--')
        
        # Show parallelogram
        self._plot_vector(ax, [0, 0], v2, 'blue', '', alpha=0.3)
        self._plot_vector(ax, v2, v1, 'red', '', alpha=0.3)
        
        ax.set_title('Vector Addition\nParallelogram Rule')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 6)
        ax.legend()
        
        # 2. Scalar Multiplication
        ax = axes[0, 1]
        scalars = [0.5, 1, 1.5, 2]
        for i, s in enumerate(scalars):
            color = plt.cm.viridis(i / len(scalars))
            self._plot_vector(ax, [0, 0], s * v1, color, f'{s}v1', linewidth=2)
        
        ax.set_title('Scalar Multiplication\nStretching/Shrinking')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 5)
        ax.legend()
        
        # 3. Dot Product Geometric Interpretation
        ax = axes[0, 2]
        self._plot_vector(ax, [0, 0], v1, 'red', 'v1', linewidth=3)
        self._plot_vector(ax, [0, 0], v2, 'blue', 'v2', linewidth=3)
        
        # Project v2 onto v1
        projection = np.dot(v2, v1) / np.dot(v1, v1) * v1
        self._plot_vector(ax, [0, 0], projection, 'purple', 'proj(v2 onto v1)', linewidth=2, linestyle=':')
        
        # Show angle
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        ax.text(0.5, 0.5, f'θ = {np.degrees(angle):.1f}°', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_title(f'Dot Product: {np.dot(v1, v2):.1f}\nv1·v2 = |v1||v2|cos(θ)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.legend()
        
        # 4. Linear Transformation Visualization
        ax = axes[1, 0]
        
        # Original unit square
        unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        ax.plot(unit_square[0], unit_square[1], 'b-', linewidth=2, label='Original')
        
        # Transformation matrix
        A = np.array([[2, 1], [0, 1]])
        transformed_square = A @ unit_square
        ax.plot(transformed_square[0], transformed_square[1], 'r-', linewidth=2, label='Transformed')
        
        # Show basis vectors
        e1, e2 = np.array([1, 0]), np.array([0, 1])
        Ae1, Ae2 = A @ e1, A @ e2
        
        self._plot_vector(ax, [0, 0], e1, 'blue', 'e1', alpha=0.5)
        self._plot_vector(ax, [0, 0], e2, 'blue', 'e2', alpha=0.5)
        self._plot_vector(ax, [0, 0], Ae1, 'red', 'Ae1')
        self._plot_vector(ax, [0, 0], Ae2, 'red', 'Ae2')
        
        ax.set_title('Linear Transformation\nA = [[2,1], [0,1]]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.legend()
        
        # 5. Eigenvalue/Eigenvector Visualization
        ax = axes[1, 1]
        
        # Create transformation matrix with clear eigenstructure
        A = np.array([[3, 1], [0, 2]])
        eigenvals, eigenvecs = np.linalg.eig(A)
        
        # Plot eigenvectors
        for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
            # Original eigenvector
            self._plot_vector(ax, [0, 0], vec, self.colors[i], f'v{i+1}', linewidth=2)
            # Transformed eigenvector (scaled by eigenvalue)
            self._plot_vector(ax, [0, 0], val * vec, self.colors[i], f'λ{i+1}v{i+1} (λ={val:.1f})', 
                            linewidth=3, linestyle='--')
        
        # Show some other vectors and their transformations
        test_vecs = [np.array([1, 1]), np.array([2, -1])]
        for i, vec in enumerate(test_vecs):
            transformed = A @ vec
            self._plot_vector(ax, [0, 0], vec, 'gray', f'u{i+1}', alpha=0.5)
            self._plot_vector(ax, [0, 0], transformed, 'black', f'Au{i+1}', alpha=0.7, linestyle=':')
        
        ax.set_title('Eigenvalues & Eigenvectors\nSpecial directions preserved')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-2, 3)
        ax.legend()
        
        # 6. Singular Value Decomposition Geometry
        ax = axes[1, 2]
        
        # Create a matrix for SVD
        M = np.array([[3, 1], [1, 2]])
        U, s, Vt = np.linalg.svd(M)
        
        # Show the unit circle and its transformation
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)])
        
        # Transform the unit circle
        ellipse = M @ unit_circle
        
        ax.plot(unit_circle[0], unit_circle[1], 'b-', label='Unit Circle', linewidth=2)
        ax.plot(ellipse[0], ellipse[1], 'r-', label='Transformed (Ellipse)', linewidth=2)
        
        # Show singular vectors
        for i in range(2):
            # Right singular vectors (input directions)
            self._plot_vector(ax, [0, 0], Vt[i], 'green', f'v{i+1}', linewidth=2)
            # Left singular vectors (output directions) scaled by singular values
            self._plot_vector(ax, [0, 0], s[i] * U[:, i], 'purple', f'σ{i+1}u{i+1}', linewidth=2)
        
        ax.set_title('SVD Geometry\nCircle → Ellipse')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Print mathematical insights
        print("\n💡 Key Geometric Insights:")
        print("🔹 Vector addition follows parallelogram rule")
        print("🔹 Dot product measures projection and angle")
        print("🔹 Linear transformations stretch/rotate/shear space")
        print("🔹 Eigenvectors are preserved directions")
        print("🔹 SVD decomposes any transformation into rotation-scaling-rotation")
    
    def _plot_vector(self, ax, start, vector, color, label, linewidth=2, linestyle='-', alpha=1.0):
        """Helper to plot vectors as arrows"""
        ax.annotate('', xy=start + vector, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, 
                                 linestyle=linestyle, alpha=alpha))
        ax.plot([start[0], start[0] + vector[0]], [start[1], start[1] + vector[1]], 
               color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
    
    def visualize_matrix_interpretations(self):
        """
        Show different ways to interpret matrices in ML
        """
        print("\n🎯 Matrix Interpretations in Machine Learning")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Data Matrix: Each row is a sample, each column is a feature
        ax = axes[0, 0]
        np.random.seed(42)
        X = np.random.randn(8, 3)  # 8 samples, 3 features
        
        im = ax.imshow(X, cmap='RdBu', aspect='auto')
        ax.set_title('Data Matrix X\nRows = Samples, Columns = Features')
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
        
        # Add text annotations
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                ax.text(j, i, f'{X[i,j]:.1f}', ha='center', va='center', 
                       color='white' if abs(X[i,j]) > 1 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        # 2. Weight Matrix: Connections between layers
        ax = axes[0, 1]
        W = np.random.randn(4, 6) * 0.5  # 4 inputs to 6 outputs
        
        im = ax.imshow(W, cmap='RdBu', aspect='auto')
        ax.set_title('Weight Matrix W\nConnections: Input → Output')
        ax.set_xlabel('Output Neurons')
        ax.set_ylabel('Input Neurons')
        
        # Draw connection interpretation
        for i in range(min(3, W.shape[0])):
            for j in range(min(3, W.shape[1])):
                alpha = abs(W[i, j]) / np.max(np.abs(W))
                color = 'red' if W[i, j] > 0 else 'blue'
                ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                         facecolor=color, alpha=alpha))
        
        plt.colorbar(im, ax=ax)
        
        # 3. Covariance Matrix: Feature relationships
        ax = axes[1, 0]
        
        # Generate correlated data
        mean = [0, 0, 0]
        cov_matrix = np.array([[1.0, 0.7, 0.3],
                              [0.7, 1.0, -0.2],
                              [0.3, -0.2, 1.0]])
        
        im = ax.imshow(cov_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Covariance Matrix\nFeature Correlations')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Feature')
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cov_matrix[i,j]:.1f}', ha='center', va='center',
                       color='white' if abs(cov_matrix[i,j]) > 0.5 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # 4. Kernel Matrix: Similarity between samples
        ax = axes[1, 1]
        
        # Create kernel matrix (RBF kernel)
        n_samples = 6
        X_samples = np.random.randn(n_samples, 2)
        
        # Compute RBF kernel
        gamma = 1.0
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                diff = X_samples[i] - X_samples[j]
                kernel_matrix[i, j] = np.exp(-gamma * np.dot(diff, diff))
        
        im = ax.imshow(kernel_matrix, cmap='YlOrRd')
        ax.set_title('Kernel Matrix\nSample Similarities')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Sample')
        
        # Add similarity values
        for i in range(n_samples):
            for j in range(n_samples):
                ax.text(j, i, f'{kernel_matrix[i,j]:.2f}', ha='center', va='center',
                       color='white' if kernel_matrix[i,j] < 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Matrix Interpretation Guide:")
        print("🔹 Data Matrix: Organize samples and features")
        print("🔹 Weight Matrix: Learn feature transformations")
        print("🔹 Covariance Matrix: Capture feature relationships")
        print("🔹 Kernel Matrix: Measure sample similarities")
    
    def demonstrate_dimensionality_curse(self):
        """
        Visualize the curse of dimensionality
        """
        print("\n🎯 The Curse of Dimensionality")
        print("=" * 35)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Volume of unit hypersphere
        dimensions = range(1, 11)
        volumes = []
        
        for d in dimensions:
            # Volume of unit hypersphere in d dimensions
            if d % 2 == 1:
                # Odd dimension
                volume = (2 ** ((d+1)//2) * np.pi ** ((d-1)//2)) / math.factorial((d-1)//2) / d
            else:
                # Even dimension
                volume = np.pi ** (d//2) / math.factorial(d//2)
            volumes.append(volume)
        
        axes[0, 0].plot(dimensions, volumes, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Unit Hypersphere Volume\nvs Dimension')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Volume')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. Distance concentration
        np.random.seed(42)
        n_samples = 1000
        dimensions = [2, 5, 10, 50, 100]
        
        distance_ratios = []
        for d in dimensions:
            # Generate random points
            points = np.random.randn(n_samples, d)
            
            # Compute distances from origin
            distances = np.linalg.norm(points, axis=1)
            
            # Ratio of max to min distance
            ratio = np.max(distances) / np.min(distances)
            distance_ratios.append(ratio)
        
        axes[0, 1].plot(dimensions, distance_ratios, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Distance Concentration\nMax/Min Distance Ratio')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Distance Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Nearest neighbor behavior
        dimensions = [2, 5, 10, 20, 50]
        nn_distances = []
        
        for d in dimensions:
            points = np.random.randn(100, d)
            
            # Find nearest neighbor distances
            distances = []
            for i in range(len(points)):
                dists = [np.linalg.norm(points[i] - points[j]) 
                        for j in range(len(points)) if i != j]
                distances.append(min(dists))
            
            nn_distances.append(np.mean(distances))
        
        axes[1, 0].plot(dimensions, nn_distances, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average Nearest Neighbor\nDistance vs Dimension')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Average NN Distance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sample density requirements
        dimensions = range(1, 6)
        samples_needed = []
        
        # To maintain same density, samples needed grows exponentially
        base_samples = 100  # samples needed in 1D for good coverage
        for d in dimensions:
            # Rough approximation: need base_samples^d for same density
            needed = base_samples ** d
            samples_needed.append(needed)
        
        axes[1, 1].semilogy(dimensions, samples_needed, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Samples Needed for\nSame Density')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Samples Required')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Curse of Dimensionality Effects:")
        print("🔹 High-dimensional unit sphere has almost zero volume")
        print("🔹 All points become equidistant from each other")
        print("🔹 Nearest neighbors become meaningless")
        print("🔹 Sample requirements grow exponentially")
        print("🔹 Intuition from low dimensions often fails")


# ==========================================
# CALCULUS INTUITION FOR OPTIMIZATION
# ==========================================

class CalculusIntuition:
    """
    Visual understanding of calculus concepts in optimization
    """
    
    def __init__(self):
        pass
    
    def visualize_gradient_descent_landscape(self):
        """
        Visualize optimization landscapes and gradient descent
        """
        print("\n🎯 Optimization Landscape Intuition")
        print("=" * 40)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 1D Function and Gradient Descent
        ax1 = plt.subplot(2, 3, 1)
        
        x = np.linspace(-3, 3, 1000)
        y = x**4 - 4*x**2 + x + 1  # Function with multiple minima
        
        ax1.plot(x, y, 'b-', linewidth=2, label='f(x)')
        
        # Show gradient descent trajectory
        learning_rate = 0.01
        x_current = 2.5
        trajectory_x, trajectory_y = [x_current], []
        
        for _ in range(100):
            y_current = x_current**4 - 4*x_current**2 + x_current + 1
            trajectory_y.append(y_current)
            
            # Compute gradient
            gradient = 4*x_current**3 - 8*x_current + 1
            
            # Update
            x_current = x_current - learning_rate * gradient
            trajectory_x.append(x_current)
            
            if abs(gradient) < 0.01:  # Convergence
                break
        
        trajectory_y.append(x_current**4 - 4*x_current**2 + x_current + 1)
        
        ax1.plot(trajectory_x, trajectory_y, 'ro-', markersize=4, alpha=0.7, label='Gradient Descent')
        ax1.set_title('1D Gradient Descent')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 2D Contour Plot
        ax2 = plt.subplot(2, 3, 2)
        
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = (X**2 + Y**2) + 0.5*(X**2 * Y**2)  # Bowl with some complexity
        
        contours = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
        ax2.clabel(contours, inline=True, fontsize=8)
        
        # Gradient descent on 2D function
        x_start, y_start = 2, 2
        lr = 0.01
        path_x, path_y = [x_start], [y_start]
        
        for _ in range(200):
            # Gradients
            grad_x = 2*path_x[-1] + path_x[-1] * path_y[-1]**2
            grad_y = 2*path_y[-1] + path_y[-1] * path_x[-1]**2
            
            # Update
            new_x = path_x[-1] - lr * grad_x
            new_y = path_y[-1] - lr * grad_y
            
            path_x.append(new_x)
            path_y.append(new_y)
            
            if grad_x**2 + grad_y**2 < 0.001:
                break
        
        ax2.plot(path_x, path_y, 'r.-', markersize=3, alpha=0.8, linewidth=2)
        ax2.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax2.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
        ax2.set_title('2D Gradient Descent\nContour Lines')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend()
        
        # 3. 3D Surface
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        
        X_3d = X[::5, ::5]  # Subsample for better visualization
        Y_3d = Y[::5, ::5]
        Z_3d = Z[::5, ::5]
        
        ax3.plot_surface(X_3d, Y_3d, Z_3d, alpha=0.6, cmap='viridis')
        
        # Plot gradient descent path on surface
        path_z = [(px**2 + py**2) + 0.5*(px**2 * py**2) for px, py in zip(path_x[::10], path_y[::10])]
        ax3.plot(path_x[::10], path_y[::10], path_z, 'r.-', markersize=5, linewidth=3)
        
        ax3.set_title('3D Optimization Surface')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('f(x,y)')
        
        # 4. Different Optimization Challenges
        ax4 = plt.subplot(2, 3, 4)
        
        # Saddle point function
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z_saddle = X**2 - Y**2  # Saddle point at origin
        
        contours = ax4.contour(X, Y, Z_saddle, levels=20, colors='black', alpha=0.5)
        contours_filled = ax4.contourf(X, Y, Z_saddle, levels=20, cmap='RdBu', alpha=0.7)
        plt.colorbar(contours_filled, ax=ax4)
        
        ax4.plot(0, 0, 'ro', markersize=10, label='Saddle Point')
        ax4.set_title('Saddle Point\nf(x,y) = x² - y²')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.legend()
        
        # 5. Local vs Global Minima
        ax5 = plt.subplot(2, 3, 5)
        
        x = np.linspace(-4, 4, 1000)
        y = np.sin(x) + 0.1*x**2  # Function with multiple local minima
        
        ax5.plot(x, y, 'b-', linewidth=2)
        
        # Mark local and global minima
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(-y)  # Find minima by looking for peaks in -y
        
        for idx in peaks_idx:
            ax5.plot(x[idx], y[idx], 'ro', markersize=8)
        
        # Global minimum
        global_min_idx = np.argmin(y)
        ax5.plot(x[global_min_idx], y[global_min_idx], 'go', markersize=12, label='Global Minimum')
        
        ax5.set_title('Local vs Global Minima')
        ax5.set_xlabel('x')
        ax5.set_ylabel('f(x)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Learning Rate Effects
        ax6 = plt.subplot(2, 3, 6)
        
        x = np.linspace(-2, 2, 1000)
        y = x**2  # Simple quadratic
        
        ax6.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
        
        # Different learning rates
        learning_rates = [0.1, 0.5, 1.1]
        colors = ['green', 'orange', 'red']
        labels = ['Good LR', 'Large LR', 'Too Large LR']
        
        for lr, color, label in zip(learning_rates, colors, labels):
            x_curr = 1.5
            trajectory = [x_curr]
            
            for _ in range(10):
                gradient = 2 * x_curr  # Derivative of x²
                x_curr = x_curr - lr * gradient
                trajectory.append(x_curr)
                
                if abs(x_curr) > 3:  # Divergence
                    break
            
            y_traj = [x**2 for x in trajectory]
            ax6.plot(trajectory, y_traj, 'o-', color=color, markersize=4, label=label)
        
        ax6.set_title('Learning Rate Effects')
        ax6.set_xlabel('x')
        ax6.set_ylabel('f(x)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 5)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Optimization Insights:")
        print("🔹 Gradients point in direction of steepest increase")
        print("🔹 Gradient descent follows steepest decrease")
        print("🔹 Learning rate controls step size")
        print("🔹 Saddle points can trap optimization")
        print("🔹 Local minima vs global optimization challenge")
    
        def demonstrate_convexity_intuition(self):
        """
        Visualize convex vs non-convex functions and their optimization properties
        """
        print("\n🎯 Convexity: The Shape of Easy Optimization")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Convex Function
        ax = axes[0, 0]
        x = np.linspace(-3, 3, 1000)
        y_convex = x**2 + 1  # Simple convex function
        
        ax.plot(x, y_convex, 'b-', linewidth=3, label='Convex: f(x) = x² + 1')
        
        # Show that any line segment between two points lies above the function
        x1, x2 = -2, 1.5
        y1, y2 = x1**2 + 1, x2**2 + 1
        
        ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, label='Line segment')
        ax.plot([x1, x2], [y1, y2], 'ro', markersize=8)
        
        # Show intermediate points
        for t in [0.25, 0.5, 0.75]:
            x_interp = (1-t)*x1 + t*x2
            y_line = (1-t)*y1 + t*y2
            y_func = x_interp**2 + 1
            
            ax.plot(x_interp, y_line, 'ro', markersize=6, alpha=0.7)
            ax.plot(x_interp, y_func, 'bo', markersize=6, alpha=0.7)
            ax.plot([x_interp, x_interp], [y_func, y_line], 'g-', alpha=0.5)
        
        ax.set_title('Convex Function\nLine segment ≥ function')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Non-convex Function
        ax = axes[0, 1]
        y_nonconvex = x**4 - 4*x**2 + 2  # Non-convex with multiple minima
        
        ax.plot(x, y_nonconvex, 'r-', linewidth=3, label='Non-convex: f(x) = x⁴ - 4x² + 2')
        
        # Show line segment that goes below function
        x1, x2 = -1.5, 1.5
        y1, y2 = x1**4 - 4*x1**2 + 2, x2**4 - 4*x2**2 + 2
        
        ax.plot([x1, x2], [y1, y2], 'b--', linewidth=2, label='Line segment')
        ax.plot([x1, x2], [y1, y2], 'bo', markersize=8)
        
        ax.set_title('Non-Convex Function\nLine segment can go below function')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Convex Set
        ax = axes[0, 2]
        
        # Draw a convex set (circle)
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        
        ax.fill(x_circle, y_circle, alpha=0.3, color='blue', label='Convex Set')
        ax.plot(x_circle, y_circle, 'b-', linewidth=2)
        
        # Show that line segment between any two points stays inside
        p1 = np.array([0.5, 0.7])
        p2 = np.array([-0.6, -0.4])
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3, label='Line segment inside')
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro', markersize=8)
        
        ax.set_title('Convex Set\nLine segment between any\ntwo points stays inside')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 4. Non-convex Set
        ax = axes[1, 0]
        
        # Draw a non-convex set (crescent)
        theta1 = np.linspace(0, np.pi, 50)
        theta2 = np.linspace(np.pi, 2*np.pi, 50)
        
        x_outer = 1.5 * np.cos(theta1)
        y_outer = 1.5 * np.sin(theta1)
        
        x_inner = 0.8 * np.cos(theta2) + 0.5
        y_inner = 0.8 * np.sin(theta2)
        
        # Create crescent shape
        x_crescent = np.concatenate([x_outer, x_inner])
        y_crescent = np.concatenate([y_outer, y_inner])
        
        ax.fill(x_crescent, y_crescent, alpha=0.3, color='red', label='Non-Convex Set')
        ax.plot(x_crescent, y_crescent, 'r-', linewidth=2)
        
        # Show line segment that goes outside
        p1 = np.array([-1, 0.5])
        p2 = np.array([1, 0.5])
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', linewidth=3, label='Line segment exits set')
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo', markersize=8)
        
        ax.set_title('Non-Convex Set\nLine segment can exit set')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 5. Optimization Comparison
        ax = axes[1, 1]
        
        x = np.linspace(-3, 3, 1000)
        
        # Convex function
        y_conv = x**2
        ax.plot(x, y_conv, 'b-', linewidth=2, label='Convex (one minimum)')
        
        # Non-convex function
        y_nonconv = x**4 - 4*x**2 + 3
        ax.plot(x, y_nonconv, 'r-', linewidth=2, label='Non-convex (multiple minima)')
        
        # Mark minima
        ax.plot(0, 0, 'bo', markersize=10, label='Global minimum (convex)')
        
        # Find local minima for non-convex
        from scipy.optimize import minimize_scalar
        local_minima = []
        for start in [-2, 0, 2]:
            result = minimize_scalar(lambda x: x**4 - 4*x**2 + 3, bounds=(-3, 3), method='bounded')
            if abs(result.x) not in [abs(m) for m in local_minima]:
                local_minima.append(result.x)
        
        for x_min in local_minima:
            y_min = x_min**4 - 4*x_min**2 + 3
            ax.plot(x_min, y_min, 'ro', markersize=8)
        
        ax.set_title('Optimization Landscape\nConvex vs Non-Convex')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Gradient Descent Behavior
        ax = axes[1, 2]
        
        # Different starting points for non-convex function
        x = np.linspace(-3, 3, 1000)
        y = x**4 - 4*x**2 + 3
        
        ax.plot(x, y, 'k-', linewidth=2, label='f(x) = x⁴ - 4x² + 3')
        
        # Run gradient descent from different starting points
        starting_points = [-2.5, -0.5, 0.5, 2.5]
        colors = ['red', 'blue', 'green', 'purple']
        
        for start, color in zip(starting_points, colors):
            x_curr = start
            trajectory_x = [x_curr]
            
            for _ in range(100):
                gradient = 4*x_curr**3 - 8*x_curr
                x_curr = x_curr - 0.01 * gradient
                trajectory_x.append(x_curr)
                
                if abs(gradient) < 0.01:
                    break
            
            trajectory_y = [x**4 - 4*x**2 + 3 for x in trajectory_x]
            ax.plot(trajectory_x, trajectory_y, color=color, alpha=0.7, linewidth=1)
            ax.plot(start, start**4 - 4*start**2 + 3, 'o', color=color, markersize=8, 
                   label=f'Start: {start}')
            ax.plot(trajectory_x[-1], trajectory_y[-1], 's', color=color, markersize=8)
        
        ax.set_title('Different Starting Points\nLead to Different Minima')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Convexity Insights:")
        print("🔹 Convex functions have unique global minimum")
        print("🔹 Any local minimum of convex function is global")
        print("🔹 Gradient descent on convex functions always converges")
        print("🔹 Non-convex optimization can get stuck in local minima")
        print("🔹 Convex sets: line segments between points stay inside")


# ==========================================
# PROBABILITY INTUITION FOR UNCERTAINTY
# ==========================================

class ProbabilityIntuition:
    """
    Visual understanding of probability and statistics for ML
    """
    
    def __init__(self):
        pass
    
    def visualize_probability_distributions(self):
        """
        Interactive visualization of common probability distributions
        """
        print("\n🎯 Probability Distributions in Machine Learning")
        print("=" * 55)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Normal Distribution
        ax = axes[0, 0]
        x = np.linspace(-4, 4, 1000)
        
        # Different parameters
        params = [(0, 1), (0, 0.5), (1, 1)]
        colors = ['blue', 'red', 'green']
        
        for (mu, sigma), color in zip(params, colors):
            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, color=color, linewidth=2, label=f'μ={mu}, σ={sigma}')
        
        ax.set_title('Normal Distribution\nGaussian Bell Curve')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Bernoulli Distribution
        ax = axes[0, 1]
        
        p_values = [0.2, 0.5, 0.8]
        x_pos = [0, 1]
        
        for i, p in enumerate(p_values):
            probs = [1-p, p]
            ax.bar([x + i*0.25 for x in x_pos], probs, width=0.2, 
                  alpha=0.7, label=f'p={p}')
        
        ax.set_title('Bernoulli Distribution\nBinary Outcomes')
        ax.set_xlabel('Outcome (0 or 1)')
        ax.set_ylabel('Probability')
        ax.set_xticks([0.25, 1.25])
        ax.set_xticklabels(['0', '1'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Exponential Distribution
        ax = axes[0, 2]
        x = np.linspace(0, 5, 1000)
        
        lambdas = [0.5, 1, 2]
        for lam in lambdas:
            y = stats.expon.pdf(x, scale=1/lam)
            ax.plot(x, y, linewidth=2, label=f'λ={lam}')
        
        ax.set_title('Exponential Distribution\nWaiting Times')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Beta Distribution
        ax = axes[1, 0]
        x = np.linspace(0, 1, 1000)
        
        params = [(1, 1), (2, 5), (5, 2), (2, 2)]
        for alpha, beta in params:
            y = stats.beta.pdf(x, alpha, beta)
            ax.plot(x, y, linewidth=2, label=f'α={alpha}, β={beta}')
        
        ax.set_title('Beta Distribution\nProbabilities as Outcomes')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Gamma Distribution
        ax = axes[1, 1]
        x = np.linspace(0, 10, 1000)
        
        params = [(1, 1), (2, 1), (3, 1), (2, 2)]
        for shape, scale in params:
            y = stats.gamma.pdf(x, shape, scale=scale)
            ax.plot(x, y, linewidth=2, label=f'k={shape}, θ={scale}')
        
        ax.set_title('Gamma Distribution\nPositive Continuous Values')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Poisson Distribution
        ax = axes[1, 2]
        x = np.arange(0, 15)
        
        lambdas = [1, 3, 5]
        for lam in lambdas:
            y = stats.poisson.pmf(x, lam)
            ax.plot(x, y, 'o-', linewidth=2, markersize=6, label=f'λ={lam}')
        
        ax.set_title('Poisson Distribution\nCounting Events')
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Multivariate Normal
        ax = axes[2, 0]
        
        # Create 2D Gaussian
        x, y = np.mgrid[-3:3:.1, -3:3:.1]
        pos = np.dstack((x, y))
        
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        rv = stats.multivariate_normal(mean, cov)
        
        contours = ax.contour(x, y, rv.pdf(pos), levels=10)
        ax.clabel(contours, inline=True, fontsize=8)
        ax.set_title('Multivariate Normal\n2D Gaussian')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        
        # 8. Central Limit Theorem Demo
        ax = axes[2, 1]
        
        # Sample from uniform distribution and show CLT
        np.random.seed(42)
        n_samples = 1000
        sample_sizes = [1, 5, 30]
        
        for n in sample_sizes:
            # Sample means from uniform distribution
            means = []
            for _ in range(n_samples):
                sample = np.random.uniform(0, 1, n)
                means.append(np.mean(sample))
            
            ax.hist(means, bins=30, alpha=0.5, density=True, label=f'n={n}')
        
        ax.set_title('Central Limit Theorem\nSample Means Become Normal')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Law of Large Numbers
        ax = axes[2, 2]
        
        # Coin flipping simulation
        np.random.seed(42)
        n_flips = 1000
        flips = np.random.binomial(1, 0.5, n_flips)
        running_average = np.cumsum(flips) / np.arange(1, n_flips + 1)
        
        ax.plot(range(1, n_flips + 1), running_average, 'b-', linewidth=1)
        ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='True Probability')
        ax.fill_between(range(1, n_flips + 1), 0.45, 0.55, alpha=0.2, color='red')
        
        ax.set_title('Law of Large Numbers\nConvergence to True Probability')
        ax.set_xlabel('Number of Flips')
        ax.set_ylabel('Running Average')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 0.7)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Distribution Applications in ML:")
        print("🔹 Normal: Feature distributions, noise models")
        print("🔹 Bernoulli: Binary classification outputs")
        print("🔹 Beta: Bayesian priors for probabilities")
        print("🔹 Gamma: Positive continuous variables")
        print("🔹 Poisson: Count data modeling")
        print("🔹 Multivariate Normal: High-dimensional data")
    
    def demonstrate_bayes_theorem_intuition(self):
        """
        Visual demonstration of Bayes' theorem and its ML applications
        """
        print("\n🎯 Bayes' Theorem: The Foundation of Probabilistic ML")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Medical Diagnosis Example
        ax = axes[0, 0]
        
        # Disease prevalence: 1%
        # Test accuracy: 99% (both sensitivity and specificity)
        disease_rate = 0.01
        test_accuracy = 0.99
        
        # Bayes' theorem calculation
        p_test_given_disease = test_accuracy  # Sensitivity
        p_test_given_no_disease = 1 - test_accuracy  # 1 - Specificity
        p_disease = disease_rate
        p_no_disease = 1 - disease_rate
        
        # P(Disease | Positive Test)
        p_disease_given_positive = (p_test_given_disease * p_disease) / \
                                  (p_test_given_disease * p_disease + 
                                   p_test_given_no_disease * p_no_disease)
        
        # Visualization
        categories = ['Prior\nP(Disease)', 'Likelihood\nP(Test|Disease)', 'Posterior\nP(Disease|Test)']
        probabilities = [p_disease, p_test_given_disease, p_disease_given_positive]
        
        bars = ax.bar(categories, probabilities, color=['blue', 'green', 'red'], alpha=0.7)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Medical Diagnosis with Bayes\nSurprisingly Low Posterior!')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 2. Bayesian Updates
        ax = axes[0, 1]
        
        # Coin bias estimation
        true_bias = 0.7
        n_experiments = 50
        
        # Prior belief (uniform)
        alpha_prior, beta_prior = 1, 1
        
        # Simulate coin flips and update belief
        np.random.seed(42)
        outcomes = np.random.binomial(1, true_bias, n_experiments)
        
        alphas, betas = [alpha_prior], [beta_prior]
        posteriors = []
        
        x = np.linspace(0, 1, 100)
        
        for i, outcome in enumerate(outcomes):
            # Update Beta distribution parameters
            if outcome == 1:  # Heads
                alphas.append(alphas[-1] + 1)
                betas.append(betas[-1])
            else:  # Tails
                alphas.append(alphas[-1])
                betas.append(betas[-1] + 1)
            
            # Store posterior for visualization
            if i in [0, 4, 9, 24, 49]:
                posterior = stats.beta.pdf(x, alphas[-1], betas[-1])
                posteriors.append((i+1, posterior))
        
        # Plot evolution of belief
        colors = plt.cm.viridis(np.linspace(0, 1, len(posteriors)))
        for (n_obs, posterior), color in zip(posteriors, colors):
            ax.plot(x, posterior, color=color, linewidth=2, label=f'After {n_obs} flips')
        
        ax.axvline(x=true_bias, color='red', linestyle='--', linewidth=2, label='True Bias')
        ax.set_title('Bayesian Learning\nUpdating Belief About Coin Bias')
        ax.set_xlabel('Coin Bias')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Classification with Uncertainty
        ax = axes[1, 0]
        
        # Generate data with uncertainty
        np.random.seed(42)
        n_points = 100
        
        # Two classes with some overlap
        class_0 = np.random.multivariate_normal([1, 1], [[0.5, 0.2], [0.2, 0.5]], n_points//2)
        class_1 = np.random.multivariate_normal([2, 2], [[0.5, -0.2], [-0.2, 0.5]], n_points//2)
        
        # Plot data points
        ax.scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Class 0')
        ax.scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Class 1')
        
        # Create decision boundary with uncertainty
        x_grid = np.linspace(-1, 4, 50)
        y_grid = np.linspace(-1, 4, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Simple probabilistic classifier (distance-based)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Compute probabilities based on distances to class centers
        center_0, center_1 = np.mean(class_0, axis=0), np.mean(class_1, axis=0)
        
        dist_0 = np.linalg.norm(grid_points - center_0, axis=1)
        dist_1 = np.linalg.norm(grid_points - center_1, axis=1)
        
        # Convert distances to probabilities
        prob_class_1 = 1 / (1 + np.exp(dist_1 - dist_0))
        prob_grid = prob_class_1.reshape(X_grid.shape)
        
        # Plot probability contours
        contours = ax.contour(X_grid, Y_grid, prob_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                             colors='black', alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Highlight decision boundary
        ax.contour(X_grid, Y_grid, prob_grid, levels=[0.5], colors='black', linewidths=3)
        
        ax.set_title('Probabilistic Classification\nUncertainty via Probability Contours')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        
        # 4. Prior vs Posterior Comparison
        ax = axes[1, 1]
        
        # Gaussian prior vs posterior after seeing data
        x = np.linspace(-5, 15, 1000)
        
        # Prior belief about parameter
        prior_mean, prior_std = 5, 3
        prior = stats.norm.pdf(x, prior_mean, prior_std)
        
        # Observed data
        observed_data = [8, 9, 7, 10, 8.5]  # Sample with mean ≈ 8.5
        data_mean = np.mean(observed_data)
        data_std = 1  # Assumed known
        n = len(observed_data)
        
        # Posterior (conjugate prior)
        posterior_precision = 1/prior_std**2 + n/data_std**2
        posterior_std = 1/np.sqrt(posterior_precision)
        posterior_mean = (prior_mean/prior_std**2 + n*data_mean/data_std**2) / posterior_precision
        
        posterior = stats.norm.pdf(x, posterior_mean, posterior_std)
        
        ax.plot(x, prior, 'b-', linewidth=3, label=f'Prior: N({prior_mean}, {prior_std}²)')
        ax.plot(x, posterior, 'r-', linewidth=3, 
               label=f'Posterior: N({posterior_mean:.1f}, {posterior_std:.1f}²)')
        ax.axvline(x=data_mean, color='green', linestyle='--', linewidth=2, 
                  label=f'Data Mean: {data_mean}')
        
        # Fill areas to show the update
        ax.fill_between(x, 0, prior, alpha=0.2, color='blue')
        ax.fill_between(x, 0, posterior, alpha=0.2, color='red')
        
        ax.set_title('Bayesian Parameter Estimation\nPrior Knowledge + Data = Posterior')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n💡 Medical Diagnosis Insight:")
        print(f"   Even with 99% accurate test, P(Disease|Positive) = {p_disease_given_positive:.1%}")
        print(f"   Base rate matters! Low prevalence means many false positives.")
        
        print(f"\n💡 Bayesian Learning Insights:")
        print("🔹 Prior beliefs get updated with evidence")
        print("🔹 More data → more confident posterior")
        print("🔹 Uncertainty quantification is built-in")
        print("🔹 Ideal for learning with limited data")


# ==========================================
# INFORMATION THEORY INTUITION
# ==========================================

class InformationTheoryIntuition:
    """
    Visual understanding of information theory concepts in ML
    """
    
    def __init__(self):
        pass
    
    def visualize_entropy_and_information(self):
        """
        Demonstrate entropy, mutual information, and KL divergence
        """
        print("\n🎯 Information Theory: Measuring Uncertainty and Information")
        print("=" * 65)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Entropy of Binary Distribution
        ax = axes[0, 0]
        
        p_values = np.linspace(0.01, 0.99, 100)
        entropy_values = [-p*np.log2(p) - (1-p)*np.log2(1-p) for p in p_values]
        
        ax.plot(p_values, entropy_values, 'b-', linewidth=3)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Maximum Entropy')
        ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
        
        # Mark special points
        ax.plot(0.5, 1, 'ro', markersize=10, label='p=0.5, H=1')
        ax.plot([0.1, 0.9], [entropy_values[9], entropy_values[89]], 'go', markersize=8, 
               label='Low entropy')
        
        ax.set_title('Binary Entropy Function\nH(p) = -p log₂(p) - (1-p) log₂(1-p)')
        ax.set_xlabel('Probability p')
        ax.set_ylabel('Entropy (bits)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Mutual Information Visualization
        ax = axes[0, 1]
        
        # Create two correlated random variables
        np.random.seed(42)
        n_samples = 1000
        
        # Independent case
        x_indep = np.random.randn(n_samples)
        y_indep = np.random.randn(n_samples)
        
        # Correlated case
        x_corr = np.random.randn(n_samples)
        y_corr = 0.8 * x_corr + 0.6 * np.random.randn(n_samples)
        
        # Plot both cases
        ax.scatter(x_indep, y_indep, alpha=0.3, s=10, color='blue', label='Independent (MI ≈ 0)')
        ax.scatter(x_corr + 4, y_corr, alpha=0.3, s=10, color='red', label='Correlated (MI > 0)')
        
        ax.set_title('Mutual Information\nMI measures dependence between variables')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display MI (simplified estimation)
        def estimate_mi(x, y, bins=20):
            """Estimate mutual information using histograms"""
            # Create 2D histogram
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
            hist_2d = hist_2d + 1e-10  # Avoid log(0)
            
            # Marginal distributions
            hist_x = np.sum(hist_2d, axis=1)
            hist_y = np.sum(hist_2d, axis=0)
            
            # Normalize to probabilities
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # Compute MI
            mi = 0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return mi
        
        mi_indep = estimate_mi(x_indep, y_indep)
        mi_corr = estimate_mi(x_corr, y_corr)
        
        ax.text(0.02, 0.98, f'MI(independent) ≈ {mi_indep:.2f} bits', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(0.02, 0.85, f'MI(correlated) ≈ {mi_corr:.2f} bits', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # 3. KL Divergence Visualization
        ax = axes[0, 2]
        
        x = np.linspace(-4, 6, 1000)
        
        # Reference distribution P
        p_dist = stats.norm.pdf(x, 0, 1)
        
        # Different Q distributions
        q_distributions = [
            (stats.norm.pdf(x, 0, 1), "Q = P", "blue"),
            (stats.norm.pdf(x, 1, 1), "Q shifted", "red"),
            (stats.norm.pdf(x, 0, 1.5), "Q wider", "green"),
            (stats.norm.pdf(x, 2, 0.5), "Q shifted+narrow", "purple")
        ]
        
        ax.plot(x, p_dist, 'k-', linewidth=3, label='P (reference)')
        
        kl_values = []
        for q_dist, label, color in q_distributions:
            ax.plot(x, q_dist, color=color, linewidth=2, label=label, alpha=0.7)
            
            # Calculate KL divergence
            # KL(P||Q) = ∫ P(x) log(P(x)/Q(x)) dx
            # Approximate with discrete sum
            dx = x[1] - x[0]
            kl = np.sum(p_dist * np.log(p_dist / (q_dist + 1e-10))) * dx
            kl_values.append(kl)
        
        ax.set_title('KL Divergence D(P||Q)\nMeasures difference between distributions')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add KL divergence values as text
        for i, (_, label, _) in enumerate(q_distributions):
            ax.text(0.02, 0.95 - i*0.1, f'KL({label}) = {kl_values[i]:.2f}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"))
        
        # 4. Decision Tree Information Gain
        ax = axes[1, 0]
        
        # Simulate decision tree split
        np.random.seed(42)
        
        # Create dataset
        n_samples = 100
        feature = np.random.randn(n_samples)
        # Labels based on feature value with some noise
        labels = (feature > 0).astype(int)
        noise_indices = np.random.choice(n_samples, size=15, replace=False)
        labels[noise_indices] = 1 - labels[noise_indices]  # Add noise
        
        # Calculate entropy before split
        p_class1 = np.mean(labels)
        entropy_before = -p_class1 * np.log2(p_class1 + 1e-10) - (1-p_class1) * np.log2(1-p_class1 + 1e-10)
        
        # Try different split points
        split_points = np.linspace(-2, 2, 50)
        information_gains = []
        
        for split in split_points:
            left_mask = feature <= split
            right_mask = feature > split
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                information_gains.append(0)
                continue
            
            # Calculate weighted entropy after split
            left_labels = labels[left_mask]
            right_labels = labels[right_mask]
            
            p_left_class1 = np.mean(left_labels) if len(left_labels) > 0 else 0
            p_right_class1 = np.mean(right_labels) if len(right_labels) > 0 else 0
            
            entropy_left = 0 if p_left_class1 == 0 or p_left_class1 == 1 else \
                          -p_left_class1 * np.log2(p_left_class1) - (1-p_left_class1) * np.log2(1-p_left_class1)
            entropy_right = 0 if p_right_class1 == 0 or p_right_class1 == 1 else \
                           -p_right_class1 * np.log2(p_right_class1) - (1-p_right_class1) * np.log2(1-p_right_class1)
            
            weighted_entropy = (np.sum(left_mask) * entropy_left + np.sum(right_mask) * entropy_right) / n_samples
            information_gain = entropy_before - weighted_entropy
            information_gains.append(information_gain)
        
        ax.plot(split_points, information_gains, 'b-', linewidth=2)
        
        # Mark best split
        best_split_idx = np.argmax(information_gains)
        best_split = split_points[best_split_idx]
        best_gain = information_gains[best_split_idx]
        
        ax.plot(best_split, best_gain, 'ro', markersize=10, label=f'Best split: {best_split:.2f}')
        ax.axvline(x=best_split, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Information Gain vs Split Point\nEntropy before: {entropy_before:.2f} bits')
        ax.set_xlabel('Split Point')
        ax.set_ylabel('Information Gain (bits)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Cross-entropy Loss Visualization
        ax = axes[1, 1]
        
        # Binary cross-entropy
        y_true = 1  # True label
        p_pred = np.linspace(0.001, 0.999, 1000)  # Predicted probabilities
        
        cross_entropy = -np.log(p_pred)  # For true label = 1
        cross_entropy_0 = -np.log(1 - p_pred)  # For true label = 0
        
        ax.plot(p_pred, cross_entropy, 'r-', linewidth=3, label='True label = 1')
        ax.plot(p_pred, cross_entropy_0, 'b-', linewidth=3, label='True label = 0')
        
        # Mark specific points
        ax.plot(0.5, -np.log(0.5), 'go', markersize=8, label='Uncertain prediction')
        ax.plot(0.9, -np.log(0.9), 'ro', markersize=8, label='Confident correct')
        ax.plot(0.1, -np.log(0.1), 'mo', markersize=8, label='Confident wrong')
        
        ax.set_title('Cross-Entropy Loss\nPenalizes confident wrong predictions')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. Compression and Information
        ax = axes[1, 2]
        
        # Show relationship between probability and code length
        probabilities = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
        optimal_code_lengths = -np.log2(probabilities)  # Shannon's optimal code length
        
        # Practical code lengths (must be integers)
        practical_code_lengths = np.ceil(optimal_code_lengths)
        
        x_pos = np.arange(len(probabilities))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, optimal_code_lengths, width, 
                      label='Optimal (Shannon)', alpha=0.7, color='blue')
        bars2 = ax.bar(x_pos + width/2, practical_code_lengths, width,
                      label='Practical (integer)', alpha=0.7, color='red')
        
        # Add value labels
        for i, (opt, prac) in enumerate(zip(optimal_code_lengths, practical_code_lengths)):
            ax.text(i - width/2, opt + 0.1, f'{opt:.1f}', ha='center', va='bottom')
            ax.text(i + width/2, prac + 0.1, f'{int(prac)}', ha='center', va='bottom')
        
        ax.set_title('Information Content\nRare events need more bits')
        ax.set_xlabel('Event')
        ax.set_ylabel('Code Length (bits)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'p={p:.3f}' for p in probabilities], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Information Theory in ML:")
        print("🔹 Entropy measures uncertainty/information content")
        print("🔹 Mutual Information quantifies variable dependence")
        print("🔹 KL Divergence measures distribution differences")
        print("🔹 Information Gain guides decision tree splits")
        print("🔹 Cross-entropy loss minimizes KL divergence")
        print("🔹 Compression connects to optimal representation")


# ==========================================
# COMPREHENSIVE MATHEMATICAL INTUITION DEMO
# ==========================================

def comprehensive_mathematical_intuition_demo():
    """
    Complete demonstration of mathematical intuition for ML
    """
    print("🧮 Neural Odyssey - Week 11: Mathematical Intuition Development")
    print("=" * 70)
    print("Building Deep Understanding: From Abstract Math to ML Applications")
    print("=" * 70)
    
    # Linear Algebra Intuition
    print("\n" + "="*70)
    print("📐 LINEAR ALGEBRA: The Geometry of Data")
    print("="*70)
    
    la_intuition = LinearAlgebraIntuition()
    la_intuition.visualize_vector_operations()
    la_intuition.visualize_matrix_interpretations()
    la_intuition.demonstrate_dimensionality_curse()
    
    # Calculus Intuition
    print("\n" + "="*70)
    print("📈 CALCULUS: The Mathematics of Change and Optimization")
    print("="*70)
    
    calc_intuition = CalculusIntuition()
    calc_intuition.visualize_gradient_descent_landscape()
    calc_intuition.demonstrate_convexity_intuition()
    
    # Probability Intuition
    print("\n" + "="*70)
    print("🎲 PROBABILITY: Handling Uncertainty and Inference")
    print("="*70)
    
    prob_intuition = ProbabilityIntuition()
    prob_intuition.visualize_probability_distributions()
    prob_intuition.demonstrate_bayes_theorem_intuition()
    
    # Information Theory Intuition
    print("\n" + "="*70)
    print("📊 INFORMATION THEORY: Measuring and Encoding Information")
    print("="*70)
    
    info_intuition = InformationTheoryIntuition()
    info_intuition.visualize_entropy_and_information()
    
    # Final Integration
    print("\n" + "="*70)
    print("🎓 MATHEMATICAL INTUITION MASTERY COMPLETE!")
    print("="*70)
    
    key_insights = [
        "🔹 Linear Algebra: Data lives in high-dimensional vector spaces",
        "🔹 Matrices transform and relate different vector spaces", 
        "🔹 Eigenvalues reveal the 'natural directions' of transformations",
        "🔹 Gradients point toward steepest increase (opposite for descent)",
        "🔹 Convex functions have unique global minima (easy optimization)",
        "🔹 Probability distributions model uncertainty and randomness",
        "🔹 Bayes' theorem updates beliefs with new evidence",
        "🔹 Entropy measures information content and uncertainty",
        "🔹 All these concepts work together in ML algorithms"
    ]
    
    print("\n💡 Key Mathematical Insights:")
    for insight in key_insights:
        print(f"   {insight}")
    
    ml_connections = [
        "🧠 Neural Networks: Linear algebra + calculus + probability",
        "🎯 Optimization: Calculus concepts guide parameter updates",
        "📊 Bayesian ML: Probability theory for uncertainty quantification",
        "🌳 Decision Trees: Information theory for optimal splits",
        "📈 Loss Functions: Connect geometry to learning objectives",
        "🔍 Dimensionality: Understanding curse guides feature selection",
        "⚖️ Regularization: Geometric constraints prevent overfitting"
    ]
    
    print(f"\n🔗 Connections to Machine Learning:")
    for connection in ml_connections:
        print(f"   {connection}")
    
    mental_models = [
        "📐 Think of data as points in geometric space",
        "🎯 Optimization as navigating landscapes to find valleys",
        "🎲 Uncertainty as probability distributions over possibilities",
        "📊 Information as surprise - rare events carry more bits",
        "🔄 Learning as iterative updates using mathematical principles",
        "🧭 Intuition comes from visualizing abstract concepts"
    ]
    
    print(f"\n🧠 Mental Models for Mathematical Thinking:")
    for model in mental_models:
        print(f"   {model}")
    
    print(f"\n🚀 Ready for Phase 2: Core Machine Learning!")
    print("   Your mathematical foundation is now rock-solid.")
    print("   Next: Apply these concepts to real ML algorithms.")
    
    return {
        'linear_algebra': la_intuition,
        'calculus': calc_intuition,
        'probability': prob_intuition,
        'information_theory': info_intuition
    }


# ==========================================
# INTERACTIVE MATHEMATICAL EXPLORATION
# ==========================================

class InteractiveMathExplorer:
    """
    Interactive tools for exploring mathematical concepts
    """
    
    def __init__(self):
        pass
    
    def create_concept_map(self):
        """
        Create visual concept map of mathematical relationships
        """
        print("\n🗺️  Mathematical Concept Map for ML")
        print("=" * 40)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define concept positions and connections
        concepts = {
            'Linear Algebra': (2, 8),
            'Vectors': (1, 7),
            'Matrices': (3, 7),
            'Eigenvalues': (2, 6),
            'SVD': (4, 6),
            
            'Calculus': (8, 8),
            'Derivatives': (7, 7),
            'Gradients': (9, 7),
            'Optimization': (8, 6),
            'Convexity': (10, 6),
            
            'Probability': (2, 4),
            'Distributions': (1, 3),
            'Bayes Theorem': (3, 3),
            'Random Variables': (2, 2),
            
            'Information Theory': (8, 4),
            'Entropy': (7, 3),
            'KL Divergence': (9, 3),
            'Mutual Information': (8, 2),
            
            'Machine Learning': (5, 1),
            'Neural Networks': (3, 0.5),
            'Decision Trees': (5, 0.5),
            'SVM': (7, 0.5)
        }
        
        # Define connections
        connections = [
            ('Linear Algebra', 'Vectors'),
            ('Linear Algebra', 'Matrices'),
            ('Matrices', 'Eigenvalues'),
            ('Matrices', 'SVD'),
            ('Calculus', 'Derivatives'),
            ('Calculus', 'Gradients'),
            ('Gradients', 'Optimization'),
            ('Optimization', 'Convexity'),
            ('Probability', 'Distributions'),
            ('Probability', 'Bayes Theorem'),
            ('Probability', 'Random Variables'),
            ('Information Theory', 'Entropy'),
            ('Information Theory', 'KL Divergence'),
            ('Information Theory', 'Mutual Information'),
            ('Linear Algebra', 'Neural Networks'),
            ('Calculus', 'Neural Networks'),
            ('Probability', 'Neural Networks'),
            ('Information Theory', 'Decision Trees'),
            ('Linear Algebra', 'SVM'),
            ('Optimization', 'SVM'),
            ('Machine Learning', 'Neural Networks'),
            ('Machine Learning', 'Decision Trees'),
            ('Machine Learning', 'SVM')
        ]
        
        # Draw connections
        for start, end in connections:
            start_pos = concepts[start]
            end_pos = concepts[end]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   'k-', alpha=0.3, linewidth=1)
        
        # Draw concepts
        colors = {
            'Linear Algebra': 'lightblue', 'Vectors': 'lightblue', 'Matrices': 'lightblue',
            'Eigenvalues': 'lightblue', 'SVD': 'lightblue',
            'Calculus': 'lightgreen', 'Derivatives': 'lightgreen', 'Gradients': 'lightgreen',
            'Optimization': 'lightgreen', 'Convexity': 'lightgreen',
            'Probability': 'lightcoral', 'Distributions': 'lightcoral', 'Bayes Theorem': 'lightcoral',
            'Random Variables': 'lightcoral',
            'Information Theory': 'lightyellow', 'Entropy': 'lightyellow', 'KL Divergence': 'lightyellow',
            'Mutual Information': 'lightyellow',
            'Machine Learning': 'lightgray', 'Neural Networks': 'lightgray', 
            'Decision Trees': 'lightgray', 'SVM': 'lightgray'
        }
        
        for concept, (x, y) in concepts.items():
            ax.scatter(x, y, s=1000, c=colors[concept], alpha=0.7, edgecolors='black')
            ax.text(x, y, concept, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 9)
        ax.set_title('Mathematical Concept Map for Machine Learning', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Linear Algebra'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Calculus'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Probability'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', 
                      markersize=10, label='Information Theory'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                      markersize=10, label='ML Applications')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        print("💡 This map shows how mathematical concepts connect to create ML!")
    
    def mathematical_intuition_quiz(self):
        """
        Interactive quiz to test mathematical intuition
        """
        print("\n🎯 Mathematical Intuition Quiz")
        print("=" * 35)
        
        questions = [
            {
                "question": "What does the dot product geometrically represent?",
                "options": ["A) Vector addition", "B) Projection and angle", "C) Matrix multiplication", "D) Cross product"],
                "answer": "B",
                "explanation": "Dot product = |a||b|cos(θ), measuring projection of one vector onto another"
            },
            {
                "question": "Why do we want convex loss functions in ML?",
                "options": ["A) They're faster to compute", "B) They guarantee global minimum", "C) They're more accurate", "D) They use less memory"],
                "answer": "B",
                "explanation": "Convex functions have unique global minima, ensuring optimization algorithms converge to the best solution"
            },
            {
                "question": "What does high entropy indicate about a probability distribution?",
                "options": ["A) Low uncertainty", "B) High uncertainty", "C) Normal distribution", "D) Biased distribution"],
                "answer": "B",
                "explanation": "High entropy means high uncertainty - the distribution is spread out with no clear peak"
            },
            {
                "question": "In high dimensions, what happens to the distance between random points?",
                "options": ["A) They get closer", "B) They become more varied", "C) They become similar", "D) Nothing changes"],
                "answer": "C",
                "explanation": "Curse of dimensionality: all points become roughly equidistant in high dimensions"
            },
            {
                "question": "What does the gradient vector point towards?",
                "options": ["A) Minimum", "B) Maximum", "C) Steepest increase", "D) Steepest decrease"],
                "answer": "C",
                "explanation": "Gradient points in direction of steepest increase; we go opposite direction for gradient descent"
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
            print("🏆 Perfect! Your mathematical intuition is excellent!")
        elif score >= len(questions) * 0.8:
            print("👍 Great job! You have strong mathematical understanding.")
        elif score >= len(questions) * 0.6:
            print("📚 Good foundation, but review the concepts you missed.")
        else:
            print("🔄 Keep studying! Mathematical intuition takes time to develop.")
        
        return score


# ==========================================
# MAIN EXECUTION AND TESTING
# ==========================================

if __name__ == "__main__":
    """
    Run this file to develop deep mathematical intuition for ML!
    
    This comprehensive exploration covers:
    1. Geometric understanding of linear algebra
    2. Visual intuition for calculus and optimization
    3. Probabilistic thinking and Bayesian reasoning
    4. Information theory concepts and applications
    5. Interactive exploration and concept mapping
    
    To get started, run: python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey - Week 11: Mathematical Intuition!")
    print("Develop deep understanding that bridges abstract math to practical ML.")
    print("\nThis mathematical journey includes:")
    print("1. 📐 Geometric intuition for linear algebra operations")
    print("2. 📈 Visual understanding of calculus and optimization")
    print("3. 🎲 Probabilistic thinking and Bayesian reasoning")
    print("4. 📊 Information theory for measuring uncertainty")
    print("5. 🗺️  Interactive concept mapping and exploration")
    print("6. 🎯 Mathematical intuition assessment")
    
    # Run the comprehensive demonstration
    print("\n" + "="*70)
    print("🧮 Starting Mathematical Intuition Development...")
    print("="*70)
    
    # Main mathematical intuition demo
    math_components = comprehensive_mathematical_intuition_demo()
    
    # Interactive exploration
    print("\n" + "="*70)
    print("🎮 Interactive Mathematical Exploration")
    print("="*70)
    
    explorer = InteractiveMathExplorer()
    explorer.create_concept_map()
    
    # Mathematical intuition quiz
    quiz_score = explorer.mathematical_intuition_quiz()
    
    print("\n" + "="*70)
    print("🎓 Week 11 Complete: Mathematical Intuition Mastered!")
    print("="*70)
    print("🧠 You now have deep intuitive understanding of:")
    print("   ✅ How linear algebra operations work geometrically")
    print("   ✅ Why gradient descent finds minima through calculus")
    print("   ✅ How probability captures uncertainty and enables inference")
    print("   ✅ Why information theory guides optimal representations")
    print("   ✅ How all mathematical concepts interconnect in ML")
    
    print("\n🚀 Ready for Phase 2: Core Machine Learning!")
    print("   Your mathematical foundation provides the intuition needed")
    print("   to understand why ML algorithms work, not just how.")
    
    # Mathematical wisdom
    print("\n🧭 Mathematical Wisdom for ML:")
    wisdom = [
        "Intuition comes from visualization, not memorization",
        "Every ML algorithm has geometric and probabilistic interpretations",
        "Mathematics is the language - ML algorithms are the stories",
        "Understanding 'why' is more powerful than knowing 'how'",
        "Abstract concepts become concrete through visual understanding"
    ]
    
    for insight in wisdom:
        print(f"   💡 {insight}")
    
    print("\n📚 Suggested Practice:")
    practice_suggestions = [
        "Visualize every mathematical operation before computing",
        "Ask 'what does this mean geometrically?' for each concept",
        "Connect probability to real-world uncertainty",
        "Think of optimization as navigation through landscapes",
        "Draw concept maps to see mathematical relationships"
    ]
    
    for suggestion in practice_suggestions:
        print(f"   📖 {suggestion}")
    
    # Return comprehensive results
    final_results = {
        'mathematical_components': math_components,
        'interactive_exploration': explorer,
        'quiz_score': quiz_score,
        'intuition_level': 'Expert' if quiz_score >= 4 else 'Advanced' if quiz_score >= 3 else 'Developing'
    }
    
    print(f"\n🎯 Mathematical Intuition Assessment:")
    print(f"   Quiz score: {quiz_score}/5")
    print(f"   Intuition level: {final_results['intuition_level']}")
    print(f"   Concepts mastered: Linear Algebra, Calculus, Probability, Information Theory")
    print(f"   Ready for advanced ML: {'Yes' if quiz_score >= 3 else 'Review recommended'}")
    
    print("\n🌟 Mathematics is not about computation - it's about understanding!")
    print("    You now see the mathematical beauty underlying machine learning.")