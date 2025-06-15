"""
Neural Odyssey - Week 4: Eigenvalues and Principal Component Analysis
Exercises for mastering the mathematical foundation of dimensionality reduction

This module implements core concepts for understanding data structure:
- Eigenvalues and eigenvectors as special matrix directions
- Eigendecomposition and matrix diagonalization
- Principal Component Analysis from first principles
- Real-world applications: compression, visualization, feature extraction

Complete the TODO functions to build your eigenanalysis toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: EIGENVALUES AND EIGENVECTORS
# ==============================================

def power_iteration(A, max_iterations=1000, tolerance=1e-6):
    """
    TODO: Implement power iteration to find dominant eigenvalue and eigenvector
    
    Power iteration is the simplest eigenvalue algorithm. It repeatedly applies
    the matrix to a vector, which converges to the dominant eigenvector.
    
    This is the foundation of Google's PageRank algorithm!
    
    Args:
        A: Square matrix
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple (eigenvalue, eigenvector) for dominant eigenvalue
    """
    # TODO: Implement power iteration algorithm
    # 1. Start with random vector
    # 2. Repeatedly apply: v = A @ v, then normalize
    # 3. Eigenvalue = v^T @ A @ v (Rayleigh quotient)
    # 4. Check convergence based on eigenvalue change
    
    pass

def characteristic_polynomial_2x2(A):
    """
    TODO: Compute eigenvalues of 2x2 matrix using characteristic polynomial
    
    For 2x2 matrix [[a,b],[c,d]], characteristic polynomial is:
    λ² - (a+d)λ + (ad-bc) = 0
    
    This gives exact eigenvalues using quadratic formula.
    
    Args:
        A: 2x2 matrix
        
    Returns:
        Array of eigenvalues
    """
    # TODO: Implement analytical eigenvalue computation for 2x2 case
    # Use quadratic formula: λ = (trace ± √(trace² - 4*det)) / 2
    
    pass

def compute_eigenvector(A, eigenvalue, tolerance=1e-10):
    """
    TODO: Compute eigenvector for given eigenvalue
    
    Solve (A - λI)v = 0 by finding null space of (A - λI)
    
    Args:
        A: Square matrix
        eigenvalue: Known eigenvalue
        tolerance: Numerical tolerance for zero
        
    Returns:
        Normalized eigenvector
    """
    # TODO: Solve (A - λI)v = 0
    # 1. Form matrix (A - λI)
    # 2. Find null space using SVD or Gaussian elimination
    # 3. Return normalized eigenvector
    
    pass

def full_eigendecomposition(A):
    """
    TODO: Compute complete eigendecomposition A = QΛQ^(-1)
    
    For symmetric matrices, this becomes A = QΛQ^T where Q has orthonormal columns.
    
    Args:
        A: Square matrix (preferably symmetric)
        
    Returns:
        Tuple (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    # TODO: Implement complete eigendecomposition
    # 1. Find all eigenvalues (can use numpy for this step)
    # 2. Find corresponding eigenvectors
    # 3. Sort by eigenvalue magnitude (descending)
    # 4. Normalize eigenvectors
    
    pass

def visualize_eigenvectors_2d(A, title="Eigenvectors Visualization"):
    """
    TODO: Visualize eigenvectors of 2x2 matrix
    
    Shows how eigenvectors are special directions that don't rotate
    under the linear transformation defined by A.
    
    Args:
        A: 2x2 matrix
        title: Plot title
    """
    # TODO: Create visualization showing:
    # 1. Original coordinate axes
    # 2. Eigenvector directions
    # 3. Effect of transformation A on unit circle
    # 4. How eigenvectors maintain their direction
    
    pass

def matrix_powers_eigenanalysis(A, max_power=10):
    """
    TODO: Analyze how matrix powers relate to eigenvalues
    
    Shows that A^n has same eigenvectors as A, but eigenvalues are λ^n.
    This reveals long-term behavior of dynamical systems.
    
    Args:
        A: Square matrix
        max_power: Maximum power to compute
        
    Returns:
        Dictionary with powers and their eigenanalysis
    """
    # TODO: Compute A^1, A^2, ..., A^max_power
    # For each power, compute eigenvalues and compare to λ^n
    # Show how dominant eigenvalue determines long-term behavior
    
    pass

# ==============================================
# PART 2: COVARIANCE MATRICES AND DATA ANALYSIS
# ==============================================

def compute_covariance_matrix(X, center=True):
    """
    TODO: Compute covariance matrix from data
    
    Covariance matrix is fundamental to PCA. Its eigenvalues are
    the variances along principal component directions.
    
    Args:
        X: Data matrix (n_samples, n_features)
        center: Whether to center data before computing covariance
        
    Returns:
        Covariance matrix (n_features, n_features)
    """
    # TODO: Implement covariance matrix computation
    # 1. Center data if requested: X_centered = X - mean(X)
    # 2. Compute covariance: C = X_centered^T @ X_centered / (n-1)
    # 3. Return symmetric covariance matrix
    
    pass

def analyze_data_variance_structure(X):
    """
    TODO: Analyze variance structure of dataset
    
    Shows how variance is distributed across features and
    what the covariance matrix reveals about feature relationships.
    
    Args:
        X: Data matrix
        
    Returns:
        Dictionary with variance analysis results
    """
    # TODO: Compute and analyze:
    # 1. Individual feature variances (diagonal of covariance matrix)
    # 2. Feature correlations and covariances
    # 3. Total variance in dataset
    # 4. Condition number of covariance matrix
    
    pass

def demonstrate_covariance_geometry(X):
    """
    TODO: Visualize geometric meaning of covariance
    
    Shows how covariance matrix defines an ellipse that
    characterizes the data distribution.
    
    Args:
        X: 2D data matrix for visualization
    """
    # TODO: Create visualization showing:
    # 1. Data scatter plot
    # 2. Covariance ellipse
    # 3. Principal axes (eigenvectors of covariance)
    # 4. Relationship between eigenvalues and ellipse shape
    
    pass

# ==============================================
# PART 3: PRINCIPAL COMPONENT ANALYSIS
# ==============================================

def pca_from_scratch(X, n_components=None):
    """
    TODO: Implement complete PCA algorithm from scratch
    
    PCA finds the directions of maximum variance in data by
    computing eigenvectors of the covariance matrix.
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components to keep (default: all)
        
    Returns:
        Dictionary with components, eigenvalues, transformed data, etc.
    """
    # TODO: Implement complete PCA algorithm:
    # 1. Center the data
    # 2. Compute covariance matrix
    # 3. Find eigenvalues and eigenvectors
    # 4. Sort by eigenvalue (descending)
    # 5. Keep top n_components
    # 6. Transform data to principal component space
    # 7. Compute explained variance ratios
    
    pass

def explained_variance_analysis(eigenvalues):
    """
    TODO: Analyze explained variance from eigenvalues
    
    Helps determine how many principal components to keep
    by showing how much variance each component explains.
    
    Args:
        eigenvalues: Array of eigenvalues from PCA
        
    Returns:
        Dictionary with variance analysis
    """
    # TODO: Compute:
    # 1. Individual explained variance ratios
    # 2. Cumulative explained variance
    # 3. Number of components for different variance thresholds (90%, 95%, 99%)
    
    pass

def pca_reconstruction(X_transformed, components, mean_vector, n_components=None):
    """
    TODO: Reconstruct original data from principal components
    
    Shows how PCA can be used for data compression and denoising
    by reconstructing from fewer components.
    
    Args:
        X_transformed: Data in principal component space
        components: Principal component vectors
        mean_vector: Original data mean
        n_components: Number of components to use for reconstruction
        
    Returns:
        Reconstructed data matrix
    """
    # TODO: Implement PCA reconstruction:
    # 1. Take first n_components of transformed data
    # 2. Project back to original space: X_reconstructed = X_transformed @ components^T
    # 3. Add back the mean
    
    pass

def visualize_pca_2d(X, pca_result):
    """
    TODO: Visualize PCA results for 2D data
    
    Shows original data, principal components, and transformed data.
    
    Args:
        X: Original 2D data
        pca_result: Result dictionary from pca_from_scratch
    """
    # TODO: Create visualization with:
    # 1. Original data scatter plot
    # 2. Principal component vectors overlaid
    # 3. Data projected onto principal components
    # 4. Explained variance for each component
    
    pass

def dimensionality_reduction_demo(X, target_variance=0.95):
    """
    TODO: Demonstrate dimensionality reduction with PCA
    
    Shows how PCA can reduce dimensionality while preserving
    most of the variance in the data.
    
    Args:
        X: High-dimensional data
        target_variance: Fraction of variance to preserve
        
    Returns:
        Dictionary with reduction results
    """
    # TODO: 
    # 1. Apply PCA to find all components
    # 2. Determine number of components needed for target variance
    # 3. Reduce dimensionality
    # 4. Measure reconstruction error
    # 5. Compare original vs. reduced data
    
    pass

# ==============================================
# PART 4: REAL-WORLD APPLICATIONS
# ==============================================

def eigenfaces_analysis(face_images):
    """
    TODO: Implement eigenfaces for facial recognition
    
    Eigenfaces are the principal components of face images.
    They reveal the most important facial variations.
    
    Args:
        face_images: Array of face images (n_faces, height, width)
        
    Returns:
        Dictionary with eigenfaces and analysis results
    """
    # TODO: Implement eigenfaces:
    # 1. Reshape face images to vectors
    # 2. Apply PCA to find principal "face directions"
    # 3. Visualize eigenfaces as images
    # 4. Show reconstruction quality vs. number of eigenfaces
    # 5. Implement face recognition using eigenface projections
    
    pass

def data_compression_pca(data, compression_ratios=[0.1, 0.25, 0.5, 0.75]):
    """
    TODO: Use PCA for data compression
    
    Shows how PCA can compress data by keeping only
    the most important principal components.
    
    Args:
        data: Original data matrix
        compression_ratios: Fractions of components to keep
        
    Returns:
        Dictionary with compression results
    """
    # TODO: For each compression ratio:
    # 1. Determine number of components to keep
    # 2. Apply PCA and keep only those components
    # 3. Reconstruct data
    # 4. Compute compression ratio and reconstruction error
    # 5. Visualize quality vs. compression tradeoff
    
    pass

def feature_extraction_pca(X, y, n_components=2):
    """
    TODO: Use PCA for feature extraction in classification
    
    Shows how PCA can extract meaningful features
    that preserve class separability.
    
    Args:
        X: Feature matrix
        y: Class labels
        n_components: Number of principal components to extract
        
    Returns:
        Dictionary with extracted features and analysis
    """
    # TODO: 
    # 1. Apply PCA to extract n_components features
    # 2. Visualize class separation in PC space
    # 3. Compare class separability before/after PCA
    # 4. Analyze which original features contribute most to each PC
    
    pass

def anomaly_detection_pca(X, contamination=0.1):
    """
    TODO: Use PCA for anomaly detection
    
    Anomalies often have large reconstruction errors when
    projected to principal component subspace.
    
    Args:
        X: Data matrix
        contamination: Expected fraction of anomalies
        
    Returns:
        Dictionary with anomaly detection results
    """
    # TODO: Implement PCA-based anomaly detection:
    # 1. Apply PCA with reduced dimensionality
    # 2. Reconstruct all data points
    # 3. Compute reconstruction errors
    # 4. Identify anomalies as points with highest errors
    # 5. Visualize normal vs. anomalous points
    
    pass

def noise_reduction_pca(noisy_data, noise_components=None):
    """
    TODO: Use PCA for noise reduction
    
    Assumes signal is captured by top principal components
    and noise is distributed across many small components.
    
    Args:
        noisy_data: Data with noise
        noise_components: Number of small components assumed to be noise
        
    Returns:
        Denoised data
    """
    # TODO: Implement PCA denoising:
    # 1. Apply PCA to noisy data
    # 2. Identify noise components (smallest eigenvalues)
    # 3. Reconstruct using only signal components
    # 4. Compare original, noisy, and denoised data
    
    pass

# ==============================================
# PART 5: ADVANCED PCA TECHNIQUES
# ==============================================

class IncrementalPCA:
    """
    TODO: Implement incremental PCA for large datasets
    
    Regular PCA requires all data in memory. Incremental PCA
    updates principal components as new data arrives.
    """
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.n_samples_seen_ = 0
    
    def partial_fit(self, X):
        """
        TODO: Update PCA with new batch of data
        
        Uses online algorithms to update mean and covariance.
        """
        # TODO: Implement incremental updates:
        # 1. Update running mean
        # 2. Update covariance matrix incrementally
        # 3. Recompute eigendecomposition
        
        pass
    
    def transform(self, X):
        """
        TODO: Transform data using current principal components
        """
        # TODO: Apply current PCA transformation
        
        pass

def robust_pca(X, max_iterations=100, tolerance=1e-6):
    """
    TODO: Implement robust PCA for outlier-contaminated data
    
    Regular PCA is sensitive to outliers. Robust PCA separates
    data into low-rank (signal) + sparse (outliers) components.
    
    Args:
        X: Data matrix (possibly with outliers)
        max_iterations: Maximum iterations for optimization
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary with low-rank and sparse components
    """
    # TODO: Implement robust PCA using iterative approach:
    # 1. Initialize low-rank and sparse components
    # 2. Alternate between updating each component
    # 3. Use soft thresholding for sparsity
    # 4. Use SVD for low-rank updates
    
    pass

def kernel_pca_preview(X, kernel='rbf', gamma=1.0, n_components=2):
    """
    TODO: Preview of kernel PCA for nonlinear dimensionality reduction
    
    Kernel PCA applies PCA in a higher-dimensional feature space
    defined by a kernel function, allowing nonlinear dimensionality reduction.
    
    Args:
        X: Data matrix
        kernel: Kernel type ('rbf', 'poly', 'sigmoid')
        gamma: Kernel parameter
        n_components: Number of components to extract
        
    Returns:
        Transformed data in kernel PCA space
    """
    # TODO: Implement basic kernel PCA:
    # 1. Compute kernel matrix K
    # 2. Center kernel matrix
    # 3. Find eigenvalues/eigenvectors of centered K
    # 4. Transform data using kernel eigenvectors
    
    pass

# ==============================================
# PART 6: COMPREHENSIVE PCA TOOLKIT
# ==============================================

class PCAToolkit:
    """
    TODO: Build comprehensive PCA analysis toolkit
    
    This class should provide all tools needed for PCA analysis:
    - Standard PCA with all options
    - Variance analysis and component selection
    - Visualization tools
    - Real-world applications
    """
    
    def __init__(self):
        self.pca_result_ = None
        self.original_data_ = None
    
    def fit(self, X, n_components=None):
        """
        TODO: Fit PCA to data
        """
        # TODO: Store data and compute PCA
        
        pass
    
    def transform(self, X):
        """
        TODO: Transform data to principal component space
        """
        # TODO: Apply learned PCA transformation
        
        pass
    
    def fit_transform(self, X, n_components=None):
        """
        TODO: Fit PCA and transform data in one step
        """
        # TODO: Combine fit and transform
        
        pass
    
    def inverse_transform(self, X_transformed, n_components=None):
        """
        TODO: Reconstruct original data from principal components
        """
        # TODO: Implement reconstruction
        
        pass
    
    def plot_variance_explained(self):
        """
        TODO: Plot explained variance vs. number of components
        """
        # TODO: Create informative variance plot
        
        pass
    
    def plot_components_2d(self):
        """
        TODO: Visualize first two principal components
        """
        # TODO: Create 2D visualization
        
        pass
    
    def analyze_component_loadings(self, feature_names=None):
        """
        TODO: Analyze which features contribute most to each component
        """
        # TODO: Create loadings analysis
        
        pass
    
    def generate_report(self):
        """
        TODO: Generate comprehensive PCA analysis report
        """
        # TODO: Create detailed analysis report
        
        pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_eigenvalue_concepts():
    """Demonstrate fundamental eigenvalue concepts."""
    
    print("🎯 Eigenvalue and Eigenvector Demonstration")
    print("=" * 50)
    
    # Create test matrix with known eigenstructure
    A = np.array([[3, 1], [0, 2]])  # Upper triangular - eigenvalues are diagonal
    
    print(f"Test matrix A:")
    print(A)
    print(f"True eigenvalues: 3, 2")
    
    try:
        # Test power iteration
        eigenval, eigenvec = power_iteration(A)
        print(f"Power iteration - Dominant eigenvalue: {eigenval:.3f}")
        print(f"Dominant eigenvector: [{eigenvec[0]:.3f}, {eigenvec[1]:.3f}]")
        
        # Test characteristic polynomial
        eigenvals_2x2 = characteristic_polynomial_2x2(A)
        print(f"Characteristic polynomial eigenvalues: {eigenvals_2x2}")
        
        # Test full eigendecomposition
        eigenvals, eigenvecs = full_eigendecomposition(A)
        print(f"Full eigendecomposition:")
        print(f"Eigenvalues: {eigenvals}")
        print(f"Eigenvectors:\n{eigenvecs}")
        
        # Visualize for 2D case
        visualize_eigenvectors_2d(A)
        print("✅ Eigenvalue visualization created")
        
    except Exception as e:
        print(f"❌ Eigenvalue demo failed: {e}")
        print("Implement the eigenvalue functions!")

def demonstrate_covariance_analysis():
    """Demonstrate covariance matrix analysis."""
    
    print("\n📊 Covariance Matrix Analysis")
    print("=" * 50)
    
    # Generate correlated 2D data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 1]]  # Positive correlation
    data = np.random.multivariate_normal(mean, cov, 200)
    
    print(f"Generated {len(data)} samples of 2D correlated data")
    
    try:
        # Test covariance computation
        computed_cov = compute_covariance_matrix(data)
        print(f"Computed covariance matrix:")
        print(computed_cov)
        print(f"True covariance matrix:")
        print(np.array(cov))
        
        # Test variance structure analysis
        variance_analysis = analyze_data_variance_structure(data)
        print(f"✅ Variance structure analysis completed")
        
        # Visualize covariance geometry
        demonstrate_covariance_geometry(data)
        print("✅ Covariance geometry visualization created")
        
    except Exception as e:
        print(f"❌ Covariance analysis failed: {e}")
        print("Implement the covariance analysis functions!")

def demonstrate_pca_algorithm():
    """Demonstrate PCA algorithm implementation."""
    
    print("\n🧮 Principal Component Analysis")
    print("=" * 50)
    
    # Load iris dataset for PCA demo
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    try:
        # Test PCA implementation
        pca_result = pca_from_scratch(X, n_components=2)
        print(f"✅ PCA completed")
        print(f"Principal components shape: {pca_result['components'].shape}")
        print(f"Transformed data shape: {pca_result['X_transformed'].shape}")
        
        # Test explained variance analysis
        variance_analysis = explained_variance_analysis(pca_result['eigenvalues'])
        print(f"Explained variance ratios: {variance_analysis['explained_variance_ratio']}")
        print(f"Cumulative explained variance: {variance_analysis['cumulative_variance']}")
        
        # Test reconstruction
        X_reconstructed = pca_reconstruction(
            pca_result['X_transformed'], 
            pca_result['components'],
            pca_result['mean'],
            n_components=2
        )
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        print(f"Reconstruction error (2 components): {reconstruction_error:.4f}")
        
        # Visualize PCA results
        visualize_pca_2d(X[:, :2], pca_result)  # Use first 2 features for viz
        print("✅ PCA visualization created")
        
    except Exception as e:
        print(f"❌ PCA demo failed: {e}")
        print("Implement the PCA functions!")

def demonstrate_real_world_applications():
    """Demonstrate real-world PCA applications."""
    
    print("\n🌍 Real-World PCA Applications")
    print("=" * 50)
    
    try:
        # Eigenfaces demonstration
        print("Loading face dataset for eigenfaces demo...")
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        face_images = faces.data.reshape(-1, 64, 64)[:20]  # Use subset for demo
        
        eigenfaces_result = eigenfaces_analysis(face_images)
        print("✅ Eigenfaces analysis completed")
        
        # Data compression demo
        iris = load_iris()
        X = iris.data
        
        compression_result = data_compression_pca(X)
        print("✅ Data compression analysis completed")
        
        # Feature extraction demo
        feature_result = feature_extraction_pca(X, iris.target, n_components=2)
        print("✅ Feature extraction analysis completed")
        
        # Anomaly detection demo
        anomaly_result = anomaly_detection_pca(X)
        print("✅ Anomaly detection analysis completed")
        
    except Exception as e:
        print(f"❌ Real-world applications demo failed: {e}")
        print("Implement the application functions!")

def comprehensive_pca_analysis():
    """Run comprehensive PCA analysis using toolkit."""
    
    print("\n🚀 Comprehensive PCA Analysis Toolkit")
    print("=" * 50)
    
    # Load dataset for comprehensive analysis
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    try:
        # Create PCA toolkit
        pca_toolkit = PCAToolkit()
        
        # Run complete analysis
        X_transformed = pca_toolkit.fit_transform(X)
        print("✅ PCA toolkit fitting completed")
        
        # Generate visualizations
        pca_toolkit.plot_variance_explained()
        pca_toolkit.plot_components_2d()
        print("✅ Visualizations created")
        
        # Analyze component loadings
        loadings_analysis = pca_toolkit.analyze_component_loadings(iris.feature_names)
        print("✅ Component loadings analysis completed")
        
        # Generate comprehensive report
        report = pca_toolkit.generate_report()
        print("✅ Comprehensive report generated")
        
        print("\n🎉 Congratulations! You've built a complete PCA analysis toolkit!")
        print("You now understand the mathematical foundation of dimensionality reduction.")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("Implement the PCAToolkit class!")

if __name__ == "__main__":
    """
    Run this file to explore eigenvalues and PCA!
    
    Complete the TODO functions above, then run:
    python week4_exercises.py
    """
    
    print("🧮 Welcome to Neural Odyssey Week 4: Eigenvalues and PCA!")
    print("Complete the TODO functions to master the mathematics of data structure.")
    print("\nTo get started:")
    print("1. Implement eigenvalue computation (power_iteration, characteristic_polynomial)")
    print("2. Build covariance matrix analysis tools")
    print("3. Create PCA algorithm from scratch")
    print("4. Apply to real-world problems (eigenfaces, compression, etc.)")
    print("5. Build the comprehensive PCAToolkit")
    
    # Uncomment these lines after implementing the functions:
    # demonstrate_eigenvalue_concepts()
    # demonstrate_covariance_analysis()
    # demonstrate_pca_algorithm()
    # demonstrate_real_world_applications()
    # comprehensive_pca_analysis()
    
    print("\n💡 Pro tip: Eigenvalues reveal the fundamental structure of data!")
    print("Master this and you'll understand how many ML algorithms work under the hood.")
    
    print("\n🎯 Success metrics:")
    print("• Can you explain eigenvectors as directions that don't rotate?")
    print("• Can you derive PCA from the variance maximization principle?")
    print("• Can you implement eigenfaces and understand why they work?")
    print("• Can you choose the right number of principal components?")
    
    print("\n🏆 Master this week and you'll see the hidden structure in any dataset!")
    print("PCA is not just dimensionality reduction - it's a way of thinking about data!")