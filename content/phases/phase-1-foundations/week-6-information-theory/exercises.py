"""
Neural Odyssey - Week 6: Information Theory and Entropy
Exercises for mastering the mathematical language of information

This module implements core concepts that quantify information and uncertainty:
- Entropy and information content as fundamental measures
- Mutual information and independence relationships
- Cross-entropy and KL divergence for comparing distributions
- Information-theoretic foundations of machine learning

Complete the TODO functions to build your information theory toolkit!
Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from sklearn.datasets import load_iris, make_classification
from sklearn.feature_selection import mutual_info_classif
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PART 1: INFORMATION AND ENTROPY FUNDAMENTALS
# ==============================================

def information_content(probability):
    """
    TODO: Calculate information content of an event
    
    Information content measures the "surprise" of an event.
    I(x) = -log₂(P(x))
    
    Higher probability events carry less information.
    This is the foundation of all information theory.
    
    Args:
        probability: Probability of the event (0 < p ≤ 1)
        
    Returns:
        Information content in bits
    """
    # TODO: Implement information content
    # Handle edge cases: probability = 0 (infinite information)
    # Use log base 2 for bits, natural log for nats
    
    pass

def shannon_entropy(probabilities):
    """
    TODO: Calculate Shannon entropy of a probability distribution
    
    H(X) = -Σ P(x) * log₂(P(x))
    
    Entropy measures the average information content.
    Maximum entropy = log₂(n) for uniform distribution over n outcomes.
    
    Args:
        probabilities: Array of probabilities (must sum to 1)
        
    Returns:
        Entropy in bits
    """
    # TODO: Implement Shannon entropy
    # Handle zero probabilities (0 * log(0) = 0 by convention)
    # Verify probabilities sum to 1
    
    pass

def entropy_from_counts(counts):
    """
    TODO: Calculate entropy from frequency counts
    
    Convenient wrapper for real data where we have counts instead of probabilities.
    
    Args:
        counts: Array of frequency counts
        
    Returns:
        Entropy in bits
    """
    # TODO: Convert counts to probabilities and calculate entropy
    # Handle empty or all-zero counts
    
    pass

def maximum_entropy_distribution(n):
    """
    TODO: Generate maximum entropy distribution over n outcomes
    
    The uniform distribution maximizes entropy for a given number of outcomes.
    This is a fundamental result in information theory.
    
    Args:
        n: Number of possible outcomes
        
    Returns:
        Uniform probability distribution and its entropy
    """
    # TODO: Create uniform distribution and calculate its entropy
    # Show that H_max = log₂(n)
    
    pass

def entropy_of_coin_flip(p):
    """
    TODO: Calculate entropy of biased coin flip
    
    Classic example showing how entropy varies with bias.
    Maximum entropy at p=0.5, minimum at p=0 or p=1.
    
    Args:
        p: Probability of heads
        
    Returns:
        Entropy of the coin flip
    """
    # TODO: Calculate entropy for Bernoulli distribution
    # Handle edge cases p=0 and p=1
    
    pass

def visualize_entropy_vs_probability():
    """
    TODO: Visualize how entropy changes with probability
    
    Shows the characteristic inverted-U shape of binary entropy.
    """
    # TODO: Plot entropy vs probability for coin flips
    # Show maximum at p=0.5, symmetry around this point
    
    pass

# ==============================================
# PART 2: JOINT AND CONDITIONAL ENTROPY
# ==============================================

def joint_entropy(joint_probabilities):
    """
    TODO: Calculate joint entropy H(X,Y)
    
    H(X,Y) = -Σ Σ P(x,y) * log₂(P(x,y))
    
    Measures uncertainty in joint distribution.
    
    Args:
        joint_probabilities: 2D array of joint probabilities P(X=i, Y=j)
        
    Returns:
        Joint entropy in bits
    """
    # TODO: Implement joint entropy calculation
    # Ensure joint_probabilities sums to 1
    # Handle zero probabilities
    
    pass

def conditional_entropy(joint_probabilities):
    """
    TODO: Calculate conditional entropy H(Y|X)
    
    H(Y|X) = H(X,Y) - H(X)
    
    Measures remaining uncertainty in Y after observing X.
    
    Args:
        joint_probabilities: 2D array of joint probabilities
        
    Returns:
        Conditional entropy H(Y|X) in bits
    """
    # TODO: Calculate H(Y|X) using the formula:
    # H(Y|X) = -Σ Σ P(x,y) * log₂(P(y|x))
    # Or equivalently: H(Y|X) = H(X,Y) - H(X)
    
    pass

def mutual_information(joint_probabilities):
    """
    TODO: Calculate mutual information I(X;Y)
    
    I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    
    Measures how much information X and Y share.
    I(X;Y) = 0 iff X and Y are independent.
    
    Args:
        joint_probabilities: 2D array of joint probabilities
        
    Returns:
        Mutual information in bits
    """
    # TODO: Calculate mutual information
    # Use any of the equivalent formulas
    # Verify result is non-negative
    
    pass

def independence_test_information_theory(joint_probs, tolerance=1e-6):
    """
    TODO: Test independence using mutual information
    
    Two variables are independent iff I(X;Y) = 0.
    
    Args:
        joint_probs: Joint probability distribution
        tolerance: Tolerance for considering MI = 0
        
    Returns:
        Boolean indicating independence
    """
    # TODO: Test if mutual information is approximately zero
    
    pass

def demonstrate_information_relationships():
    """
    TODO: Demonstrate fundamental information theory relationships
    
    Shows key relationships between different entropy measures.
    """
    # TODO: Create example distributions and verify:
    # 1. H(X,Y) ≤ H(X) + H(Y) (equality iff independent)
    # 2. H(X|Y) ≤ H(X) (conditioning reduces entropy)
    # 3. I(X;Y) ≥ 0 (mutual information is non-negative)
    # 4. I(X;X) = H(X) (self-information equals entropy)
    
    pass

# ==============================================
# PART 3: DIVERGENCES AND DISTANCE MEASURES
# ==============================================

def kl_divergence(p, q, epsilon=1e-10):
    """
    TODO: Calculate Kullback-Leibler divergence
    
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    
    Measures how different Q is from P. Not symmetric!
    KL(P||Q) ≠ KL(Q||P) in general.
    
    Args:
        p: "True" probability distribution
        q: "Approximate" probability distribution  
        epsilon: Small value to add for numerical stability
        
    Returns:
        KL divergence (always ≥ 0)
    """
    # TODO: Implement KL divergence
    # Add epsilon to avoid log(0)
    # Verify both distributions sum to 1
    # Handle cases where p[i] > 0 but q[i] = 0 (infinite divergence)
    
    pass

def js_divergence(p, q):
    """
    TODO: Calculate Jensen-Shannon divergence
    
    JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Symmetric and bounded version of KL divergence.
    
    Args:
        p, q: Probability distributions
        
    Returns:
        Jensen-Shannon divergence (0 ≤ JS ≤ log(2))
    """
    # TODO: Implement Jensen-Shannon divergence
    # Calculate average distribution M
    # Compute symmetric combination of KL divergences
    
    pass

def cross_entropy(p, q, epsilon=1e-10):
    """
    TODO: Calculate cross-entropy
    
    H(P,Q) = -Σ P(x) * log(Q(x))
    
    Cross-entropy is fundamental to machine learning loss functions.
    H(P,Q) = H(P) + KL(P||Q)
    
    Args:
        p: True distribution
        q: Predicted distribution
        epsilon: Numerical stability constant
        
    Returns:
        Cross-entropy
    """
    # TODO: Implement cross-entropy
    # This is the loss function used in classification!
    
    pass

def compare_divergence_measures(p, q):
    """
    TODO: Compare different ways to measure distribution differences
    
    Shows relationships between KL, JS, and cross-entropy.
    
    Args:
        p, q: Two probability distributions
        
    Returns:
        Dictionary with various divergence measures
    """
    # TODO: Calculate and compare:
    # 1. KL(P||Q) and KL(Q||P) - show asymmetry
    # 2. JS(P,Q) - symmetric version
    # 3. Cross-entropy H(P,Q)
    # 4. Relationship: H(P,Q) = H(P) + KL(P||Q)
    
    pass

def visualize_kl_divergence():
    """
    TODO: Visualize KL divergence between distributions
    
    Shows how KL divergence changes as distributions become more different.
    """
    # TODO: Create visualization showing:
    # 1. Two probability distributions
    # 2. KL divergence in both directions
    # 3. How divergence changes as distributions shift
    
    pass

# ==============================================
# PART 4: INFORMATION THEORY IN MACHINE LEARNING
# ==============================================

def mutual_information_feature_selection(X, y, k=5):
    """
    TODO: Use mutual information for feature selection
    
    Select features that have highest mutual information with target.
    This is a fundamental application in ML preprocessing.
    
    Args:
        X: Feature matrix
        y: Target labels
        k: Number of top features to select
        
    Returns:
        Selected feature indices and their MI scores
    """
    # TODO: Calculate mutual information between each feature and target
    # For continuous features, use binning or kernel density estimation
    # Return indices of top k features
    
    pass

def information_gain_decision_tree(X, y, feature_idx, threshold=None):
    """
    TODO: Calculate information gain for decision tree splitting
    
    Information Gain = H(parent) - Σ (|child_i| / |parent|) * H(child_i)
    
    This is how decision trees choose the best splits.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_idx: Index of feature to split on
        threshold: Threshold for continuous features (None for categorical)
        
    Returns:
        Information gain from this split
    """
    # TODO: Implement information gain calculation
    # Handle both categorical and continuous features
    # Calculate weighted average of child entropies
    
    pass

def cross_entropy_loss_classification(y_true, y_pred, epsilon=1e-15):
    """
    TODO: Implement cross-entropy loss for classification
    
    This is the standard loss function for classification problems.
    CE = -Σ y_true * log(y_pred)
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        epsilon: Small value for numerical stability
        
    Returns:
        Cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    # Handle both binary and multiclass cases
    # Add epsilon to prevent log(0)
    # Average over all samples
    
    pass

def entropy_regularization(model_predictions, alpha=0.01):
    """
    TODO: Implement entropy regularization
    
    Entropy regularization encourages prediction uncertainty,
    preventing overconfident predictions.
    
    Loss = Original_Loss - α * H(predictions)
    
    Args:
        model_predictions: Model probability predictions
        alpha: Regularization strength
        
    Returns:
        Entropy regularization term
    """
    # TODO: Calculate entropy of predictions
    # Higher entropy = more uncertain predictions
    # This is subtracted from loss to encourage uncertainty
    
    pass

def analyze_model_uncertainty(predictions, true_labels):
    """
    TODO: Analyze model uncertainty using information theory
    
    Use entropy and mutual information to understand model behavior.
    
    Args:
        predictions: Model probability predictions
        true_labels: True class labels
        
    Returns:
        Dictionary with uncertainty analysis
    """
    # TODO: Calculate:
    # 1. Average prediction entropy (model uncertainty)
    # 2. Mutual information between predictions and true labels
    # 3. Calibration analysis using information metrics
    
    pass

# ==============================================
# PART 5: ADVANCED INFORMATION THEORY
# ==============================================

def conditional_mutual_information(joint_probs_xyz):
    """
    TODO: Calculate conditional mutual information I(X;Y|Z)
    
    I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
    
    Measures information X and Y share given knowledge of Z.
    
    Args:
        joint_probs_xyz: 3D array of joint probabilities P(X,Y,Z)
        
    Returns:
        Conditional mutual information
    """
    # TODO: Implement conditional mutual information
    # This requires careful marginalization over the third variable
    
    pass

def data_processing_inequality_demo(n_samples=1000):
    """
    TODO: Demonstrate the data processing inequality
    
    If X → Y → Z forms a Markov chain, then I(X;Z) ≤ I(X;Y)
    
    Information cannot be created by processing data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Demonstration that processing reduces mutual information
    """
    # TODO: Create Markov chain X → Y → Z
    # Show that I(X;Z) ≤ I(X;Y)
    # This is fundamental to understanding information flow
    
    pass

def compression_and_entropy(data_string):
    """
    TODO: Demonstrate relationship between entropy and compression
    
    Shannon's source coding theorem: optimal compression length
    approaches entropy as message length increases.
    
    Args:
        data_string: String of symbols to analyze
        
    Returns:
        Entropy and various compression metrics
    """
    # TODO: Calculate:
    # 1. Empirical entropy of string
    # 2. Huffman coding length
    # 3. Compare with theoretical limits
    
    pass

def channel_capacity_demo(noise_level=0.1):
    """
    TODO: Demonstrate channel capacity concept
    
    Channel capacity = max I(X;Y) over all input distributions
    
    Shows fundamental limits of communication.
    
    Args:
        noise_level: Amount of noise in the channel
        
    Returns:
        Channel capacity and optimal input distribution
    """
    # TODO: Model noisy communication channel
    # Find input distribution that maximizes mutual information
    # This gives the channel capacity
    
    pass

def rate_distortion_theory_demo(data, distortion_levels):
    """
    TODO: Demonstrate rate-distortion trade-off
    
    Rate-distortion theory quantifies the trade-off between
    compression rate and reconstruction quality.
    
    Args:
        data: Original data to compress
        distortion_levels: Array of acceptable distortion levels
        
    Returns:
        Rate-distortion curve
    """
    # TODO: For each distortion level, find minimum rate
    # Show fundamental trade-off between compression and quality
    
    pass

# ==============================================
# PART 6: INFORMATION THEORY TOOLKIT
# ==============================================

class InformationTheoryAnalyzer:
    """
    TODO: Build comprehensive information theory analysis toolkit
    
    This class should provide all tools needed for information-theoretic
    analysis of data and models:
    - Entropy calculations and analysis
    - Mutual information and feature selection
    - Divergence measures and model evaluation
    - Advanced information theory concepts
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_dataset(self, X, y=None):
        """
        TODO: Comprehensive information-theoretic analysis of dataset
        
        Args:
            X: Feature matrix
            y: Target labels (optional)
            
        Returns:
            Dictionary with complete analysis
        """
        # TODO: Calculate:
        # 1. Feature entropies
        # 2. Target entropy (if provided)
        # 3. Mutual information between features and target
        # 4. Feature dependencies (mutual information matrix)
        
        pass
    
    def evaluate_model_information(self, predictions, true_labels):
        """
        TODO: Information-theoretic evaluation of model predictions
        
        Args:
            predictions: Model probability predictions
            true_labels: True class labels
            
        Returns:
            Comprehensive model evaluation
        """
        # TODO: Calculate:
        # 1. Cross-entropy loss
        # 2. Model uncertainty (prediction entropy)
        # 3. Information gain from predictions
        # 4. Calibration metrics
        
        pass
    
    def feature_selection_analysis(self, X, y, methods=['mutual_info', 'information_gain']):
        """
        TODO: Compare different information-based feature selection methods
        """
        # TODO: Implement multiple feature selection approaches
        # Compare their effectiveness and computational cost
        
        pass
    
    def visualize_information_landscape(self, data):
        """
        TODO: Create comprehensive visualizations of information structure
        """
        # TODO: Create plots showing:
        # 1. Entropy vs features
        # 2. Mutual information heatmap
        # 3. Information flow diagrams
        
        pass
    
    def generate_information_report(self):
        """
        TODO: Generate comprehensive information theory report
        """
        # TODO: Create detailed report with:
        # 1. Summary statistics
        # 2. Information-theoretic insights
        # 3. Recommendations for ML pipeline
        
        pass

# ==============================================
# DEMONSTRATION AND TESTING
# ==============================================

def demonstrate_information_fundamentals():
    """Demonstrate fundamental information theory concepts."""
    
    print("📊 Information Theory Fundamentals")
    print("=" * 50)
    
    try:
        # Information content examples
        events = [
            ("Coin flip heads", 0.5),
            ("Rolling a 6", 1/6),
            ("Rare event", 0.01),
            ("Certain event", 1.0)
        ]
        
        print("Information content of different events:")
        for event, prob in events:
            if prob > 0:
                info = information_content(prob)
                print(f"{event} (p={prob}): {info:.2f} bits")
        
        # Entropy examples
        distributions = [
            ("Fair coin", [0.5, 0.5]),
            ("Biased coin", [0.9, 0.1]),
            ("Fair die", [1/6] * 6),
            ("Deterministic", [1.0, 0.0])
        ]
        
        print("\nEntropy of different distributions:")
        for name, dist in distributions:
            h = shannon_entropy(dist)
            max_h = np.log2(len(dist))
            print(f"{name}: {h:.3f} bits (max: {max_h:.3f})")
        
        # Visualize entropy vs probability
        visualize_entropy_vs_probability()
        print("✅ Entropy visualization created")
        
    except Exception as e:
        print(f"❌ Information fundamentals demo failed: {e}")
        print("Implement the basic information theory functions!")

def demonstrate_mutual_information():
    """Demonstrate mutual information and dependencies."""
    
    print("\n🔗 Mutual Information and Dependencies")
    print("=" * 50)
    
    try:
        # Create example joint distributions
        # Independent variables
        p_x = [0.5, 0.5]
        p_y = [0.3, 0.7]
        independent_joint = np.outer(p_x, p_y)
        
        # Dependent variables  
        dependent_joint = np.array([[0.4, 0.1], [0.2, 0.3]])
        
        print("Independent variables:")
        mi_indep = mutual_information(independent_joint)
        print(f"Mutual information: {mi_indep:.6f} bits")
        
        print("\nDependent variables:")
        mi_dep = mutual_information(dependent_joint)
        print(f"Mutual information: {mi_dep:.3f} bits")
        
        # Test independence
        is_indep = independence_test_information_theory(independent_joint)
        print(f"Independence test (independent): {is_indep}")
        
        is_indep_dep = independence_test_information_theory(dependent_joint)
        print(f"Independence test (dependent): {is_indep_dep}")
        
        # Demonstrate information relationships
        demonstrate_information_relationships()
        print("✅ Information relationships verified")
        
    except Exception as e:
        print(f"❌ Mutual information demo failed: {e}")
        print("Implement the mutual information functions!")

def demonstrate_divergences():
    """Demonstrate divergence measures."""
    
    print("\n📏 Divergence Measures")
    print("=" * 50)
    
    try:
        # Example distributions
        p = np.array([0.5, 0.3, 0.2])
        q1 = np.array([0.4, 0.4, 0.2])  # Similar to p
        q2 = np.array([0.1, 0.1, 0.8])  # Very different from p
        
        print("Comparing distributions:")
        print(f"P: {p}")
        print(f"Q1 (similar): {q1}")
        print(f"Q2 (different): {q2}")
        
        # KL divergences
        kl_p_q1 = kl_divergence(p, q1)
        kl_p_q2 = kl_divergence(p, q2)
        
        print(f"\nKL(P||Q1): {kl_p_q1:.3f}")
        print(f"KL(P||Q2): {kl_p_q2:.3f}")
        
        # Cross-entropies
        ce_p_q1 = cross_entropy(p, q1)
        ce_p_q2 = cross_entropy(p, q2)
        
        print(f"\nH(P,Q1): {ce_p_q1:.3f}")
        print(f"H(P,Q2): {ce_p_q2:.3f}")
        
        # Compare all measures
        comparison = compare_divergence_measures(p, q1)
        print("✅ Divergence comparison completed")
        
        # Visualize KL divergence
        visualize_kl_divergence()
        print("✅ KL divergence visualization created")
        
    except Exception as e:
        print(f"❌ Divergences demo failed: {e}")
        print("Implement the divergence measure functions!")

def demonstrate_ml_applications():
    """Demonstrate information theory in ML applications."""
    
    print("\n🤖 Machine Learning Applications")
    print("=" * 50)
    
    try:
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Feature selection using mutual information
        selected_features, mi_scores = mutual_information_feature_selection(X, y, k=2)
        print(f"Top 2 features by mutual information: {selected_features}")
        print(f"MI scores: {mi_scores}")
        
        # Information gain for decision trees
        for feature_idx in range(X.shape[1]):
            gain = information_gain_decision_tree(X, y, feature_idx)
            print(f"Information gain for feature {feature_idx}: {gain:.3f}")
        
        # Cross-entropy loss example
        n_classes = len(np.unique(y))
        # Create dummy predictions
        y_pred = np.random.dirichlet(np.ones(n_classes), size=len(y))
        y_true_onehot = np.eye(n_classes)[y]
        
        ce_loss = cross_entropy_loss_classification(y_true_onehot, y_pred)
        print(f"Cross-entropy loss: {ce_loss:.3f}")
        
        # Model uncertainty analysis
        uncertainty_analysis = analyze_model_uncertainty(y_pred, y)
        print("✅ Model uncertainty analysis completed")
        
    except Exception as e:
        print(f"❌ ML applications demo failed: {e}")
        print("Implement the ML application functions!")

def comprehensive_information_analysis():
    """Run comprehensive information theory analysis."""
    
    print("\n🚀 Comprehensive Information Analysis")
    print("=" * 50)
    
    try:
        # Create information theory analyzer
        analyzer = InformationTheoryAnalyzer()
        
        # Load dataset for analysis
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                 n_redundant=2, random_state=42)
        
        # Run comprehensive analysis
        dataset_analysis = analyzer.analyze_dataset(X, y)
        print("✅ Dataset information analysis completed")
        
        # Evaluate model information
        n_classes = len(np.unique(y))
        y_pred = np.random.dirichlet(np.ones(n_classes), size=len(y))
        model_analysis = analyzer.evaluate_model_information(y_pred, y)
        print("✅ Model information evaluation completed")
        
        # Feature selection analysis
        fs_analysis = analyzer.feature_selection_analysis(X, y)
        print("✅ Feature selection analysis completed")
        
        # Generate comprehensive report
        report = analyzer.generate_information_report()
        print("✅ Information theory report generated")
        
        print("\n🎉 Congratulations! You've built a complete information theory toolkit!")
        print("You now understand the mathematical language of information and uncertainty.")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("Implement the InformationTheoryAnalyzer class!")

if __name__ == "__main__":
    """
    Run this file to explore information theory and entropy!
    
    Complete the TODO functions above, then run:
    python week6_exercises.py
    """
    
    print("📊 Welcome to Neural Odyssey Week 6: Information Theory and Entropy!")
    print("Complete the TODO functions to master the language of information.")
    print("\nTo get started:")
    print("1. Implement basic information measures (entropy, information content)")
    print("2. Build mutual information and conditional entropy functions")
    print("3. Create divergence measures (KL, JS, cross-entropy)")
    print("4. Apply to ML problems (feature selection, loss functions)")
    print("5. Build comprehensive information theory analyzer")
    
    # Uncomment these lines after implementing the functions:
    # demonstrate_information_fundamentals()
    # demonstrate_mutual_information()
    # demonstrate_divergences()
    # demonstrate_ml_applications()
    # comprehensive_information_analysis()
    
    print("\n💡 Pro tip: Information theory is the foundation of modern ML!")
    print("Understanding entropy and mutual information will make you a better data scientist.")
    
    print("\n🎯 Success metrics:")
    print("• Can you explain why cross-entropy is used as a loss function?")
    print("• Can you use mutual information for feature selection?")
    print("• Can you interpret model uncertainty using entropy?")
    print("• Can you apply information theory to real ML problems?")
    
    print("\n🏆 Master this week and you'll understand the deep connections between")
    print("information, uncertainty, and learning that power all of AI!")