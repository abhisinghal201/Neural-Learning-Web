"""
Neural Odyssey - Week 15: Tree-Based Methods and Ensemble Learning
Phase 2: Core Machine Learning (Week 3)

The Wisdom of Crowds in Machine Learning

Building on linear models from Week 14, this week explores the power of tree-based methods
and ensemble learning. You'll journey from single decision trees to powerful ensembles
like Random Forests and Gradient Boosting, understanding how multiple weak learners
can combine to create strong predictive models.

This comprehensive exploration covers:
1. Decision trees from information theory foundations
2. Tree construction algorithms and splitting criteria
3. Pruning techniques and overfitting prevention
4. Bootstrap aggregating (Bagging) and Random Forests
5. Boosting algorithms: AdaBoost and Gradient Boosting
6. Modern implementations: XGBoost, LightGBM, CatBoost
7. Feature importance and model interpretation
8. Ensemble theory and bias-variance analysis
9. Real-world applications and deployment strategies

To get started, run: python exercises.py

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)


# ==========================================
# DECISION TREE FROM INFORMATION THEORY
# ==========================================

class DecisionTreeFromScratch:
    """
    Decision tree implementation from information theory first principles
    Demonstrates entropy, information gain, and tree construction
    """
    
    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.prediction = None
            self.is_leaf = False
            self.samples = 0
            self.entropy = 0
            self.info_gain = 0
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.class_names = None
        
    def entropy(self, y):
        """Calculate entropy: H(S) = -∑ p_i log2(p_i)"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Avoid log(0) by adding small epsilon
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def gini_impurity(self, y):
        """Calculate Gini impurity: Gini(S) = 1 - ∑ p_i²"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def calculate_impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.criterion == 'entropy':
            return self.entropy(y)
        elif self.criterion == 'gini':
            return self.gini_impurity(y)
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")
    
    def information_gain(self, X, y, feature, threshold):
        """Calculate information gain for a split"""
        # Parent impurity
        parent_impurity = self.calculate_impurity(y)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Check if split is valid
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return 0
        
        # Weighted impurity of children
        n = len(y)
        left_impurity = self.calculate_impurity(y[left_mask])
        right_impurity = self.calculate_impurity(y[right_mask])
        
        weighted_impurity = (np.sum(left_mask) / n) * left_impurity + (np.sum(right_mask) / n) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def best_split(self, X, y):
        """Find best feature and threshold for splitting"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # Get unique values as potential thresholds
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build decision tree"""
        node = self.Node()
        node.samples = len(y)
        node.entropy = self.calculate_impurity(y)
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1 or
            node.entropy == 0):
            
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Find best split
        feature, threshold, gain = self.best_split(X, y)
        
        if feature is None or gain <= 0:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        node.feature = feature
        node.threshold = threshold
        node.info_gain = gain
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y, feature_names=None, class_names=None):
        """Train decision tree"""
        print(f"🌳 Building Decision Tree from Information Theory")
        print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Criterion: {self.criterion}")
        print(f"   Initial impurity: {self.calculate_impurity(y):.4f}")
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.class_names = class_names or [f'Class_{i}' for i in np.unique(y)]
        
        self.root = self.build_tree(X, y)
        
        # Calculate tree statistics
        depth = self.tree_depth(self.root)
        leaves = self.count_leaves(self.root)
        
        print(f"   Tree depth: {depth}")
        print(f"   Number of leaves: {leaves}")
        print(f"   ✅ Tree construction complete")
        
        return self
    
    def predict_sample(self, x, node):
        """Predict single sample"""
        if node.is_leaf:
            return node.prediction
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions"""
        return np.array([self.predict_sample(x, self.root) for x in X])
    
    def tree_depth(self, node):
        """Calculate tree depth"""
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self.tree_depth(node.left), self.tree_depth(node.right))
    
    def count_leaves(self, node):
        """Count number of leaves"""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """Print tree structure in human-readable format"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            class_name = self.class_names[node.prediction] if self.class_names else f"Class {node.prediction}"
            print("  " * depth + f"{prefix}Predict {class_name} (samples: {node.samples}, impurity: {node.entropy:.3f})")
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names else f"Feature {node.feature}"
            print("  " * depth + f"{prefix}{feature_name} <= {node.threshold:.3f} (gain: {node.info_gain:.3f})")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")
    
    def visualize_tree_performance(self, X, y, X_test=None, y_test=None):
        """Comprehensive tree performance visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Decision boundary (for 2D data)
        if X.shape[1] == 2:
            ax = axes[0, 0]
            self._plot_decision_boundary(X, y, ax)
            ax.set_title('Decision Boundary')
        else:
            ax = axes[0, 0]
            ax.text(0.5, 0.5, 'Decision boundary\nvisualization only\navailable for 2D data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Decision Boundary (N/A for >2D)')
        
        # 2. Feature importance
        ax = axes[0, 1]
        importance = self.feature_importance()
        feature_names = self.feature_names or [f'F{i}' for i in range(len(importance))]
        
        bars = ax.bar(range(len(importance)), importance, alpha=0.7)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels(feature_names, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{imp:.3f}', ha='center', va='bottom')
        
        # 3. Tree complexity analysis
        ax = axes[1, 0]
        depths = range(1, min(15, self.max_depth + 1))
        train_scores = []
        val_scores = []
        
        if X_test is not None and y_test is not None:
            for d in depths:
                temp_tree = DecisionTreeFromScratch(max_depth=d, 
                                                  min_samples_split=self.min_samples_split,
                                                  criterion=self.criterion)
                temp_tree.fit(X, y, self.feature_names, self.class_names)
                
                train_pred = temp_tree.predict(X)
                test_pred = temp_tree.predict(X_test)
                
                train_scores.append(accuracy_score(y, train_pred))
                val_scores.append(accuracy_score(y_test, test_pred))
            
            ax.plot(depths, train_scores, 'b-o', label='Training Accuracy', linewidth=2)
            ax.plot(depths, val_scores, 'r-o', label='Validation Accuracy', linewidth=2)
            ax.set_xlabel('Tree Depth')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Complexity vs Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Complexity analysis\nrequires test data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Complexity Analysis (N/A)')
        
        # 4. Confusion matrix
        ax = axes[1, 1]
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Print tree structure
        print(f"\n🌳 Tree Structure:")
        print("=" * 50)
        self.print_tree()
    
    def _plot_decision_boundary(self, X, y, ax):
        """Plot decision boundary for 2D data"""
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, class_val in enumerate(np.unique(y)):
            mask = y == class_val
            class_name = self.class_names[i] if self.class_names else f'Class {class_val}'
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i % len(colors)], 
                      label=class_name, alpha=0.7, s=50)
        
        ax.set_xlabel(self.feature_names[0] if self.feature_names else 'Feature 0')
        ax.set_ylabel(self.feature_names[1] if self.feature_names else 'Feature 1')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def feature_importance(self):
        """Calculate feature importance based on information gain"""
        importance = np.zeros(len(self.feature_names))
        
        def traverse_tree(node, total_samples):
            if node is None or node.is_leaf:
                return
            
            # Weight by number of samples reaching this node
            weight = node.samples / total_samples
            importance[node.feature] += weight * node.info_gain
            
            traverse_tree(node.left, total_samples)
            traverse_tree(node.right, total_samples)
        
        traverse_tree(self.root, self.root.samples)
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


# ==========================================
# RANDOM FOREST IMPLEMENTATION
# ==========================================

class RandomForestFromScratch:
    """
    Random Forest implementation combining bootstrap sampling and random features
    Demonstrates ensemble learning and variance reduction
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees = []
        self.feature_subsets = []
        self.oob_indices = []
        self.feature_names = None
        self.class_names = None
        
    def _get_max_features(self, n_features):
        """Determine number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y, random_state):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Out-of-bag indices
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        
        return X[indices], y[indices], oob_indices
    
    def fit(self, X, y, feature_names=None, class_names=None):
        """Train Random Forest"""
        print(f"🌲 Building Random Forest Ensemble")
        print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Ensemble size: {self.n_estimators} trees")
        print(f"   Bootstrap sampling: {self.bootstrap}")
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.class_names = class_names or [f'Class_{i}' for i in np.unique(y)]
        
        n_features = X.shape[1]
        max_features_per_tree = self._get_max_features(n_features)
        
        print(f"   Features per tree: {max_features_per_tree}/{n_features}")
        
        self.trees = []
        self.feature_subsets = []
        self.oob_indices = []
        
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"   Training tree {i + 1}/{self.n_estimators}")
            
            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample, oob_idx = self._bootstrap_sample(X, y, i)
                self.oob_indices.append(oob_idx)
            else:
                X_sample, y_sample = X, y
                self.oob_indices.append([])
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, size=max_features_per_tree, replace=False)
            self.feature_subsets.append(feature_indices)
            
            # Train tree on selected features
            X_subset = X_sample[:, feature_indices]
            
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion='entropy'
            )
            
            # Create feature names for subset
            subset_feature_names = [self.feature_names[j] for j in feature_indices]
            tree.fit(X_subset, y_sample, subset_feature_names, self.class_names)
            
            self.trees.append(tree)
        
        print(f"   ✅ Random Forest training complete")
        
        # Calculate OOB error if bootstrap was used
        if self.bootstrap:
            oob_error = self._calculate_oob_error(X, y)
            print(f"   Out-of-bag error: {oob_error:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using majority voting"""
        if not self.trees:
            raise ValueError("Forest not trained yet!")
        
        n_samples = X.shape[0]
        n_classes = len(self.class_names)
        
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_subsets)):
            X_subset = X[:, feature_indices]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Majority voting
        predictions = []
        for i in range(n_samples):
            votes = all_predictions[i, :]
            prediction = Counter(votes.astype(int)).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.trees:
            raise ValueError("Forest not trained yet!")
        
        n_samples = X.shape[0]
        n_classes = len(self.class_names)
        
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_subsets)):
            X_subset = X[:, feature_indices]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Calculate probabilities
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            votes = all_predictions[i, :]
            for class_idx in range(n_classes):
                probas[i, class_idx] = np.sum(votes == class_idx) / self.n_estimators
        
        return probas
    
    def _calculate_oob_error(self, X, y):
        """Calculate out-of-bag error estimate"""
        n_samples = X.shape[0]
        oob_predictions = np.full(n_samples, -1)
        oob_counts = np.zeros(n_samples)
        
        for i, (tree, feature_indices, oob_idx) in enumerate(zip(self.trees, self.feature_subsets, self.oob_indices)):
            if len(oob_idx) == 0:
                continue
            
            X_oob = X[oob_idx][:, feature_indices]
            predictions = tree.predict(X_oob)
            
            for j, sample_idx in enumerate(oob_idx):
                if oob_predictions[sample_idx] == -1:
                    oob_predictions[sample_idx] = predictions[j]
                oob_counts[sample_idx] += 1
        
        # Calculate error for samples that have OOB predictions
        valid_mask = oob_predictions != -1
        if np.sum(valid_mask) == 0:
            return 1.0
        
        oob_error = np.mean(oob_predictions[valid_mask] != y[valid_mask])
        return oob_error
    
    def feature_importance(self):
        """Calculate feature importance by averaging over all trees"""
        importance = np.zeros(len(self.feature_names))
        
        for tree, feature_indices in zip(self.trees, self.feature_subsets):
            tree_importance = tree.feature_importance()
            for j, feat_idx in enumerate(feature_indices):
                importance[feat_idx] += tree_importance[j]
        
        # Average and normalize
        importance = importance / self.n_estimators
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def analyze_ensemble_diversity(self, X, y):
        """Analyze diversity among ensemble members"""
        print(f"\n📊 Ensemble Diversity Analysis")
        print("=" * 40)
        
        # Get predictions from all trees
        n_samples = X.shape[0]
        all_predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_subsets)):
            X_subset = X[:, feature_indices]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Calculate individual tree accuracies
        individual_accuracies = []
        for i in range(self.n_estimators):
            acc = accuracy_score(y, all_predictions[:, i])
            individual_accuracies.append(acc)
        
        # Calculate pairwise agreement
        pairwise_agreements = []
        for i in range(self.n_estimators):
            for j in range(i + 1, self.n_estimators):
                agreement = np.mean(all_predictions[:, i] == all_predictions[:, j])
                pairwise_agreements.append(agreement)
        
        # Ensemble accuracy
        ensemble_pred = self.predict(X)
        ensemble_accuracy = accuracy_score(y, ensemble_pred)
        
        print(f"   Individual tree accuracy: {np.mean(individual_accuracies):.4f} ± {np.std(individual_accuracies):.4f}")
        print(f"   Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"   Improvement: {ensemble_accuracy - np.mean(individual_accuracies):.4f}")
        print(f"   Average pairwise agreement: {np.mean(pairwise_agreements):.4f}")
        print(f"   Diversity (1 - agreement): {1 - np.mean(pairwise_agreements):.4f}")
        
        return {
            'individual_accuracies': individual_accuracies,
            'ensemble_accuracy': ensemble_accuracy,
            'pairwise_agreements': pairwise_agreements,
            'diversity': 1 - np.mean(pairwise_agreements)
        }


# ==========================================
# GRADIENT BOOSTING IMPLEMENTATION
# ==========================================

class GradientBoostingFromScratch:
    """
    Gradient Boosting implementation for classification
    Demonstrates sequential learning and error correction
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.init_prediction = None
        self.feature_names = None
        self.class_names = None
        
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)), 
                       np.exp(z) / (1 + np.exp(z)))
    
    def _log_odds(self, p):
        """Convert probability to log-odds"""
        p = np.clip(p, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return np.log(p / (1 - p))
    
    def fit(self, X, y, feature_names=None, class_names=None):
        """Train gradient boosting model"""
        print(f"🚀 Building Gradient Boosting Ensemble")
        print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Ensemble size: {self.n_estimators} trees")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Tree depth: {self.max_depth}")
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        self.class_names = class_names or ['Class_0', 'Class_1']
        
        # Convert to binary classification (0/1)
        y_binary = (y == 1).astype(int)
        
        # Initialize with log-odds of class proportion
        p_init = np.mean(y_binary)
        self.init_prediction = self._log_odds(p_init)
        
        print(f"   Initial prediction (log-odds): {self.init_prediction:.4f}")
        
        # Current predictions (log-odds)
        f_pred = np.full(len(y_binary), self.init_prediction)
        
        self.trees = []
        train_errors = []
        
        for m in range(self.n_estimators):
            if (m + 1) % 20 == 0:
                print(f"   Training tree {m + 1}/{self.n_estimators}")
            
            # Convert to probabilities
            p_pred = self._sigmoid(f_pred)
            
            # Calculate residuals (negative gradients)
            residuals = y_binary - p_pred
            
            # Subsample for stochastic gradient boosting
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * len(X))
                indices = np.random.choice(len(X), size=n_subsample, replace=False)
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals
            
            # Train tree to predict residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(X_sub, residuals_sub)
            
            # Make predictions and update
            tree_pred = tree.predict(X)
            f_pred += self.learning_rate * tree_pred
            
            self.trees.append(tree)
            
            # Calculate training error
            current_prob = self._sigmoid(f_pred)
            current_pred = (current_prob >= 0.5).astype(int)
            train_error = 1 - accuracy_score(y_binary, current_pred)
            train_errors.append(train_error)
            
            # Early stopping check
            if train_error < 1e-4:
                print(f"   ✅ Early stopping at iteration {m + 1}")
                break
        
        print(f"   ✅ Gradient boosting training complete")
        print(f"   Final training error: {train_errors[-1]:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        f_pred = self.decision_function(X)
        probabilities = self._sigmoid(f_pred)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        f_pred = self.decision_function(X)
        prob_class_1 = self._sigmoid(f_pred)
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])
    
    def decision_function(self, X):
        """Calculate decision function values (log-odds)"""
        f_pred = np.full(X.shape[0], self.init_prediction)
        
        for tree in self.trees:
            f_pred += self.learning_rate * tree.predict(X)
        
        return f_pred
    
    def staged_predict(self, X):
        """Predict at each stage of boosting"""
        f_pred = np.full(X.shape[0], self.init_prediction)
        
        for tree in self.trees:
            f_pred += self.learning_rate * tree.predict(X)
            probabilities = self._sigmoid(f_pred)
            yield (probabilities >= 0.5).astype(int)
    
    def feature_importance(self):
        """Calculate feature importance"""
        importance = np.zeros(len(self.feature_names))
        
        for tree in self.trees:
            tree_importance = tree.feature_importances_
            importance += tree_importance
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


# ==========================================
# ENSEMBLE COMPARISON FRAMEWORK
# ==========================================

class EnsembleComparisonFramework:
    """
    Framework for comparing different ensemble methods
    """
    
    def __init__(self):
        self.results = {}
        
    def compare_ensemble_methods(self, X, y, test_size=0.3, cv_folds=5):
        """Compare multiple ensemble methods"""
        print(f"\n🔍 Comprehensive Ensemble Methods Comparison")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Cross-validation folds: {cv_folds}")
        
        # Define models to compare
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                                           learning_rate=0.1, random_state=42)
        }
        
        # Add our custom implementations
        custom_models = {
            'Custom Tree': DecisionTreeFromScratch(max_depth=10),
            'Custom Random Forest': RandomForestFromScratch(n_estimators=50, max_depth=10, random_state=42),
            'Custom Gradient Boosting': GradientBoostingFromScratch(n_estimators=50, max_depth=3, 
                                                                   learning_rate=0.1, random_state=42)
        }
        
        results = {}
        
        # Evaluate sklearn models
        print(f"\n📊 Evaluating Scikit-learn Models:")
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Train on full training set and test
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': train_acc - test_acc
            }
        
        # Evaluate custom models
        print(f"\n🛠️  Evaluating Custom Implementations:")
        for name, model in custom_models.items():
            print(f"   Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                
                results[name] = {
                    'cv_mean': test_acc,  # Approximate with test accuracy
                    'cv_std': 0.0,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'overfitting': train_acc - test_acc
                }
            except Exception as e:
                print(f"   ⚠️  Error training {name}: {e}")
                continue
        
        self.results = results
        
        # Display results
        self._display_comparison_results()
        
        # Visualize comparison
        self._visualize_comparison()
        
        return results
    
    def _display_comparison_results(self):
        """Display comparison results in tabular format"""
        print(f"\n📋 Model Performance Comparison:")
        print("=" * 80)
        print(f"{'Model':<25} {'CV Score':<15} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
        print("-" * 80)
        
        for name, metrics in self.results.items():
            cv_score = f"{metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}"
            train_acc = f"{metrics['train_accuracy']:.3f}"
            test_acc = f"{metrics['test_accuracy']:.3f}"
            overfitting = f"{metrics['overfitting']:.3f}"
            
            print(f"{name:<25} {cv_score:<15} {train_acc:<12} {test_acc:<12} {overfitting:<12}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model} (Test Accuracy: {self.results[best_model]['test_accuracy']:.3f})")
    
    def _visualize_comparison(self):
        """Visualize model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(self.results.keys())
        
        # 1. Test accuracy comparison
        ax = axes[0, 0]
        test_accs = [self.results[m]['test_accuracy'] for m in models]
        bars = ax.bar(range(len(models)), test_accs, alpha=0.7, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Test Accuracy Comparison')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, acc in zip(bars, test_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Overfitting analysis
        ax = axes[0, 1]
        overfitting = [self.results[m]['overfitting'] for m in models]
        colors = ['red' if o > 0.05 else 'green' for o in overfitting]
        bars = ax.bar(range(len(models)), overfitting, alpha=0.7, color=colors)
        ax.set_xlabel('Models')
        ax.set_ylabel('Overfitting (Train - Test)')
        ax.set_title('Overfitting Analysis')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # 3. Train vs Test accuracy
        ax = axes[1, 0]
        train_accs = [self.results[m]['train_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, train_accs, width, label='Train', alpha=0.7, color='lightcoral')
        ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.7, color='lightblue')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Train vs Test Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cross-validation scores (where available)
        ax = axes[1, 1]
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        bars = ax.bar(range(len(models)), cv_means, alpha=0.7, color='lightgreen',
                     yerr=cv_stds, capsize=5)
        ax.set_xlabel('Models')
        ax.set_ylabel('CV Accuracy')
        ax.set_title('Cross-Validation Performance')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_bias_variance_tradeoff(self, X, y, n_experiments=20):
        """Analyze bias-variance tradeoff for ensemble methods"""
        print(f"\n⚖️  Bias-Variance Tradeoff Analysis")
        print("=" * 50)
        
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=None),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=None),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=None)
        }
        
        results = {}
        
        for name, model_class in models.items():
            print(f"   Analyzing {name}...")
            
            predictions = []
            
            # Run multiple experiments with different train/test splits
            for exp in range(n_experiments):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=exp
                )
                
                # Create new model instance for each experiment
                if name == 'Decision Tree':
                    model = DecisionTreeClassifier(max_depth=10, random_state=exp)
                elif name == 'Random Forest':
                    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=exp)
                else:  # Gradient Boosting
                    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=exp)
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions.append(pred)
            
            # Calculate bias and variance
            predictions = np.array(predictions)
            
            # Main prediction (average)
            main_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            
            # Bias (squared): how far is main prediction from true labels
            # Note: This is a simplified calculation for demonstration
            bias_squared = np.mean((main_pred - y_test) ** 2)
            
            # Variance: how much do individual predictions vary
            variance = np.mean(np.var(predictions, axis=0))
            
            # Accuracy
            accuracy = np.mean([accuracy_score(y_test, pred) for pred in predictions])
            
            results[name] = {
                'bias_squared': bias_squared,
                'variance': variance,
                'accuracy': accuracy
            }
            
            print(f"      Bias²: {bias_squared:.4f}")
            print(f"      Variance: {variance:.4f}")
            print(f"      Accuracy: {accuracy:.4f}")
        
        # Visualize bias-variance tradeoff
        self._visualize_bias_variance(results)
        
        return results
    
    def _visualize_bias_variance(self, results):
        """Visualize bias-variance tradeoff"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = list(results.keys())
        bias_squared = [results[m]['bias_squared'] for m in models]
        variance = [results[m]['variance'] for m in models]
        accuracy = [results[m]['accuracy'] for m in models]
        
        # Bias vs Variance
        ax1.scatter(bias_squared, variance, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (bias_squared[i], variance[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Bias²')
        ax1.set_ylabel('Variance')
        ax1.set_title('Bias-Variance Tradeoff')
        ax1.grid(True, alpha=0.3)
        
        # Decomposition bar chart
        x = np.arange(len(models))
        width = 0.35
        
        ax2.bar(x - width/2, bias_squared, width, label='Bias²', alpha=0.7)
        ax2.bar(x + width/2, variance, width, label='Variance', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Error Components')
        ax2.set_title('Bias² vs Variance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ==========================================
# MODERN BOOSTING FRAMEWORKS DEMO
# ==========================================

def modern_boosting_frameworks_demo(X, y):
    """
    Demonstrate modern boosting frameworks
    """
    print(f"\n🚀 Modern Boosting Frameworks Demonstration")
    print("=" * 60)
    
    try:
        import xgboost as xgb
        import lightgbm as lgb
        xgboost_available = True
        lightgbm_available = True
    except ImportError:
        print("   ⚠️  XGBoost or LightGBM not installed")
        print("   Install with: pip install xgboost lightgbm")
        xgboost_available = False
        lightgbm_available = False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {}
    
    # Scikit-learn Gradient Boosting
    print(f"\n📊 Training Scikit-learn Gradient Boosting...")
    gb_sklearn = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                          max_depth=3, random_state=42)
    gb_sklearn.fit(X_train, y_train)
    
    sklearn_pred = gb_sklearn.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    
    models['Sklearn GB'] = {
        'model': gb_sklearn,
        'accuracy': sklearn_acc,
        'predictions': sklearn_pred
    }
    
    print(f"   Accuracy: {sklearn_acc:.4f}")
    
    # XGBoost
    if xgboost_available:
        print(f"\n🔥 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, 
                                     max_depth=3, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        models['XGBoost'] = {
            'model': xgb_model,
            'accuracy': xgb_acc,
            'predictions': xgb_pred
        }
        
        print(f"   Accuracy: {xgb_acc:.4f}")
    
    # LightGBM
    if lightgbm_available:
        print(f"\n💡 Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, 
                                      max_depth=3, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        
        models['LightGBM'] = {
            'model': lgb_model,
            'accuracy': lgb_acc,
            'predictions': lgb_pred
        }
        
        print(f"   Accuracy: {lgb_acc:.4f}")
    
    # Compare performance
    print(f"\n📋 Performance Comparison:")
    print("-" * 40)
    for name, result in models.items():
        print(f"   {name:<15}: {result['accuracy']:.4f}")
    
    # Feature importance comparison
    if len(models) > 1:
        _compare_feature_importance(models, X_train.shape[1])
    
    return models

def _compare_feature_importance(models, n_features):
    """Compare feature importance across different models"""
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    for i, (name, result) in enumerate(models.items()):
        model = result['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = np.zeros(n_features)
        
        axes[i].bar(range(len(importance)), importance, alpha=0.7)
        axes[i].set_title(f'{name} Feature Importance')
        axes[i].set_xlabel('Features')
        axes[i].set_ylabel('Importance')
        axes[i].set_xticks(range(len(importance)))
        axes[i].set_xticklabels([f'F{j}' for j in range(len(importance))], rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# COMPREHENSIVE TREE METHODS DEMO
# ==========================================

def comprehensive_tree_methods_demo():
    """
    Complete demonstration of tree-based methods
    """
    print("🎓 Neural Odyssey - Week 15: Tree-Based Methods Deep Dive")
    print("=" * 70)
    print("The Wisdom of Crowds in Machine Learning")
    print("=" * 70)
    
    # ================================================================
    # DATASET PREPARATION
    # ================================================================
    
    print("\n📊 Preparing Datasets for Tree-Based Analysis")
    
    # 1. Synthetic classification dataset
    X_synth, y_synth = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
        n_clusters_per_class=1, random_state=42
    )
    print(f"   Synthetic dataset: {X_synth.shape[0]} samples, {X_synth.shape[1]} features")
    
    # 2. Iris dataset (for visualization)
    iris = load_iris()
    X_iris, y_iris = iris.data[:, :2], iris.target  # Use only 2 features for visualization
    print(f"   Iris dataset (2D): {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
    
    # 3. Wine dataset (for multi-class)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    print(f"   Wine dataset: {X_wine.shape[0]} samples, {X_wine.shape[1]} features")
    
    # 4. Breast cancer dataset (for binary classification)
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    print(f"   Breast cancer dataset: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")
    
    # ================================================================
    # DECISION TREE FUNDAMENTALS
    # ================================================================
    
    print("\n" + "="*70)
    print("🌳 DECISION TREE FUNDAMENTALS")
    print("="*70)
    
    print("\n1️⃣  Information Theory-Based Tree Construction")
    
    # Train custom decision tree on Iris data
    tree_custom = DecisionTreeFromScratch(max_depth=5, criterion='entropy')
    tree_custom.fit(X_iris, y_iris, 
                   feature_names=['Sepal Length', 'Sepal Width'],
                   class_names=['Setosa', 'Versicolor', 'Virginica'])
    
    # Split data for visualization
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )
    
    # Visualize tree performance
    tree_custom.visualize_tree_performance(X_iris_train, y_iris_train, X_iris_test, y_iris_test)
    
    # ================================================================
    # RANDOM FOREST ENSEMBLE
    # ================================================================
    
    print("\n" + "="*70)
    print("🌲 RANDOM FOREST ENSEMBLE LEARNING")
    print("="*70)
    
    print("\n1️⃣  Bootstrap Aggregating with Random Features")
    
    # Train custom Random Forest
    rf_custom = RandomForestFromScratch(
        n_estimators=50, max_depth=10, max_features='sqrt', 
        bootstrap=True, random_state=42
    )
    
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42
    )
    
    rf_custom.fit(X_cancer_train, y_cancer_train, 
                 feature_names=cancer.feature_names,
                 class_names=['Malignant', 'Benign'])
    
    # Analyze ensemble diversity
    diversity_results = rf_custom.analyze_ensemble_diversity(X_cancer_train, y_cancer_train)
    
    # Feature importance analysis
    importance = rf_custom.feature_importance()
    
    plt.figure(figsize=(12, 8))
    
    # Plot feature importance
    plt.subplot(2, 1, 1)
    top_features = np.argsort(importance)[-10:]  # Top 10 features
    plt.barh(range(len(top_features)), importance[top_features], alpha=0.7)
    plt.yticks(range(len(top_features)), [cancer.feature_names[i] for i in top_features])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest: Top 10 Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # Plot ensemble diversity
    plt.subplot(2, 1, 2)
    individual_accs = diversity_results['individual_accuracies']
    plt.hist(individual_accs, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(diversity_results['ensemble_accuracy'], color='red', linestyle='--', 
               linewidth=2, label=f'Ensemble: {diversity_results["ensemble_accuracy"]:.3f}')
    plt.axvline(np.mean(individual_accs), color='green', linestyle='--', 
               linewidth=2, label=f'Individual: {np.mean(individual_accs):.3f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Individual Tree vs Ensemble Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # GRADIENT BOOSTING SEQUENTIAL LEARNING
    # ================================================================
    
    print("\n" + "="*70)
    print("🚀 GRADIENT BOOSTING SEQUENTIAL LEARNING")
    print("="*70)
    
    print("\n1️⃣  Error Correction Through Sequential Models")
    
    # Convert to binary classification for gradient boosting
    y_cancer_binary = (y_cancer == 1).astype(int)
    X_gb_train, X_gb_test, y_gb_train, y_gb_test = train_test_split(
        X_cancer, y_cancer_binary, test_size=0.3, random_state=42
    )
    
    # Train custom gradient boosting
    gb_custom = GradientBoostingFromScratch(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    
    gb_custom.fit(X_gb_train, y_gb_train, 
                 feature_names=cancer.feature_names,
                 class_names=['Malignant', 'Benign'])
    
    # Analyze boosting progression
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training progression
    ax = axes[0, 0]
    train_errors = []
    test_errors = []
    
    for i, pred in enumerate(gb_custom.staged_predict(X_gb_train)):
        train_error = 1 - accuracy_score(y_gb_train, pred)
        train_errors.append(train_error)
    
    for i, pred in enumerate(gb_custom.staged_predict(X_gb_test)):
        test_error = 1 - accuracy_score(y_gb_test, pred)
        test_errors.append(test_error)
    
    ax.plot(train_errors, 'b-', label='Training Error', linewidth=2)
    ax.plot(test_errors, 'r-', label='Test Error', linewidth=2)
    ax.set_xlabel('Boosting Iteration')
    ax.set_ylabel('Classification Error')
    ax.set_title('Gradient Boosting: Error Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Feature importance
    ax = axes[0, 1]
    gb_importance = gb_custom.feature_importance()
    top_gb_features = np.argsort(gb_importance)[-10:]
    
    ax.barh(range(len(top_gb_features)), gb_importance[top_gb_features], alpha=0.7, color='orange')
    ax.set_yticks(range(len(top_gb_features)))
    ax.set_yticklabels([cancer.feature_names[i] for i in top_gb_features])
    ax.set_xlabel('Feature Importance')
    ax
