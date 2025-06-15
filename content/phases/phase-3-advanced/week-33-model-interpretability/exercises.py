"""
Neural Odyssey - Week 33: Model Interpretability and Explainable AI
Phase 2: Core Machine Learning (Week 21)

Beyond Black Boxes: Understanding How Models Think

This week explores the critical field of model interpretability and explainable AI.
You'll implement SHAP, LIME, and other state-of-the-art explanation techniques from
mathematical foundations, learning to make any machine learning model interpretable
and trustworthy for high-stakes applications.

Comprehensive exploration includes:
1. SHAP (SHapley Additive exPlanations) from game theory foundations
2. LIME (Local Interpretable Model-agnostic Explanations) theory and implementation
3. Feature importance, permutation importance, and partial dependence plots
4. Counterfactual explanations and adversarial examples for interpretability
5. Model-agnostic vs model-specific interpretation techniques
6. Evaluation metrics for explanation quality and human interpretability
7. Domain-specific applications: healthcare, finance, legal, autonomous systems
8. Production interpretability systems and real-time explanation generation

To get started, run: python exercises.py

Author: Neural Explorer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, make_classification, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import shap
import lime
import lime.lime_tabular
import lime.lime_image
import lime.lime_text
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("🧠 Neural Odyssey - Week 33: Model Interpretability and Explainable AI")
print("=" * 80)
print("Making black-box models transparent and trustworthy")
print("From mathematical foundations to production explanation systems")
print("=" * 80)


# ==========================================
# SHAPLEY VALUES FROM GAME THEORY
# ==========================================

class ShapleyValueCalculator:
    """
    Pure implementation of Shapley values from cooperative game theory
    Understanding the mathematical foundations before using libraries
    """
    
    def __init__(self, model, background_data, feature_names=None):
        """
        Initialize Shapley value calculator
        
        Args:
            model: Trained ML model with predict method
            background_data: Background dataset for baseline calculations
            feature_names: Names of features for interpretation
        """
        self.model = model
        self.background_data = background_data
        self.baseline = np.mean(model.predict(background_data))
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(background_data.shape[1])]
        self.n_features = background_data.shape[1]
    
    def coalition_value(self, instance, coalition_indices):
        """
        Calculate the value of a coalition of features
        
        Args:
            instance: Single instance to explain
            coalition_indices: Indices of features in the coalition
            
        Returns:
            Expected prediction value for the coalition
        """
        # Create coalition by keeping coalition features, replacing others with background
        coalition_samples = []
        
        for bg_sample in self.background_data[:100]:  # Sample background for efficiency
            coalition_sample = bg_sample.copy()
            for idx in coalition_indices:
                coalition_sample[idx] = instance[idx]
            coalition_samples.append(coalition_sample)
        
        coalition_samples = np.array(coalition_samples)
        predictions = self.model.predict(coalition_samples)
        return np.mean(predictions)
    
    def exact_shapley_values(self, instance):
        """
        Calculate exact Shapley values for all features
        Warning: Exponential complexity - only for small feature sets!
        
        Args:
            instance: Single instance to explain
            
        Returns:
            Array of Shapley values for each feature
        """
        shapley_values = np.zeros(self.n_features)
        
        for feature_idx in range(self.n_features):
            feature_contribution = 0
            
            # Iterate over all possible coalitions not containing this feature
            for coalition_size in range(self.n_features):
                # Generate all coalitions of given size
                other_features = [i for i in range(self.n_features) if i != feature_idx]
                
                if coalition_size == 0:
                    coalitions = [[]]
                elif coalition_size <= len(other_features):
                    coalitions = list(combinations(other_features, coalition_size))
                else:
                    continue
                
                for coalition in coalitions:
                    coalition = list(coalition)
                    
                    # Calculate marginal contribution
                    value_with = self.coalition_value(instance, coalition + [feature_idx])
                    value_without = self.coalition_value(instance, coalition)
                    marginal_contribution = value_with - value_without
                    
                    # Weight by Shapley formula
                    coalition_size = len(coalition)
                    weight = (np.math.factorial(coalition_size) * 
                             np.math.factorial(self.n_features - coalition_size - 1) / 
                             np.math.factorial(self.n_features))
                    
                    feature_contribution += weight * marginal_contribution
            
            shapley_values[feature_idx] = feature_contribution
        
        return shapley_values
    
    def approximate_shapley_values(self, instance, n_samples=1000):
        """
        Approximate Shapley values using sampling
        Much more efficient for high-dimensional features
        
        Args:
            instance: Single instance to explain
            n_samples: Number of coalition samples to use
            
        Returns:
            Array of approximate Shapley values
        """
        shapley_values = np.zeros(self.n_features)
        
        for _ in range(n_samples):
            # Random permutation of features
            feature_order = np.random.permutation(self.n_features)
            
            # Calculate marginal contributions along this permutation
            for i, feature_idx in enumerate(feature_order):
                # Coalition is all features before this one in the permutation
                coalition = feature_order[:i]
                
                # Calculate marginal contribution
                value_with = self.coalition_value(instance, list(coalition) + [feature_idx])
                value_without = self.coalition_value(instance, coalition) if len(coalition) > 0 else self.baseline
                
                marginal_contribution = value_with - value_without
                shapley_values[feature_idx] += marginal_contribution
        
        # Average over all samples
        shapley_values /= n_samples
        
        return shapley_values
    
    def explain_instance(self, instance, method='approximate', n_samples=1000):
        """
        Generate complete explanation for an instance
        
        Args:
            instance: Instance to explain
            method: 'exact' or 'approximate'
            n_samples: Number of samples for approximation
            
        Returns:
            Dictionary with explanation details
        """
        if method == 'exact' and self.n_features > 10:
            print("⚠️  Warning: Exact Shapley values infeasible for >10 features. Using approximation.")
            method = 'approximate'
        
        if method == 'exact':
            shapley_values = self.exact_shapley_values(instance)
        else:
            shapley_values = self.approximate_shapley_values(instance, n_samples)
        
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        # Verify efficiency property (values should sum to prediction - baseline)
        expected_sum = prediction - self.baseline
        actual_sum = np.sum(shapley_values)
        
        explanation = {
            'prediction': prediction,
            'baseline': self.baseline,
            'shapley_values': shapley_values,
            'feature_names': self.feature_names,
            'feature_values': instance,
            'expected_sum': expected_sum,
            'actual_sum': actual_sum,
            'efficiency_error': abs(expected_sum - actual_sum)
        }
        
        return explanation
    
    def plot_explanation(self, explanation, title="Shapley Value Explanation"):
        """Plot Shapley value explanation"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Waterfall plot
        ax = axes[0]
        shapley_values = explanation['shapley_values']
        feature_names = explanation['feature_names']
        
        # Sort by absolute value for better visualization
        sorted_indices = np.argsort(np.abs(shapley_values))[::-1]
        sorted_values = shapley_values[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create waterfall plot
        cumulative = [explanation['baseline']]
        colors = ['green' if v > 0 else 'red' for v in sorted_values]
        
        bars = ax.barh(range(len(sorted_values)), sorted_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Shapley Value')
        ax.set_title('Feature Contributions (Shapley Values)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        # Summary plot
        ax = axes[1]
        components = ['Baseline', 'Feature\nContributions', 'Prediction']
        values = [explanation['baseline'], 
                 np.sum(shapley_values), 
                 explanation['prediction']]
        
        bars = ax.bar(components, values, color=['blue', 'orange', 'green'], alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Prediction Decomposition')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Add efficiency check
        ax.text(0.5, 0.95, f'Efficiency Error: {explanation["efficiency_error"]:.6f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


# ==========================================
# LIME IMPLEMENTATION FROM SCRATCH
# ==========================================

class LimeExplainerTabular:
    """
    LIME (Local Interpretable Model-agnostic Explanations) from scratch
    Understanding local linear approximations of complex models
    """
    
    def __init__(self, training_data, feature_names=None, categorical_features=None,
                 kernel_width=0.75, discretize_continuous=True):
        """
        Initialize LIME explainer for tabular data
        
        Args:
            training_data: Training dataset for sampling and statistics
            feature_names: Names of features
            categorical_features: Indices of categorical features
            kernel_width: Width of the exponential kernel
            discretize_continuous: Whether to discretize continuous features
        """
        self.training_data = training_data
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(training_data.shape[1])]
        self.categorical_features = categorical_features or []
        self.continuous_features = [i for i in range(training_data.shape[1]) 
                                  if i not in self.categorical_features]
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        
        # Compute statistics for continuous features
        self.feature_means = np.mean(training_data, axis=0)
        self.feature_stds = np.std(training_data, axis=0)
        
        # Discretization for continuous features
        if discretize_continuous:
            self.discretizer = self._build_discretizer()
    
    def _build_discretizer(self):
        """Build discretizer for continuous features"""
        discretizer = {}
        for feature_idx in self.continuous_features:
            feature_values = self.training_data[:, feature_idx]
            # Use quartiles for discretization
            quartiles = np.percentile(feature_values, [25, 50, 75])
            discretizer[feature_idx] = quartiles
        return discretizer
    
    def _discretize_feature(self, feature_idx, value):
        """Discretize a continuous feature value"""
        if feature_idx not in self.continuous_features:
            return value
        
        quartiles = self.discretizer[feature_idx]
        if value <= quartiles[0]:
            return 0  # Low
        elif value <= quartiles[1]:
            return 1  # Medium-Low
        elif value <= quartiles[2]:
            return 2  # Medium-High
        else:
            return 3  # High
    
    def _interpretable_representation(self, instance):
        """Convert instance to interpretable binary representation"""
        if not self.discretize_continuous:
            return instance
        
        interpretable = np.zeros(len(self.feature_names))
        for i, value in enumerate(instance):
            if i in self.categorical_features:
                interpretable[i] = value
            else:
                interpretable[i] = self._discretize_feature(i, value)
        
        return interpretable
    
    def _generate_perturbations(self, instance, num_samples=5000):
        """
        Generate perturbations around an instance
        
        Args:
            instance: Instance to perturb
            num_samples: Number of perturbations to generate
            
        Returns:
            perturbations: Array of perturbed instances
            interpretable_perturbations: Binary representation of perturbations
            distances: Distances from original instance
        """
        perturbations = []
        interpretable_perturbations = []
        
        instance_interpretable = self._interpretable_representation(instance)
        
        for _ in range(num_samples):
            perturbed = instance.copy()
            interpretable_perturbed = instance_interpretable.copy()
            
            # Randomly turn features on/off
            for feature_idx in range(len(instance)):
                if np.random.random() < 0.5:  # 50% chance to perturb
                    if feature_idx in self.categorical_features:
                        # Sample from training data for categorical
                        perturbed[feature_idx] = np.random.choice(
                            self.training_data[:, feature_idx]
                        )
                        interpretable_perturbed[feature_idx] = 0  # Turned off
                    else:
                        # Sample from normal distribution for continuous
                        perturbed[feature_idx] = np.random.normal(
                            self.feature_means[feature_idx],
                            self.feature_stds[feature_idx]
                        )
                        interpretable_perturbed[feature_idx] = 0  # Turned off
                else:
                    interpretable_perturbed[feature_idx] = 1  # Kept original
            
            perturbations.append(perturbed)
            interpretable_perturbations.append(interpretable_perturbed)
        
        perturbations = np.array(perturbations)
        interpretable_perturbations = np.array(interpretable_perturbations)
        
        # Calculate distances for kernel weights
        distances = np.sqrt(np.sum((interpretable_perturbations - 
                                  instance_interpretable) ** 2, axis=1))
        
        return perturbations, interpretable_perturbations, distances
    
    def _calculate_weights(self, distances):
        """Calculate kernel weights based on distances"""
        kernel_width = self.kernel_width * np.sqrt(len(self.feature_names))
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        return weights
    
    def explain_instance(self, instance, model, num_features=10, num_samples=5000):
        """
        Explain a single instance using LIME
        
        Args:
            instance: Instance to explain
            model: Black-box model to explain
            num_features: Number of top features to include in explanation
            num_samples: Number of perturbations to generate
            
        Returns:
            Dictionary with explanation details
        """
        # Generate perturbations
        perturbations, interpretable_perturbations, distances = self._generate_perturbations(
            instance, num_samples
        )
        
        # Get predictions for perturbations
        predictions = model.predict(perturbations)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # For classification, use probability of positive class
            predictions = predictions[:, 1]
        
        # Calculate weights
        weights = self._calculate_weights(distances)
        
        # Fit interpretable model (weighted linear regression)
        from sklearn.linear_model import Ridge
        interpretable_model = Ridge(alpha=1.0)
        
        # Weighted fit
        sample_weights = weights
        interpretable_model.fit(interpretable_perturbations, predictions, 
                              sample_weight=sample_weights)
        
        # Get feature importances (coefficients)
        feature_importances = interpretable_model.coef_
        
        # Get top features
        top_features = np.argsort(np.abs(feature_importances))[-num_features:][::-1]
        
        # Get original prediction
        original_prediction = model.predict(instance.reshape(1, -1))[0]
        if len(original_prediction.shape) > 0 and len(original_prediction) > 1:
            original_prediction = original_prediction[1]  # Positive class probability
        
        # Local model prediction for original instance
        local_prediction = interpretable_model.predict(
            self._interpretable_representation(instance).reshape(1, -1)
        )[0]
        
        explanation = {
            'instance': instance,
            'prediction': original_prediction,
            'local_prediction': local_prediction,
            'feature_importances': feature_importances,
            'top_features': top_features,
            'feature_names': self.feature_names,
            'interpretable_model': interpretable_model,
            'r2_score': interpretable_model.score(
                interpretable_perturbations, predictions, sample_weight=weights
            )
        }
        
        return explanation
    
    def plot_explanation(self, explanation, title="LIME Explanation"):
        """Plot LIME explanation"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance plot
        ax = axes[0]
        top_features = explanation['top_features']
        importances = explanation['feature_importances'][top_features]
        feature_names = [explanation['feature_names'][i] for i in top_features]
        
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        bars = ax.barh(range(len(importances)), importances, color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title('LIME Feature Importances')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), 
                   bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', 
                   ha='left' if width >= 0 else 'right', va='center')
        
        # Model fidelity plot
        ax = axes[1]
        predictions = ['Original\nModel', 'Local\nModel']
        values = [explanation['prediction'], explanation['local_prediction']]
        
        bars = ax.bar(predictions, values, color=['blue', 'orange'], alpha=0.7)
        ax.set_ylabel('Prediction')
        ax.set_title('Model Fidelity Check')
        ax.grid(True, alpha=0.3)
        
        # Add value labels and R² score
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.text(0.5, 0.95, f'Local Model R²: {explanation["r2_score"]:.3f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


# ==========================================
# PERMUTATION IMPORTANCE
# ==========================================

class PermutationImportance:
    """
    Model-agnostic permutation importance calculation
    Measures feature importance by performance drop when feature is shuffled
    """
    
    def __init__(self, model, X, y, metric='accuracy', n_repeats=10):
        """
        Initialize permutation importance calculator
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            metric: Performance metric ('accuracy', 'r2', 'neg_mse')
            n_repeats: Number of permutation repeats
        """
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric
        self.n_repeats = n_repeats
        
        # Calculate baseline performance
        self.baseline_score = self._calculate_score(X, y)
    
    def _calculate_score(self, X, y):
        """Calculate performance score"""
        predictions = self.model.predict(X)
        
        if self.metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, predictions)
        elif self.metric == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y, predictions)
        elif self.metric == 'neg_mse':
            from sklearn.metrics import mean_squared_error
            return -mean_squared_error(y, predictions)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def calculate_importance(self, feature_names=None):
        """
        Calculate permutation importance for all features
        
        Args:
            feature_names: Names of features
            
        Returns:
            Dictionary with importance scores and statistics
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.X.shape[1])]
        
        importances = {}
        
        for feature_idx in range(self.X.shape[1]):
            feature_name = feature_names[feature_idx]
            scores = []
            
            for repeat in range(self.n_repeats):
                # Create permuted version
                X_permuted = self.X.copy()
                permutation = np.random.permutation(len(X_permuted))
                X_permuted[:, feature_idx] = X_permuted[permutation, feature_idx]
                
                # Calculate score with permuted feature
                permuted_score = self._calculate_score(X_permuted, self.y)
                
                # Importance is decrease in performance
                importance = self.baseline_score - permuted_score
                scores.append(importance)
            
            importances[feature_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return {
            'baseline_score': self.baseline_score,
            'importances': importances,
            'feature_names': feature_names
        }
    
    def plot_importance(self, importance_results, title="Permutation Importance"):
        """Plot permutation importance results"""
        importances = importance_results['importances']
        feature_names = list(importances.keys())
        means = [importances[name]['mean'] for name in feature_names]
        stds = [importances[name]['std'] for name in feature_names]
        
        # Sort by importance
        sorted_indices = np.argsort(means)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with error bars
        bars = plt.barh(range(len(sorted_names)), sorted_means, 
                       xerr=sorted_stds, capsize=5, alpha=0.7)
        
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel(f'Importance (decrease in {self.metric})')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            if sorted_means[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add baseline score info
        plt.text(0.02, 0.98, f'Baseline {self.metric}: {importance_results["baseline_score"]:.4f}',
                transform=plt.gca().transAxes, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()


# ==========================================
# PARTIAL DEPENDENCE PLOTS
# ==========================================

class PartialDependencePlotter:
    """
    Generate partial dependence plots to understand feature effects
    Shows average effect of a feature across all other feature values
    """
    
    def __init__(self, model, X, feature_names=None):
        """
        Initialize partial dependence plotter
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
    
    def partial_dependence(self, feature_idx, grid_resolution=100):
        """
        Calculate partial dependence for a single feature
        
        Args:
            feature_idx: Index of feature to analyze
            grid_resolution: Number of points in the grid
            
        Returns:
            grid_values: Feature values used for calculation
            pd_values: Partial dependence values
        """
        feature_values = self.X[:, feature_idx]
        grid_values = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)
        
        pd_values = []
        
        for grid_value in grid_values:
            # Create dataset with feature fixed at grid_value
            X_modified = self.X.copy()
            X_modified[:, feature_idx] = grid_value
            
            # Get predictions and average
            predictions = self.model.predict(X_modified)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # For classification, use probability of positive class
                predictions = predictions[:, 1]
            
            pd_values.append(np.mean(predictions))
        
        return grid_values, np.array(pd_values)
    
    def two_way_partial_dependence(self, feature_idx1, feature_idx2, grid_resolution=50):
        """
        Calculate two-way partial dependence for feature interactions
        
        Args:
            feature_idx1: Index of first feature
            feature_idx2: Index of second feature
            grid_resolution: Number of points per dimension
            
        Returns:
            grid1, grid2: Meshgrid of feature values
            pd_values: 2D array of partial dependence values
        """
        feature1_values = self.X[:, feature_idx1]
        feature2_values = self.X[:, feature_idx2]
        
        grid1_values = np.linspace(feature1_values.min(), feature1_values.max(), grid_resolution)
        grid2_values = np.linspace(feature2_values.min(), feature2_values.max(), grid_resolution)
        
        grid1, grid2 = np.meshgrid(grid1_values, grid2_values)
        pd_values = np.zeros_like(grid1)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Create dataset with both features fixed
                X_modified = self.X.copy()
                X_modified[:, feature_idx1] = grid1[i, j]
                X_modified[:, feature_idx2] = grid2[i, j]
                
                # Get predictions and average
                predictions = self.model.predict(X_modified)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 1]
                
                pd_values[i, j] = np.mean(predictions)
        
        return grid1, grid2, pd_values
    
    def plot_partial_dependence(self, feature_indices, figsize=(15, 10)):
        """
        Plot partial dependence for multiple features
        
        Args:
            feature_indices: List of feature indices to plot
            figsize: Figure size
        """
        n_features = len(feature_indices)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feature_idx in enumerate(feature_indices):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Calculate partial dependence
            grid_values, pd_values = self.partial_dependence(feature_idx)
            
            # Plot
            ax.plot(grid_values, pd_values, linewidth=2, color='blue')
            ax.set_xlabel(self.feature_names[feature_idx])
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence: {self.feature_names[feature_idx]}')
            ax.grid(True, alpha=0.3)
            
            # Add rug plot showing data distribution
            feature_values = self.X[:, feature_idx]
            ax2 = ax.twinx()
            ax2.hist(feature_values, bins=50, alpha=0.3, color='gray')
            ax2.set_ylabel('Data Density')
            ax2.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Partial Dependence Plots')
        plt.tight_layout()
        plt.show()
    
    def plot_interaction(self, feature_idx1, feature_idx2):
        """
        Plot two-way partial dependence interaction
        
        Args:
            feature_idx1: Index of first feature
            feature_idx2: Index of second feature
        """
        grid1, grid2, pd_values = self.two_way_partial_dependence(feature_idx1, feature_idx2)
        
        plt.figure(figsize=(12, 5))
        
        # Contour plot
        plt.subplot(1, 2, 1)
        contour = plt.contourf(grid1, grid2, pd_values, levels=20, cmap='viridis')
        plt.colorbar(contour)
        plt.xlabel(self.feature_names[feature_idx1])
        plt.ylabel(self.feature_names[feature_idx2])
        plt.title('Partial Dependence Interaction (Contour)')
        
        # 3D surface plot
        ax = plt.subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(grid1, grid2, pd_values, cmap='viridis', alpha=0.8)
        ax.set_xlabel(self.feature_names[feature_idx1])
        ax.set_ylabel(self.feature_names[feature_idx2])
        ax.set_zlabel('Partial Dependence')
        ax.set_title('Partial Dependence Interaction (3D)')
        plt.colorbar(surf)
        
        plt.tight_layout()
        plt.show()


# ==========================================
# COUNTERFACTUAL EXPLANATIONS
# ==========================================

class CounterfactualExplainer:
    """
    Generate counterfactual explanations - minimal changes for different predictions
    """
    
    def __init__(self, model, X_train, feature_ranges=None, categorical_features=None):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Trained model
            X_train: Training data for feature ranges
            feature_ranges: Manual feature ranges as [(min, max), ...]
            categorical_features: Indices of categorical features
        """
        self.model = model
        self.X_train = X_train
        self.categorical_features = categorical_features or []
        
        # Calculate feature ranges
        if feature_ranges is None:
            self.feature_ranges = [
                (X_train[:, i].min(), X_train[:, i].max()) 
                for i in range(X_train.shape[1])
            ]
        else:
            self.feature_ranges = feature_ranges
    
    def generate_counterfactual(self, instance, target_class=None, max_iterations=1000, 
                              learning_rate=0.01, lambda_reg=0.1):
        """
        Generate counterfactual using gradient-based optimization
        
        Args:
            instance: Original instance
            target_class: Desired prediction class (None for opposite)
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            lambda_reg: Regularization strength for L1 distance
            
        Returns:
            Dictionary with counterfactual and metadata
        """
        instance = instance.copy().astype(float)
        original_prediction = self.model.predict_proba(instance.reshape(1, -1))[0]
        original_class = np.argmax(original_prediction)
        
        # Set target class
        if target_class is None:
            target_class = 1 - original_class  # Flip for binary
        
        # Initialize counterfactual
        counterfactual = instance.copy()
        
        # Optimization history
        history = {
            'distances': [],
            'predictions': [],
            'losses': []
        }
        
        for iteration in range(max_iterations):
            # Forward pass
            pred_proba = self.model.predict_proba(counterfactual.reshape(1, -1))[0]
            
            # Loss: prediction loss + L1 regularization
            prediction_loss = -np.log(pred_proba[target_class] + 1e-10)
            distance_loss = lambda_reg * np.sum(np.abs(counterfactual - instance))
            total_loss = prediction_loss + distance_loss
            
            # Store history
            history['distances'].append(np.sum(np.abs(counterfactual - instance)))
            history['predictions'].append(pred_proba[target_class])
            history['losses'].append(total_loss)
            
            # Check convergence
            if pred_proba[target_class] > 0.5:
                break
            
            # Numerical gradient computation
            gradient = np.zeros_like(counterfactual)
            epsilon = 1e-6
            
            for i in range(len(counterfactual)):
                # Skip categorical features for now
                if i in self.categorical_features:
                    continue
                
                # Forward difference
                counterfactual[i] += epsilon
                pred_plus = self.model.predict_proba(counterfactual.reshape(1, -1))[0]
                loss_plus = -np.log(pred_plus[target_class] + 1e-10) + \
                           lambda_reg * np.sum(np.abs(counterfactual - instance))
                
                counterfactual[i] -= 2 * epsilon
                pred_minus = self.model.predict_proba(counterfactual.reshape(1, -1))[0]
                loss_minus = -np.log(pred_minus[target_class] + 1e-10) + \
                            lambda_reg * np.sum(np.abs(counterfactual - instance))
                
                # Restore original value
                counterfactual[i] += epsilon
                
                # Compute gradient
                gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Gradient descent step
            counterfactual -= learning_rate * gradient
            
            # Project to feature ranges
            for i in range(len(counterfactual)):
                if i not in self.categorical_features:
                    min_val, max_val = self.feature_ranges[i]
                    counterfactual[i] = np.clip(counterfactual[i], min_val, max_val)
        
        # Final predictions
        final_prediction = self.model.predict_proba(counterfactual.reshape(1, -1))[0]
        final_class = np.argmax(final_prediction)
        
        return {
            'original_instance': instance,
            'counterfactual': counterfactual,
            'original_prediction': original_prediction,
            'counterfactual_prediction': final_prediction,
            'original_class': original_class,
            'counterfactual_class': final_class,
            'distance': np.sum(np.abs(counterfactual - instance)),
            'feature_changes': counterfactual - instance,
            'success': final_class == target_class,
            'iterations': iteration + 1,
            'history': history
        }
    
    def plot_counterfactual(self, result, feature_names=None):
        """Plot counterfactual explanation"""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(result['original_instance']))]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature changes
        ax = axes[0, 0]
        changes = result['feature_changes']
        significant_changes = np.abs(changes) > 1e-6
        
        if np.any(significant_changes):
            change_indices = np.where(significant_changes)[0]
            change_values = changes[change_indices]
            change_names = [feature_names[i] for i in change_indices]
            
            colors = ['green' if c > 0 else 'red' for c in change_values]
            bars = ax.barh(range(len(change_values)), change_values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(change_names)))
            ax.set_yticklabels(change_names)
            ax.set_xlabel('Feature Change')
            ax.set_title('Required Feature Changes')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, change in zip(bars, change_values):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01), 
                       bar.get_y() + bar.get_height()/2,
                       f'{change:.3f}', 
                       ha='left' if width >= 0 else 'right', va='center')
        else:
            ax.text(0.5, 0.5, 'No significant changes required', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Required Feature Changes')
        
        # Prediction comparison
        ax = axes[0, 1]
        original_pred = result['original_prediction']
        counterfactual_pred = result['counterfactual_prediction']
        
        x = np.arange(len(original_pred))
        width = 0.35
        
        ax.bar(x - width/2, original_pred, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, counterfactual_pred, width, label='Counterfactual', alpha=0.7)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in range(len(original_pred))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Optimization history
        ax = axes[1, 0]
        history = result['history']
        ax.plot(history['distances'], label='L1 Distance', color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance from Original')
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Target class probability
        ax = axes[1, 1]
        ax.plot(history['predictions'], label='Target Class Probability', color='green')
        ax.axhline(y=0.5, color='red', linestyle='--', label='Decision Boundary')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Probability')
        ax.set_title('Target Class Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        success_text = "SUCCESS" if result['success'] else "FAILED"
        color = "green" if result['success'] else "red"
        fig.suptitle(f'Counterfactual Explanation - {success_text}\n'
                    f'Distance: {result["distance"]:.3f}, Iterations: {result["iterations"]}',
                    color=color, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# ==========================================
# COMPREHENSIVE INTERPRETABILITY FRAMEWORK
# ==========================================

class InterpretabilityFramework:
    """
    Comprehensive framework combining multiple interpretation techniques
    """
    
    def __init__(self, model, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Initialize comprehensive interpretability framework
        
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Feature names
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Initialize explanation methods
        self.shapley_calculator = ShapleyValueCalculator(model, X_train, feature_names)
        self.lime_explainer = LimeExplainerTabular(X_train, feature_names)
        self.perm_importance = PermutationImportance(model, X_test, y_test)
        self.pdp_plotter = PartialDependencePlotter(model, X_train, feature_names)
        self.counterfactual_explainer = CounterfactualExplainer(model, X_train)
        
        print("🔍 Comprehensive Interpretability Framework Initialized")
        print(f"   Model: {type(model).__name__}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(feature_names)}")
    
    def global_interpretation(self):
        """Generate global model interpretation"""
        print("\n🌍 Global Model Interpretation")
        print("=" * 50)
        
        # Permutation importance
        print("Computing permutation importance...")
        perm_results = self.perm_importance.calculate_importance(self.feature_names)
        self.perm_importance.plot_importance(perm_results)
        
        # Partial dependence plots for top features
        print("Generating partial dependence plots...")
        importances = perm_results['importances']
        top_features = sorted(importances.keys(), 
                            key=lambda x: importances[x]['mean'], reverse=True)[:6]
        top_indices = [self.feature_names.index(name) for name in top_features]
        
        self.pdp_plotter.plot_partial_dependence(top_indices)
        
        # Feature interactions for top 2 features
        if len(top_indices) >= 2:
            print("Analyzing feature interactions...")
            self.pdp_plotter.plot_interaction(top_indices[0], top_indices[1])
        
        return perm_results
    
    def local_interpretation(self, instance_idx=0, comparison_methods=True):
        """
        Generate local interpretation for a specific instance
        
        Args:
            instance_idx: Index of instance to explain
            comparison_methods: Whether to compare SHAP and LIME
        """
        print(f"\n🎯 Local Interpretation - Instance {instance_idx}")
        print("=" * 50)
        
        instance = self.X_test[instance_idx]
        true_label = self.y_test[instance_idx]
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        print(f"True label: {true_label}")
        print(f"Prediction: {prediction}")
        
        results = {}
        
        # SHAP explanation
        print("Generating SHAP explanation...")
        shap_explanation = self.shapley_calculator.explain_instance(instance)
        self.shapley_calculator.plot_explanation(shap_explanation, 
                                                f"SHAP Explanation - Instance {instance_idx}")
        results['shap'] = shap_explanation
        
        # LIME explanation
        print("Generating LIME explanation...")
        lime_explanation = self.lime_explainer.explain_instance(instance, self.model)
        self.lime_explainer.plot_explanation(lime_explanation,
                                            f"LIME Explanation - Instance {instance_idx}")
        results['lime'] = lime_explanation
        
        # Compare methods if requested
        if comparison_methods:
            self._compare_explanations(shap_explanation, lime_explanation, instance_idx)
        
        return results
    
    def counterfactual_analysis(self, instance_idx=0):
        """Generate counterfactual explanation for an instance"""
        print(f"\n🔄 Counterfactual Analysis - Instance {instance_idx}")
        print("=" * 50)
        
        instance = self.X_test[instance_idx]
        
        print("Generating counterfactual explanation...")
        counterfactual_result = self.counterfactual_explainer.generate_counterfactual(instance)
        self.counterfactual_explainer.plot_counterfactual(counterfactual_result, self.feature_names)
        
        return counterfactual_result
    
    def _compare_explanations(self, shap_explanation, lime_explanation, instance_idx):
        """Compare SHAP and LIME explanations"""
        print(f"\n📊 Comparing SHAP vs LIME - Instance {instance_idx}")
        
        # Get top features from both methods
        shap_values = shap_explanation['shapley_values']
        shap_top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
        
        lime_importances = lime_explanation['feature_importances']
        lime_top_indices = np.argsort(np.abs(lime_importances))[-5:][::-1]
        
        plt.figure(figsize=(15, 6))
        
        # SHAP values
        plt.subplot(1, 2, 1)
        shap_top_values = shap_values[shap_top_indices]
        shap_top_names = [self.feature_names[i] for i in shap_top_indices]
        
        colors = ['green' if v > 0 else 'red' for v in shap_top_values]
        plt.barh(range(len(shap_top_values)), shap_top_values, color=colors, alpha=0.7)
        plt.yticks(range(len(shap_top_names)), shap_top_names)
        plt.xlabel('SHAP Value')
        plt.title('Top Features - SHAP')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # LIME values
        plt.subplot(1, 2, 2)
        lime_top_values = lime_importances[lime_top_indices]
        lime_top_names = [self.feature_names[i] for i in lime_top_indices]
        
        colors = ['green' if v > 0 else 'red' for v in lime_top_values]
        plt.barh(range(len(lime_top_values)), lime_top_values, color=colors, alpha=0.7)
        plt.yticks(range(len(lime_top_names)), lime_top_names)
        plt.xlabel('LIME Importance')
        plt.title('Top Features - LIME')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'SHAP vs LIME Comparison - Instance {instance_idx}')
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        # Align features and compute correlation
        common_indices = np.intersect1d(shap_top_indices, lime_top_indices)
        if len(common_indices) > 2:
            shap_common = [shap_values[i] for i in common_indices]
            lime_common = [lime_importances[i] for i in common_indices]
            
            correlation = np.corrcoef(shap_common, lime_common)[0, 1]
            print(f"Correlation between SHAP and LIME for common features: {correlation:.3f}")
    
    def comprehensive_report(self, instance_indices=[0, 1, 2]):
        """Generate comprehensive interpretability report"""
        print("\n📋 COMPREHENSIVE INTERPRETABILITY REPORT")
        print("=" * 60)
        
        # Global interpretation
        global_results = self.global_interpretation()
        
        # Local interpretations for multiple instances
        local_results = {}
        for idx in instance_indices:
            if idx < len(self.X_test):
                local_results[idx] = self.local_interpretation(idx, comparison_methods=False)
        
        # Counterfactual analysis
        counterfactual_results = {}
        for idx in instance_indices[:2]:  # Limit to 2 for efficiency
            if idx < len(self.X_test):
                counterfactual_results[idx] = self.counterfactual_analysis(idx)
        
        # Summary statistics
        print("\n📈 SUMMARY STATISTICS")
        print("=" * 30)
        
        # Model performance
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Feature importance summary
        importances = global_results['importances']
        top_feature = max(importances.keys(), key=lambda x: importances[x]['mean'])
        print(f"Most important feature: {top_feature} ({importances[top_feature]['mean']:.4f})")
        
        # Explanation consistency
        if len(local_results) > 1:
            shap_correlations = []
            for i in range(len(instance_indices)):
                for j in range(i+1, len(instance_indices)):
                    if instance_indices[i] in local_results and instance_indices[j] in local_results:
                        shap1 = local_results[instance_indices[i]]['shap']['shapley_values']
                        shap2 = local_results[instance_indices[j]]['shap']['shapley_values']
                        corr = np.corrcoef(shap1, shap2)[0, 1]
                        shap_correlations.append(corr)
            
            if shap_correlations:
                avg_correlation = np.mean(shap_correlations)
                print(f"Average SHAP correlation between instances: {avg_correlation:.4f}")
        
        return {
            'global': global_results,
            'local': local_results,
            'counterfactual': counterfactual_results
        }


# ==========================================
# DEMONSTRATION AND EXPERIMENTS
# ==========================================

def demonstrate_shapley_values():
    """Demonstrate Shapley value calculation from scratch"""
    print("\n🎲 Shapley Values from Game Theory")
    print("=" * 50)
    
    # Create simple dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=8, n_informative=5, 
                             n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Initialize Shapley calculator
    shapley_calc = ShapleyValueCalculator(model, X_train, feature_names)
    
    # Explain first test instance
    instance = X_test[0]
    print(f"Explaining instance with {len(instance)} features...")
    
    # Compare exact vs approximate (for small feature set)
    if len(instance) <= 8:
        print("Computing exact Shapley values...")
        explanation_exact = shapley_calc.explain_instance(instance, method='exact')
        shapley_calc.plot_explanation(explanation_exact, "Exact Shapley Values")
    
    print("Computing approximate Shapley values...")
    explanation_approx = shapley_calc.explain_instance(instance, method='approximate', n_samples=1000)
    shapley_calc.plot_explanation(explanation_approx, "Approximate Shapley Values")
    
    # Verify efficiency property
    print(f"\n🔍 Shapley Value Properties:")
    print(f"Prediction: {explanation_approx['prediction']:.4f}")
    print(f"Baseline: {explanation_approx['baseline']:.4f}")
    print(f"Sum of Shapley values: {explanation_approx['actual_sum']:.4f}")
    print(f"Expected sum (prediction - baseline): {explanation_approx['expected_sum']:.4f}")
    print(f"Efficiency error: {explanation_approx['efficiency_error']:.6f}")
    
    print("✅ Shapley values demonstration completed!")


def demonstrate_lime_explanation():
    """Demonstrate LIME explanation from scratch"""
    print("\n🔍 LIME Local Explanations")
    print("=" * 50)
    
    # Load wine dataset
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    feature_names = wine_data.feature_names
    
    # Binary classification for simplicity
    binary_mask = y != 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42
    )
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize LIME explainer
    lime_explainer = LimeExplainerTabular(X_train, feature_names)
    
    # Explain several instances
    for i in range(3):
        print(f"\nExplaining instance {i}...")
        instance = X_test[i]
        true_label = y_test[i]
        prediction = model.predict_proba(instance.reshape(1, -1))[0]
        
        print(f"True label: {true_label}")
        print(f"Predicted probabilities: {prediction}")
        
        explanation = lime_explainer.explain_instance(instance, model, num_features=8)
        lime_explainer.plot_explanation(explanation, f"LIME Explanation - Instance {i}")
        
        print(f"Local model R²: {explanation['r2_score']:.3f}")
    
    print("✅ LIME explanation demonstration completed!")


def demonstrate_interpretability_methods():
    """Demonstrate various interpretability methods"""
    print("\n🔬 Comprehensive Interpretability Methods")
    print("=" * 50)
    
    # Load breast cancer dataset
    cancer_data = load_breast_cancer()
    X, y = cancer_data.data, cancer_data.target
    feature_names = cancer_data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"Model accuracy: {model.score(X_test_scaled, y_test):.4f}")
    
    # Permutation importance
    print("\n📊 Permutation Importance Analysis")
    perm_imp = PermutationImportance(model, X_test_scaled, y_test, metric='accuracy')
    perm_results = perm_imp.calculate_importance(feature_names)
    perm_imp.plot_importance(perm_results)
    
    # Partial dependence plots
    print("\n📈 Partial Dependence Analysis")
    pdp_plotter = PartialDependencePlotter(model, X_train_scaled, feature_names)
    
    # Get top 4 features from permutation importance
    importances = perm_results['importances']
    top_features = sorted(importances.keys(), key=lambda x: importances[x]['mean'], reverse=True)[:4]
    top_indices = [list(feature_names).index(name) for name in top_features]
    
    pdp_plotter.plot_partial_dependence(top_indices)
    
    # Feature interaction
    if len(top_indices) >= 2:
        print("\n🔗 Feature Interaction Analysis")
        pdp_plotter.plot_interaction(top_indices[0], top_indices[1])
    
    print("✅ Interpretability methods demonstration completed!")


def demonstrate_counterfactual_explanations():
    """Demonstrate counterfactual explanation generation"""
    print("\n🔄 Counterfactual Explanations")
    print("=" * 50)
    
    # Create simple 2D dataset for visualization
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
    
    # Initialize counterfactual explainer
    cf_explainer = CounterfactualExplainer(model, X_train, feature_names=['Feature_0', 'Feature_1'])
    
    # Generate counterfactuals for several instances
    for i in range(3):
        print(f"\nGenerating counterfactual for instance {i}...")
        instance = X_test[i]
        original_class = model.predict(instance.reshape(1, -1))[0]
        
        print(f