"""
Neural Odyssey - Week 13: Supervised Learning Foundations
Phase 2: Core Machine Learning (Week 1)

The Foundation of Predictive Learning

This week establishes the fundamental concepts of supervised learning that underpin
all predictive algorithms. Building on Phase 1's mathematical foundations, you'll
explore bias-variance tradeoffs, overfitting, cross-validation, and the learning
problem from both theoretical and practical perspectives.

Learning Objectives:
- Master the bias-variance tradeoff and its implications
- Understand overfitting, underfitting, and generalization
- Implement cross-validation and model selection techniques
- Explore learning curves and capacity control
- Build comprehensive model evaluation frameworks

Author: Neural Explorer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# BIAS-VARIANCE DECOMPOSITION FRAMEWORK
# ==========================================

class BiasVarianceAnalyzer:
    """
    Complete framework for understanding and visualizing bias-variance tradeoff
    """
    
    def __init__(self):
        self.results = {}
        
    def generate_true_function_data(self, n_samples=100, noise_level=0.1, random_state=42):
        """Generate data from a known true function for bias-variance analysis"""
        np.random.seed(random_state)
        
        # True function: quadratic with some complexity
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        true_function = lambda x: 1.5 * x.ravel() - 2 * (x.ravel() ** 2) + 0.5 * np.sin(15 * x.ravel())
        y_true = true_function(X)
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        y_observed = y_true + noise
        
        return X, y_observed, y_true, true_function
    
    def bias_variance_decomposition(self, model_class, model_params, X, y, true_function, 
                                   n_experiments=100, test_size=0.3):
        """
        Perform bias-variance decomposition by training many models on different datasets
        """
        print(f"🎯 Bias-Variance Decomposition Analysis")
        print(f"   Model: {model_class.__name__}")
        print(f"   Parameters: {model_params}")
        print(f"   Experiments: {n_experiments}")
        
        # Create test set for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Get true values for test set
        y_test_true = true_function(X_test)
        
        # Store predictions from multiple experiments
        predictions = []
        
        for experiment in range(n_experiments):
            # Create bootstrap sample
            n_train = len(X_train)
            bootstrap_indices = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[bootstrap_indices]
            y_boot = y_train[bootstrap_indices]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_boot, y_boot)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)  # Shape: (n_experiments, n_test_samples)
        
        # Calculate bias-variance decomposition
        # Mean prediction across all experiments
        mean_prediction = np.mean(predictions, axis=0)
        
        # Bias²: (mean_prediction - true_value)²
        bias_squared = (mean_prediction - y_test_true) ** 2
        
        # Variance: E[(prediction - mean_prediction)²]
        variance = np.mean((predictions - mean_prediction) ** 2, axis=0)
        
        # Noise: inherent irreducible error
        noise = np.var(y_test - y_test_true) if len(y_test) == len(y_test_true) else 0.01
        
        # Total error decomposition: Error = Bias² + Variance + Noise
        total_error = bias_squared + variance + noise
        
        # Calculate averages
        avg_bias_squared = np.mean(bias_squared)
        avg_variance = np.mean(variance)
        avg_total_error = np.mean(total_error)
        
        print(f"\n📊 Decomposition Results:")
        print(f"   Average Bias²: {avg_bias_squared:.4f}")
        print(f"   Average Variance: {avg_variance:.4f}")
        print(f"   Noise: {noise:.4f}")
        print(f"   Total Error: {avg_total_error:.4f}")
        print(f"   Bias²/Total: {avg_bias_squared/avg_total_error:.1%}")
        print(f"   Variance/Total: {avg_variance/avg_total_error:.1%}")
        print(f"   Noise/Total: {noise/avg_total_error:.1%}")
        
        results = {
            'X_test': X_test,
            'y_test_true': y_test_true,
            'predictions': predictions,
            'mean_prediction': mean_prediction,
            'bias_squared': bias_squared,
            'variance': variance,
            'noise': noise,
            'avg_bias_squared': avg_bias_squared,
            'avg_variance': avg_variance,
            'avg_total_error': avg_total_error
        }
        
        return results
    
    def compare_model_complexity(self, X, y, true_function):
        """Compare bias-variance tradeoff across different model complexities"""
        print(f"\n🔄 Model Complexity Comparison")
        
        # Test different polynomial degrees
        degrees = [1, 2, 5, 10, 15]
        complexity_results = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, degree in enumerate(degrees):
            print(f"\n   Testing Polynomial Degree {degree}...")
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            
            class PolynomialRegression:
                def __init__(self, degree):
                    self.degree = degree
                    self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                    self.linear_model = LinearRegression()
                    
                def fit(self, X, y):
                    X_poly = self.poly_features.fit_transform(X)
                    self.linear_model.fit(X_poly, y)
                    
                def predict(self, X):
                    X_poly = self.poly_features.transform(X)
                    return self.linear_model.predict(X_poly)
            
            # Perform bias-variance decomposition
            results = self.bias_variance_decomposition(
                PolynomialRegression, 
                {'degree': degree}, 
                X, y, true_function,
                n_experiments=50
            )
            
            complexity_results[degree] = results
            
            # Plot individual model results
            if i < 5:  # Plot first 5
                ax = axes[i//3, i%3]
                
                # Plot true function
                X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
                y_true_plot = true_function(X_plot)
                ax.plot(X_plot, y_true_plot, 'r-', linewidth=3, label='True Function', alpha=0.8)
                
                # Plot some individual predictions
                X_test = results['X_test']
                predictions = results['predictions']
                
                for j in range(0, min(10, len(predictions))):
                    ax.plot(X_test, predictions[j], 'b-', alpha=0.1, linewidth=1)
                
                # Plot mean prediction
                ax.plot(X_test, results['mean_prediction'], 'g-', linewidth=2, 
                       label='Mean Prediction')
                
                # Plot original data points
                ax.scatter(X, y, alpha=0.3, s=20, color='black', label='Data')
                
                ax.set_title(f'Degree {degree}\nBias²={results["avg_bias_squared"]:.3f}, '
                           f'Var={results["avg_variance"]:.3f}')
                ax.set_xlabel('X')
                ax.set_ylabel('y')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Summary plot
        ax = axes[1, 2]
        degrees_list = list(complexity_results.keys())
        bias_values = [complexity_results[d]['avg_bias_squared'] for d in degrees_list]
        variance_values = [complexity_results[d]['avg_variance'] for d in degrees_list]
        total_error_values = [complexity_results[d]['avg_total_error'] for d in degrees_list]
        
        ax.plot(degrees_list, bias_values, 'r.-', linewidth=2, markersize=8, label='Bias²')
        ax.plot(degrees_list, variance_values, 'b.-', linewidth=2, markersize=8, label='Variance')
        ax.plot(degrees_list, total_error_values, 'g.-', linewidth=2, markersize=8, label='Total Error')
        
        ax.set_title('Bias-Variance Tradeoff')
        ax.set_xlabel('Model Complexity (Polynomial Degree)')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return complexity_results
    
    def demonstrate_regularization_effect(self, X, y, true_function):
        """Show how regularization affects bias-variance tradeoff"""
        print(f"\n🎛️  Regularization Effects on Bias-Variance")
        
        # Test different regularization strengths
        alpha_values = [0.0, 0.01, 0.1, 1.0, 10.0]
        reg_results = {}
        
        # Use high-degree polynomial with regularization
        degree = 10
        
        class RegularizedPolynomialRegression:
            def __init__(self, degree, alpha):
                self.degree = degree
                self.alpha = alpha
                self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                self.ridge_model = Ridge(alpha=alpha)
                
            def fit(self, X, y):
                X_poly = self.poly_features.fit_transform(X)
                self.ridge_model.fit(X_poly, y)
                
            def predict(self, X):
                X_poly = self.poly_features.transform(X)
                return self.ridge_model.predict(X_poly)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, alpha in enumerate(alpha_values):
            print(f"   Testing Ridge Alpha = {alpha}...")
            
            results = self.bias_variance_decomposition(
                RegularizedPolynomialRegression,
                {'degree': degree, 'alpha': alpha},
                X, y, true_function,
                n_experiments=50
            )
            
            reg_results[alpha] = results
            
            # Plot results
            if i < 5:
                ax = axes[i//3, i%3]
                
                # Plot true function
                X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
                y_true_plot = true_function(X_plot)
                ax.plot(X_plot, y_true_plot, 'r-', linewidth=3, label='True Function')
                
                # Plot some predictions
                X_test = results['X_test']
                predictions = results['predictions']
                
                for j in range(0, min(10, len(predictions))):
                    ax.plot(X_test, predictions[j], 'b-', alpha=0.1, linewidth=1)
                
                ax.plot(X_test, results['mean_prediction'], 'g-', linewidth=2, 
                       label='Mean Prediction')
                ax.scatter(X, y, alpha=0.3, s=20, color='black')
                
                ax.set_title(f'Ridge α={alpha}\nBias²={results["avg_bias_squared"]:.3f}, '
                           f'Var={results["avg_variance"]:.3f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Summary plot
        ax = axes[1, 2]
        alphas_list = list(reg_results.keys())
        bias_values = [reg_results[a]['avg_bias_squared'] for a in alphas_list]
        variance_values = [reg_results[a]['avg_variance'] for a in alphas_list]
        total_error_values = [reg_results[a]['avg_total_error'] for a in alphas_list]
        
        ax.semilogx(alphas_list, bias_values, 'r.-', linewidth=2, label='Bias²')
        ax.semilogx(alphas_list, variance_values, 'b.-', linewidth=2, label='Variance')
        ax.semilogx(alphas_list, total_error_values, 'g.-', linewidth=2, label='Total Error')
        
        ax.set_title('Regularization Effect on Bias-Variance')
        ax.set_xlabel('Regularization Strength (α)')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return reg_results


# ==========================================
# OVERFITTING AND GENERALIZATION ANALYSIS
# ==========================================

class OverfittingAnalyzer:
    """
    Comprehensive analysis of overfitting, underfitting, and generalization
    """
    
    def __init__(self):
        self.experiments = {}
        
    def demonstrate_overfitting_progression(self):
        """Show how overfitting develops as model complexity increases"""
        print(f"📈 Overfitting Progression Analysis")
        
        # Generate dataset with limited samples to encourage overfitting
        np.random.seed(42)
        n_samples = 50
        X = np.random.uniform(0, 1, n_samples).reshape(-1, 1)
        true_function = lambda x: 2 * np.sin(2 * np.pi * x.ravel()) + 0.5 * x.ravel()
        y = true_function(X) + np.random.normal(0, 0.1, n_samples)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        
        # Test different polynomial degrees
        degrees = range(1, 16)
        train_errors = []
        test_errors = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for degree in degrees:
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Calculate errors
            train_pred = model.predict(X_train_poly)
            test_pred = model.predict(X_test_poly)
            
            train_error = mean_squared_error(y_train, train_pred)
            test_error = mean_squared_error(y_test, test_pred)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
        
        # Plot 1: Training vs Test Error
        ax = axes[0, 0]
        ax.plot(degrees, train_errors, 'b.-', linewidth=2, label='Training Error')
        ax.plot(degrees, test_errors, 'r.-', linewidth=2, label='Test Error')
        
        # Mark optimal point
        optimal_degree = degrees[np.argmin(test_errors)]
        min_test_error = min(test_errors)
        ax.plot(optimal_degree, min_test_error, 'go', markersize=10, 
               label=f'Optimal (degree={optimal_degree})')
        
        ax.set_title('Overfitting Progression')
        ax.set_xlabel('Model Complexity (Polynomial Degree)')
        ax.set_ylabel('Mean Squared Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2-4: Show fits for different complexity levels
        complexity_examples = [2, optimal_degree, 12]
        titles = ['Underfitting', 'Good Fit', 'Overfitting']
        
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        y_true_plot = true_function(X_plot)
        
        for i, (degree, title) in enumerate(zip(complexity_examples, titles)):
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            
            # Fit model with this degree
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_plot_poly = poly_features.transform(X_plot)
            
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_plot_pred = model.predict(X_plot_poly)
            
            # Calculate errors for title
            train_error = train_errors[degree-1]
            test_error = test_errors[degree-1]
            
            # Plot
            ax.plot(X_plot, y_true_plot, 'g-', linewidth=3, label='True Function', alpha=0.7)
            ax.plot(X_plot, y_plot_pred, 'r-', linewidth=2, label=f'Polynomial (degree {degree})')
            ax.scatter(X_train, y_train, color='blue', alpha=0.6, s=50, label='Training Data')
            ax.scatter(X_test, y_test, color='orange', alpha=0.6, s=50, label='Test Data')
            
            ax.set_title(f'{title}\nTrain MSE: {train_error:.3f}, Test MSE: {test_error:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'degrees': degrees,
            'train_errors': train_errors,
            'test_errors': test_errors,
            'optimal_degree': optimal_degree
        }
    
    def learning_curve_analysis(self, model_class, model_params, X, y):
        """Analyze how performance changes with training set size"""
        print(f"\n📚 Learning Curve Analysis")
        print(f"   Model: {model_class.__name__}")
        
        # Different training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        model = model_class(**model_params)
        
        # Use sklearn's learning_curve function
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, random_state=42
        )
        
        # Convert to positive values
        train_scores = -train_scores
        test_scores = -test_scores
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Learning curve
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes_abs, train_mean, 'b.-', linewidth=2, label='Training Error')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        plt.plot(train_sizes_abs, test_mean, 'r.-', linewidth=2, label='Validation Error')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2, color='red')
        
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Gap analysis
        plt.subplot(1, 2, 2)
        gap = test_mean - train_mean
        plt.plot(train_sizes_abs, gap, 'g.-', linewidth=2, label='Generalization Gap')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title('Generalization Gap')
        plt.xlabel('Training Set Size')
        plt.ylabel('Test Error - Training Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analysis
        final_gap = gap[-1]
        print(f"\n📊 Learning Curve Analysis:")
        print(f"   Final training error: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
        print(f"   Final validation error: {test_mean[-1]:.4f} ± {test_std[-1]:.4f}")
        print(f"   Final generalization gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("   ⚠️  Large generalization gap suggests overfitting")
        elif final_gap < 0.02:
            print("   ✅ Small generalization gap suggests good generalization")
        else:
            print("   📊 Moderate generalization gap")
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'generalization_gap': gap
        }
    
    def validation_curve_analysis(self, model_class, param_name, param_range, X, y):
        """Analyze how a hyperparameter affects training and validation performance"""
        print(f"\n🎛️  Validation Curve Analysis")
        print(f"   Parameter: {param_name}")
        print(f"   Range: {param_range}")
        
        # Use sklearn's validation_curve function
        train_scores, test_scores = validation_curve(
            model_class(), X, y, param_name=param_name, param_range=param_range,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Convert to positive values
        train_scores = -train_scores
        test_scores = -test_scores
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Find optimal parameter
        optimal_idx = np.argmin(test_mean)
        optimal_param = param_range[optimal_idx]
        optimal_score = test_mean[optimal_idx]
        
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        
        plt.semilogx(param_range, train_mean, 'b.-', linewidth=2, label='Training Error')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        plt.semilogx(param_range, test_mean, 'r.-', linewidth=2, label='Validation Error')
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color='red')
        
        # Mark optimal point
        plt.plot(optimal_param, optimal_score, 'go', markersize=10, 
                label=f'Optimal {param_name}={optimal_param}')
        
        plt.title(f'Validation Curve - {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n📊 Optimal hyperparameter:")
        print(f"   {param_name} = {optimal_param}")
        print(f"   Validation error = {optimal_score:.4f}")
        
        return {
            'param_range': param_range,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'optimal_param': optimal_param,
            'optimal_score': optimal_score
        }


# ==========================================
# CROSS-VALIDATION FRAMEWORK
# ==========================================

class CrossValidationFramework:
    """
    Comprehensive cross-validation implementation and analysis
    """
    
    def __init__(self):
        self.cv_results = {}
        
    def implement_kfold_from_scratch(self, X, y, k=5, shuffle=True, random_state=42):
        """Implement k-fold cross-validation from scratch"""
        print(f"🔄 K-Fold Cross-Validation Implementation (k={k})")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        
        # Calculate fold sizes
        fold_size = n_samples // k
        remainder = n_samples % k
        
        folds = []
        start = 0
        
        for i in range(k):
            # Some folds get one extra sample if there's a remainder
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            
            fold_indices = indices[start:end]
            folds.append(fold_indices)
            start = end
        
        print(f"   Created {k} folds with sizes: {[len(fold) for fold in folds]}")
        
        return folds
    
    def cross_validate_from_scratch(self, model_class, model_params, X, y, k=5):
        """Perform cross-validation from scratch with detailed analysis"""
        print(f"\n🎯 Cross-Validation from Scratch")
        
        # Get fold indices
        folds = self.implement_kfold_from_scratch(X, y, k=k)
        
        fold_results = []
        
        for fold_idx, test_indices in enumerate(folds):
            print(f"   Fold {fold_idx + 1}/{k}...")
            
            # Create train/test split
            train_indices = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
            
            X_train_fold = X[train_indices]
            X_test_fold = X[test_indices]
            y_train_fold = y[train_indices]
            y_test_fold = y[test_indices]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            train_pred = model.predict(X_train_fold)
            test_pred = model.predict(X_test_fold)
            
            train_error = mean_squared_error(y_train_fold, train_pred)
            test_error = mean_squared_error(y_test_fold, test_pred)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_error': train_error,
                'test_error': test_error,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            })
        
        # Calculate statistics
        train_errors = [r['train_error'] for r in fold_results]
        test_errors = [r['test_error'] for r in fold_results]
        
        cv_mean = np.mean(test_errors)
        cv_std = np.std(test_errors)
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Mean CV Error: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Individual fold errors: {[f'{e:.4f}' for e in test_errors]}")
        
        # Visualize results
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Fold-by-fold results
        plt.subplot(1, 2, 1)
        fold_numbers = [r['fold'] for r in fold_results]
        plt.bar([f - 0.2 for f in fold_numbers], train_errors, width=0.4, 
               label='Training Error', alpha=0.7)
        plt.bar([f + 0.2 for f in fold_numbers], test_errors, width=0.4, 
               label='Test Error', alpha=0.7)
        
        plt.axhline(y=cv_mean, color='red', linestyle='--', 
                   label=f'CV Mean: {cv_mean:.4f}')
        plt.fill_between([0.5, k + 0.5], 
                        cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, color='red')
        
        plt.title('Cross-Validation Results by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        plt.subplot(1, 2, 2)
        plt.hist(test_errors, bins=max(3, k//2), alpha=0.7, edgecolor='black')
        plt.axvline(cv_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {cv_mean:.4f}')
        plt.axvline(cv_mean - cv_std, color='orange', linestyle=':', 
                   label=f'±1 std: {cv_std:.4f}')
        plt.axvline(cv_mean + cv_std, color='orange', linestyle=':')
        
        plt.title('Distribution of CV Errors')
        plt.xlabel('Test Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'fold_results': fold_results,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': test_errors
        }
    
    def compare_cv_strategies(self, X, y, model_class, model_params):
        """Compare different cross-validation strategies"""
        print(f"\n🔍 Cross-Validation Strategy Comparison")
        
        from sklearn.model_selection import (KFold, StratifiedKFold, ShuffleSplit, 
                                           LeaveOneOut, cross_val_score)
        
        # Define different CV strategies
        cv_strategies = {
            'KFold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
            'KFold (k=10)': KFold(n_splits=10, shuffle=True, random_state=42),
            'ShuffleSplit': ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
            'Leave-One-Out': LeaveOneOut() if len(X) <= 100 else None  # Only for small datasets
        }
        
        results = {}
        
        for name, cv_strategy in cv_strategies.items():
            if cv_strategy is None:
                continue
                
            print(f"   Testing {name}...")
            
            model = model_class(**model_params)
            scores = cross_val_score(model, X, y, cv=cv_strategy, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            scores = -scores  # Convert to positive
            
            results[name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'n_splits': len(scores)
            }
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mean scores with error bars
        ax = axes[0]
        names = list(results.keys())
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        bars = ax.bar(names, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=['blue', 'orange', 'green', 'red'][:len(names)])
        
        ax.set_title('CV Strategy Comparison')
        ax.set_ylabel('Mean CV Error')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # Plot 2: Score distributions
        ax = axes[1]
        for i, (name, result) in enumerate(results.items()):
            scores = result['scores']
            ax.hist(scores, bins=max(3, len(scores)//3), alpha=0.6, 
                   label=name, density=True)
        
        ax.set_title('Score Distributions')
        ax.set_xlabel('CV Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n📊 Strategy Comparison Summary:")
        for name, result in results.items():
            print(f"   {name:15}: {result['mean']:.4f} ± {result['std']:.4f} "
                  f"({result['n_splits']} splits)")
        
        return results
    
    def nested_cross_validation(self, X, y, model_class, param_grid):
        """Implement nested cross-validation for unbiased model selection"""
        print(f"\n🎯 Nested Cross-Validation for Model Selection")
        
        from sklearn.model_selection import GridSearchCV, cross_val_score
        
        # Outer CV loop (for final performance estimation)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Inner CV loop (for hyperparameter selection)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        nested_scores = []
        best_params_per_fold = []
        
        print(f"   Outer CV: 5 folds")
        print(f"   Inner CV: 3 folds")
        print(f"   Parameter grid: {param_grid}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\n   Outer fold {fold_idx + 1}/5...")
            
            X_train_outer = X[train_idx]
            X_test_outer = X[test_idx]
            y_train_outer = y[train_idx]
            y_test_outer = y[test_idx]
            
            # Inner CV for hyperparameter selection
            grid_search = GridSearchCV(
                model_class(), param_grid, cv=inner_cv, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Get best model and evaluate on outer test set
            best_model = grid_search.best_estimator_
            test_score = mean_squared_error(y_test_outer, best_model.predict(X_test_outer))
            
            nested_scores.append(test_score)
            best_params_per_fold.append(grid_search.best_params_)
            
            print(f"     Best params: {grid_search.best_params_}")
            print(f"     Test score: {test_score:.4f}")
        
        # Calculate final statistics
        nested_mean = np.mean(nested_scores)
        nested_std = np.std(nested_scores)
        
        print(f"\n📊 Nested CV Results:")
        print(f"   Unbiased performance estimate: {nested_mean:.4f} ± {nested_std:.4f}")
        print(f"   Individual fold scores: {[f'{s:.4f}' for s in nested_scores]}")
        
        # Analyze parameter stability
        print(f"\n🔧 Parameter Selection Stability:")
        param_names = list(param_grid.keys())
        for param_name in param_names:
            param_values = [params[param_name] for params in best_params_per_fold]
            unique_values, counts = np.unique(param_values, return_counts=True)
            
            print(f"   {param_name}:")
            for value, count in zip(unique_values, counts):
                print(f"     {value}: selected {count}/5 times ({count/5:.1%})")
        
        return {
            'nested_scores': nested_scores,
            'nested_mean': nested_mean,
            'nested_std': nested_std,
            'best_params_per_fold': best_params_per_fold
        }


# ==========================================
# MODEL SELECTION AND COMPARISON FRAMEWORK
# ==========================================

class ModelSelectionFramework:
    """
    Comprehensive framework for comparing and selecting ML models
    """
    
    def __init__(self):
        self.model_results = {}
        
    def compare_multiple_models(self, models_dict, X, y, cv_folds=5):
        """Compare multiple models using cross-validation"""
        print(f"🏆 Model Comparison Framework")
        print(f"   Models to compare: {list(models_dict.keys())}")
        print(f"   Cross-validation: {cv_folds} folds")
        
        from sklearn.model_selection import cross_validate
        
        results = {}
        
        for model_name, model in models_dict.items():
            print(f"\n   Evaluating {model_name}...")
            
            # Perform cross-validation with multiple metrics
            cv_results = cross_validate(
                model, X, y, cv=cv_folds,
                scoring=['neg_mean_squared_error', 'r2'],
                return_train_score=True, n_jobs=-1
            )
            
            # Convert MSE to positive values
            train_mse = -cv_results['train_neg_mean_squared_error']
            test_mse = -cv_results['test_neg_mean_squared_error']
            train_r2 = cv_results['train_r2']
            test_r2 = cv_results['test_r2']
            
            results[model_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse_mean': np.mean(train_mse),
                'test_mse_mean': np.mean(test_mse),
                'train_r2_mean': np.mean(train_r2),
                'test_r2_mean': np.mean(test_r2),
                'test_mse_std': np.std(test_mse),
                'test_r2_std': np.std(test_r2)
            }
        
        # Visualize comparison
        self._plot_model_comparison(results)
        
        # Print summary
        print(f"\n📊 Model Comparison Summary:")
        print(f"{'Model':<20} {'Test MSE':<15} {'Test R²':<15} {'Overfitting':<15}")
        print(f"{'-'*65}")
        
        for model_name, result in results.items():
            test_mse = result['test_mse_mean']
            test_r2 = result['test_r2_mean']
            overfitting = result['train_mse_mean'] - result['test_mse_mean']
            
            print(f"{model_name:<20} {test_mse:<15.4f} {test_r2:<15.4f} {overfitting:<15.4f}")
        
        # Recommend best model
        best_model = min(results.keys(), key=lambda k: results[k]['test_mse_mean'])
        print(f"\n🏅 Recommended model: {best_model}")
        print(f"   Test MSE: {results[best_model]['test_mse_mean']:.4f} ± {results[best_model]['test_mse_std']:.4f}")
        print(f"   Test R²: {results[best_model]['test_r2_mean']:.4f} ± {results[best_model]['test_r2_std']:.4f}")
        
        self.model_results = results
        return results
    
    def _plot_model_comparison(self, results):
        """Plot comprehensive model comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(results.keys())
        
        # Plot 1: MSE comparison with error bars
        ax = axes[0, 0]
        train_mse_means = [results[name]['train_mse_mean'] for name in model_names]
        test_mse_means = [results[name]['test_mse_mean'] for name in model_names]
        test_mse_stds = [results[name]['test_mse_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, train_mse_means, width, label='Training MSE', alpha=0.7)
        ax.bar(x + width/2, test_mse_means, width, yerr=test_mse_stds, 
               label='Test MSE', alpha=0.7, capsize=5)
        
        ax.set_title('MSE Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Squared Error')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: R² comparison
        ax = axes[0, 1]
        train_r2_means = [results[name]['train_r2_mean'] for name in model_names]
        test_r2_means = [results[name]['test_r2_mean'] for name in model_names]
        test_r2_stds = [results[name]['test_r2_std'] for name in model_names]
        
        ax.bar(x - width/2, train_r2_means, width, label='Training R²', alpha=0.7)
        ax.bar(x + width/2, test_r2_means, width, yerr=test_r2_stds, 
               label='Test R²', alpha=0.7, capsize=5)
        
        ax.set_title('R² Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Overfitting analysis
        ax = axes[1, 0]
        overfitting = [results[name]['train_mse_mean'] - results[name]['test_mse_mean'] 
                      for name in model_names]
        
        bars = ax.bar(model_names, overfitting, alpha=0.7, 
                     color=['green' if x <= 0 else 'red' for x in overfitting])
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Overfitting Analysis\n(Training MSE - Test MSE)')
        ax.set_xlabel('Model')
        ax.set_ylabel('Overfitting Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, overfitting):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top')
        
        # Plot 4: Score distributions
        ax = axes[1, 1]
        for i, model_name in enumerate(model_names):
            test_scores = results[model_name]['test_mse']
            ax.hist(test_scores, bins=3, alpha=0.6, label=model_name, density=True)
        
        ax.set_title('Test Score Distributions')
        ax.set_xlabel('Test MSE')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def statistical_significance_testing(self, model1_scores, model2_scores, model1_name, model2_name):
        """Test statistical significance of difference between two models"""
        print(f"\n📊 Statistical Significance Testing")
        print(f"   Comparing: {model1_name} vs {model2_name}")
        
        from scipy import stats
        
        # Paired t-test (since we're using the same CV folds)
        statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        mean_diff = np.mean(model1_scores) - np.mean(model2_scores)
        
        print(f"\n📈 Results:")
        print(f"   Mean difference: {mean_diff:.4f}")
        print(f"   t-statistic: {statistic:.4f}")
        print(f"   p-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            better_model = model2_name if mean_diff > 0 else model1_name
            print(f"   ✅ Significant difference (p < {alpha})")
            print(f"   {better_model} is significantly better")
        else:
            print(f"   ❌ No significant difference (p >= {alpha})")
            print(f"   Models perform similarly")
        
        return {
            'mean_difference': mean_diff,
            't_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha
        }


# ==========================================
# COMPREHENSIVE SUPERVISED LEARNING DEMO
# ==========================================

def comprehensive_supervised_learning_demo():
    """
    Complete demonstration of supervised learning foundations
    """
    print("🎓 Neural Odyssey - Week 13: Supervised Learning Foundations")
    print("=" * 70)
    print("Mastering the Science of Predictive Learning")
    print("=" * 70)
    
    # Generate datasets for different demonstrations
    print("\n📊 Generating Demonstration Datasets...")
    
    # Dataset 1: Regression with known function (for bias-variance)
    bias_var_analyzer = BiasVarianceAnalyzer()
    X_reg, y_reg, y_true_reg, true_func = bias_var_analyzer.generate_true_function_data(
        n_samples=200, noise_level=0.15
    )
    
    # Dataset 2: Classification dataset
    X_class, y_class = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_redundant=2,
        n_classes=2, flip_y=0.1, random_state=42
    )
    
    print(f"   Regression dataset: {X_reg.shape}")
    print(f"   Classification dataset: {X_class.shape}")
    
    # ================================================================
    # BIAS-VARIANCE TRADEOFF ANALYSIS
    # ================================================================
    
    print("\n" + "="*70)
    print("🎯 BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("="*70)
    
    # Demonstrate bias-variance decomposition
    print("\n1️⃣  Basic Bias-Variance Decomposition")
    
    # Test simple linear regression
    from sklearn.linear_model import LinearRegression
    linear_results = bias_var_analyzer.bias_variance_decomposition(
        LinearRegression, {}, X_reg, y_reg, true_func, n_experiments=100
    )
    
    # Test polynomial regression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('linear', LinearRegression())
    ])
    
    class PolynomialRegressionWrapper:
        def __init__(self, degree=5):
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
        
        def fit(self, X, y):
            self.model.fit(X, y)
            return self
        
        def predict(self, X):
            return self.model.predict(X)
    
    poly_results = bias_var_analyzer.bias_variance_decomposition(
        PolynomialRegressionWrapper, {'degree': 8}, X_reg, y_reg, true_func, n_experiments=100
    )
    
    print("\n2️⃣  Model Complexity vs Bias-Variance")
    complexity_results = bias_var_analyzer.compare_model_complexity(X_reg, y_reg, true_func)
    
    print("\n3️⃣  Regularization Effects")
    regularization_results = bias_var_analyzer.demonstrate_regularization_effect(X_reg, y_reg, true_func)
    
    # ================================================================
    # OVERFITTING AND GENERALIZATION
    # ================================================================
    
    print("\n" + "="*70)
    print("📈 OVERFITTING AND GENERALIZATION ANALYSIS")
    print("="*70)
    
    overfitting_analyzer = OverfittingAnalyzer()
    
    print("\n1️⃣  Overfitting Progression")
    overfitting_results = overfitting_analyzer.demonstrate_overfitting_progression()
    
    print("\n2️⃣  Learning Curve Analysis")
    learning_curve_results = overfitting_analyzer.learning_curve_analysis(
        RandomForestRegressor, {'n_estimators': 50, 'random_state': 42}, X_reg, y_reg
    )
    
    print("\n3️⃣  Validation Curve Analysis")
    validation_curve_results = overfitting_analyzer.validation_curve_analysis(
        Ridge, 'alpha', np.logspace(-4, 2, 10), X_reg, y_reg
    )
    
    # ================================================================
    # CROSS-VALIDATION FRAMEWORK
    # ================================================================
    
    print("\n" + "="*70)
    print("🔄 CROSS-VALIDATION FRAMEWORK")
    print("="*70)
    
    cv_framework = CrossValidationFramework()
    
    print("\n1️⃣  K-Fold Implementation from Scratch")
    cv_results = cv_framework.cross_validate_from_scratch(
        LinearRegression, {}, X_reg, y_reg, k=5
    )
    
    print("\n2️⃣  Cross-Validation Strategy Comparison")
    cv_comparison = cv_framework.compare_cv_strategies(
        X_reg, y_reg, LinearRegression, {}
    )
    
    print("\n3️⃣  Nested Cross-Validation")
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    nested_cv_results = cv_framework.nested_cross_validation(
        X_reg, y_reg, Ridge, param_grid
    )
    
    # ================================================================
    # MODEL SELECTION AND COMPARISON
    # ================================================================
    
    print("\n" + "="*70)
    print("🏆 MODEL SELECTION AND COMPARISON")
    print("="*70)
    
    model_selection = ModelSelectionFramework()
    
    print("\n1️⃣  Multiple Model Comparison")
    
    # Define models to compare
    models_to_compare = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    model_comparison_results = model_selection.compare_multiple_models(
        models_to_compare, X_reg, y_reg, cv_folds=5
    )
    
    print("\n2️⃣  Statistical Significance Testing")
    
    # Compare top two models
    model_names = list(model_comparison_results.keys())
    sorted_models = sorted(model_names, 
                          key=lambda k: model_comparison_results[k]['test_mse_mean'])
    
    if len(sorted_models) >= 2:
        best_model = sorted_models[0]
        second_model = sorted_models[1]
        
        significance_test = model_selection.statistical_significance_testing(
            model_comparison_results[best_model]['test_mse'],
            model_comparison_results[second_model]['test_mse'],
            best_model, second_model
        )
    
    # ================================================================
    # FINAL INTEGRATION AND INSIGHTS
    # ================================================================
    
    print("\n" + "="*70)
    print("🎓 SUPERVISED LEARNING MASTERY SUMMARY")
    print("="*70)
    
    key_insights = [
        "🎯 Bias-Variance Tradeoff: Simple models have high bias, complex models have high variance",
        "📈 Overfitting occurs when model complexity exceeds data complexity",
        "🔄 Cross-validation provides unbiased performance estimates",
        "🏆 Model selection requires balancing performance and generalization",
        "📊 Statistical testing validates model comparison conclusions"
    ]
    
    print("\n💡 Key Insights Mastered:")
    for insight in key_insights:
        print(f"   {insight}")
    
    practical_guidelines = [
        "Always use cross-validation for model evaluation",
        "Monitor both training and validation error",
        "Use learning curves to diagnose bias vs variance",
        "Apply regularization to control overfitting",
        "Test statistical significance of model differences",
        "Choose simplest model within one standard error of best"
    ]
    
    print(f"\n📋 Practical Guidelines:")
    for guideline in practical_guidelines:
        print(f"   ✅ {guideline}")
    
    next_week_preview = [
        "Linear models deep dive (Ridge, Lasso, Elastic Net)",
        "Feature selection and regularization paths",
        "Generalized linear models",
        "Logistic regression extensions"
    ]
    
    print(f"\n🔮 Next Week Preview:")
    for topic in next_week_preview:
        print(f"   📚 {topic}")
    
    return {
        'bias_variance': {
            'analyzer': bias_var_analyzer,
            'linear_results': linear_results,
            'poly_results': poly_results,
            'complexity_results': complexity_results,
            'regularization_results': regularization_results
        },
        'overfitting': {
            'analyzer': overfitting_analyzer,
            'progression': overfitting_results,
            'learning_curves': learning_curve_results,
            'validation_curves': validation_curve_results
        },
        'cross_validation': {
            'framework': cv_framework,
            'results': cv_results,
            'comparison': cv_comparison,
            'nested_cv': nested_cv_results
        },
        'model_selection': {
            'framework': model_selection,
            'comparison': model_comparison_results,
            'significance_test': significance_test if 'significance_test' in locals() else None
        }
    }


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    """
    Run this file to master supervised learning foundations!
    
    This comprehensive exploration covers:
    1. Bias-variance tradeoff analysis and decomposition
    2. Overfitting, underfitting, and generalization
    3. Cross-validation implementation and strategies
    4. Model selection and comparison frameworks
    5. Statistical significance testing
    
    To get started, run: python exercises.py
    """
    
    print("🚀 Welcome to Neural Odyssey - Week 13: Supervised Learning Foundations!")
    print("Master the fundamental principles that govern all predictive learning.")
    print("\nThis foundational week includes:")
    print("1. 🎯 Bias-variance tradeoff analysis and decomposition")
    print("2. 📈 Overfitting progression and learning curve analysis")
    print("3. 🔄 Cross-validation implementation from scratch")
    print("4. 🏆 Model selection and comparison frameworks")
    print("5. 📊 Statistical significance testing")
    print("6. 💡 Practical guidelines for real-world ML")
    
    # Run comprehensive demonstration
    print("\n" + "="*70)
    print("🎭 Starting Supervised Learning Foundations Journey...")
    print("="*70)
    
    # Execute complete demonstration
    results = comprehensive_supervised_learning_demo()
    
    print("\n" + "="*70)
    print("🎉 WEEK 13 COMPLETE: SUPERVISED LEARNING FOUNDATIONS MASTERED!")
    print("="*70)
    
    print(f"\n🏆 Achievement Summary:")
    print(f"   ✅ Bias-variance tradeoff: Understood and implemented")
    print(f"   ✅ Overfitting analysis: Complete diagnostic framework")
    print(f"   ✅ Cross-validation: From scratch implementation")
    print(f"   ✅ Model selection: Comprehensive comparison framework")
    print(f"   ✅ Statistical testing: Significance validation")
    
    print(f"\n🧠 Core Concepts Mastered:")
    core_concepts = [
        "Bias² + Variance + Noise = Total Error",
        "Model complexity controls bias-variance tradeoff", 
        "Cross-validation provides unbiased performance estimates",
        "Learning curves diagnose underfitting vs overfitting",
        "Regularization reduces variance at cost of increased bias"
    ]
    
    for concept in core_concepts:
        print(f"   💡 {concept}")
    
    print(f"\n🚀 Ready for Week 14: Linear Models Deep Dive!")
    print(f"   Building on these foundations to master Ridge, Lasso, and beyond")
    
    return results